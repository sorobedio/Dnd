"""Hyper-convolutional VQ-VAE; each sample is one complete LoRA adapter."""
import torch
from torch import nn
from torch.nn import functional as F

from workspace.dnd.module.hyperconv import HyperConvBlock


class EMACodebook(nn.Module):
    def __init__(self, size=1024, dim=128, decay=0.99, commitment=0.25):
        super().__init__()
        if size < 1 or dim < 1 or not 0 <= decay < 1 or commitment < 0:
            raise ValueError("Invalid codebook size, dimension, EMA decay, or commitment weight")
        self.size, self.dim, self.decay, self.commitment = size, dim, decay, commitment
        self.register_buffer("embedding", torch.randn(size, dim))
        self.register_buffer("count", torch.ones(size))
        self.register_buffer("total", self.embedding.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, flat):
        indices = torch.randperm(len(flat), device=flat.device)
        indices = indices.repeat((self.size + len(flat) - 1) // len(flat))[:self.size]
        self.embedding.copy_(flat[indices])
        self.total.copy_(self.embedding)
        self.count.fill_(1)
        self.initialized.fill_(True)

    def forward(self, z):
        # The last axis is one codebook vector; the first axis is checkpoint batch.
        with torch.autocast(device_type=z.device.type, enabled=False):
            flat = z.float().reshape(-1, self.dim)
            if self.training and not self.initialized.item():
                self._initialize(flat.detach())
            if not self.initialized.item():
                raise RuntimeError("Codebook has not been trained/initialized")
            indices = []
            with torch.no_grad():
                for chunk in flat.detach().split(1024):
                    distance = (chunk.square().sum(1, keepdim=True)
                                + self.embedding.square().sum(1)[None]
                                - 2 * chunk @ self.embedding.T)
                    indices.append(distance.argmin(1))
            indices = torch.cat(indices)
            quantized = F.embedding(indices, self.embedding).view_as(z)
            commitment = self.commitment * F.mse_loss(z.float(), quantized.detach())
            counts = torch.bincount(indices, minlength=self.size).float()
            if self.training:
                with torch.no_grad():
                    sums = torch.zeros_like(self.total).index_add_(0, indices, flat.detach())
                    self.count.lerp_(counts, 1 - self.decay)
                    self.total.lerp_(sums, 1 - self.decay)
                    self.embedding.copy_(self.total / self.count.clamp_min(1e-5)[:, None])
            probabilities = counts / counts.sum()
            perplexity = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        straight_through = z + (quantized.to(z.dtype) - z).detach()
        return straight_through, indices.view(z.shape[:-1]), commitment, perplexity


class LoRAVQVAE(nn.Module):
    def __init__(self, features=None, codebook_size=1024, decay=0.99, commitment=0.25,
                 kernel_size=3):
        super().__init__()
        features = features or [(4296, 10, 130), (1024, 10, 128), (256, 10, 128)]
        features = [tuple(shape) for shape in features]
        self.config = dict(features=features, codebook_size=codebook_size, decay=decay,
                           commitment=commitment, kernel_size=kernel_size)
        self.input_shape, self.latent_shape = features[0], features[-1]
        self.encoder = HyperConvBlock(features, kernel_size)
        self.codebook = EMACodebook(codebook_size, features[-1][-1], decay, commitment)
        self.decoder = HyperConvBlock(list(reversed(features)), kernel_size)

    def forward(self, tokens):
        if tuple(tokens.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape}], got {tokens.shape}")
        latent = self.encoder(torch.nan_to_num(tokens, nan=0.0))
        quantized, codes, commitment, perplexity = self.codebook(latent)
        return self.decoder(quantized), codes, commitment, perplexity

    @torch.no_grad()
    def encode(self, tokens):
        if self.training:
            raise RuntimeError("Call eval() before exporting codes")
        return self.codebook(self.encoder(torch.nan_to_num(tokens, nan=0.0)))[1]

    def decode(self, codes):
        if tuple(codes.shape[1:]) != self.latent_shape[:-1]:
            raise ValueError(f"Unexpected code shape: {codes.shape}")
        if codes.min() < 0 or codes.max() >= self.codebook.size:
            raise ValueError("Codebook index out of range")
        return self.decoder(F.embedding(codes.long(), self.codebook.embedding))


def reconstruction_loss(prediction, target, weights):
    valid = torch.isfinite(target)
    clean = torch.nan_to_num(target.float(), nan=0.0)
    error = (prediction.float() - clean).square()
    weighted = error * weights.float().view(1, -1, 1, 1)
    return weighted[valid].mean(), error[valid].mean()
