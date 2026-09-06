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
                 kernel_size=3, reference_count=0):
        super().__init__()
        features = features or [(4296, 10, 130), (1024, 10, 128), (256, 10, 128)]
        features = [tuple(shape) for shape in features]
        self.config = dict(features=features, codebook_size=codebook_size, decay=decay,
                           commitment=commitment, kernel_size=kernel_size)
        self.reference_count = reference_count
        if reference_count:
            self.config['reference_count'] = reference_count
            self.register_buffer('reference_tokens', torch.zeros(reference_count, *features[0]))
            self.register_buffer('reference_scale', torch.ones(reference_count))
            self.register_buffer('references_initialized', torch.tensor(False))
        self.input_shape, self.latent_shape = features[0], features[-1]
        self.encoder = HyperConvBlock(features, kernel_size)
        self.codebook = EMACodebook(codebook_size, features[-1][-1], decay, commitment)
        self.decoder = HyperConvBlock(list(reversed(features)), kernel_size)

    def residual_input(self, tokens):
        clean = torch.nan_to_num(tokens, nan=0.0)
        if not self.reference_count:
            return clean, None
        if not self.references_initialized.item():
            raise RuntimeError('Training references have not been initialized')
        # Explicit differences avoid cancellation in ||x||² + ||c||² - 2 x.c.
        with torch.no_grad():
            distances = torch.stack([(clean - ref).square().flatten(1).mean(1)
                                     for ref in self.reference_tokens], dim=1)
            reference = distances.argmin(1)
        scale = self.reference_scale[reference].view(-1, 1, 1, 1)
        return (clean - self.reference_tokens[reference]) / scale, reference

    def restore_reference(self, prediction, reference):
        if reference is None:
            return prediction
        return (prediction * self.reference_scale[reference].view(-1, 1, 1, 1)
                + self.reference_tokens[reference])

    def forward(self, tokens):
        if tuple(tokens.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape}], got {tokens.shape}")
        residual, reference = self.residual_input(tokens)
        latent = self.encoder(residual)
        quantized, codes, commitment, perplexity = self.codebook(latent)
        if reference is not None:
            codes = torch.cat([reference[:, None], codes.flatten(1)], dim=1)
        return self.restore_reference(self.decoder(quantized), reference), codes, commitment, perplexity

    @torch.no_grad()
    def encode(self, tokens):
        if self.training:
            raise RuntimeError("Call eval() before exporting codes")
        residual, reference = self.residual_input(tokens)
        codes = self.codebook(self.encoder(residual))[1]
        return torch.cat([reference[:, None], codes.flatten(1)], dim=1) if reference is not None else codes

    def decode(self, codes):
        reference = None
        if self.reference_count:
            expected = self.latent_shape[0] * self.latent_shape[1] + 1
            if codes.ndim != 2 or codes.shape[1] != expected:
                raise ValueError(f'Expected {expected} reference/residual codes')
            reference, codes = codes[:, 0].long(), codes[:, 1:]
            if reference.min() < 0 or reference.max() >= self.reference_count:
                raise ValueError('Reference index out of range')
            codes = codes.reshape(-1, *self.latent_shape[:-1])
        if tuple(codes.shape[1:]) != self.latent_shape[:-1]:
            raise ValueError(f"Unexpected code shape: {codes.shape}")
        if codes.min() < 0 or codes.max() >= self.codebook.size:
            raise ValueError("Codebook index out of range")
        return self.restore_reference(self.decoder(F.embedding(codes.long(), self.codebook.embedding)), reference)


def reconstruction_loss(prediction, target, weights):
    valid = torch.isfinite(target)
    clean = torch.nan_to_num(target.float(), nan=0.0)
    error = (prediction.float() - clean).square()
    weighted = error * weights.float().view(1, -1, 1, 1)
    return weighted[valid].mean(), error[valid].mean()
