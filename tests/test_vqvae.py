import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from workspace.vqvae.model import LoRAVQVAE, reconstruction_loss
from workspace.vqvae.data import build_manifest, dnd_settings


class VQVAETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.model = LoRAVQVAE(features=[(12, 4, 8), (8, 4, 8), (4, 2, 4)],
                              codebook_size=8, kernel_size=3)

    def test_gradients_codes_and_decode_roundtrip(self):
        target = torch.randn(2, 12, 4, 8)
        target[:, 0, 0, 0] = torch.nan
        prediction, codes, commitment, perplexity = self.model(target)
        loss, _ = reconstruction_loss(prediction, target, torch.ones(12))
        (loss + commitment).backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(codes.shape, (2, 4, 2))
        self.assertGreaterEqual(codes.min().item(), 0)
        self.assertLess(codes.max().item(), 8)
        self.assertGreaterEqual(perplexity.item(), 1)
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.model.eval()
        before = {key: value.clone() for key, value in self.model.codebook.state_dict().items()}
        with torch.no_grad():
            prediction, codes, _, _ = self.model(target)
            torch.testing.assert_close(self.model.decode(self.model.encode(target)), prediction)
        for key, value in before.items():
            torch.testing.assert_close(value, self.model.codebook.state_dict()[key])
        restored = LoRAVQVAE(**self.model.config)
        restored.load_state_dict(self.model.state_dict())
        restored.eval()
        torch.testing.assert_close(restored.encode(target), codes)

    def test_weighted_loss_ignores_padding(self):
        target = torch.tensor([[[[1.0, float("nan")]], [[2.0, 4.0]]]])
        prediction = torch.zeros_like(target)
        weighted, plain = reconstruction_loss(prediction, target, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(weighted, torch.tensor((2.0 + 12.0 + 48.0) / 3))
        torch.testing.assert_close(plain, torch.tensor((1.0 + 4.0 + 16.0) / 3))

    def test_export_requires_initialized_eval_codebook(self):
        self.model.eval()
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            self.model.encode(torch.randn(1, 12, 4, 8))

    def test_manifest_freezes_dnd_selection_and_keeps_arc_c_held_out(self):
        settings = dnd_settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in settings["datasets"] + [settings["dataset_tag"]]:
                (root / dataset).mkdir()
                for index in range(51):
                    (root / dataset / f"{index:03d}.safetensors").write_text(str(index))
            listdir = os.listdir
            with patch("workspace.vqvae.data.os.listdir",
                       side_effect=lambda folder: sorted(listdir(folder), reverse=True)):
                manifest = build_manifest(root)
            train = [e for e in manifest["entries"] if e["split"] == "train"]
            held_out = [e for e in manifest["entries"] if e["split"] == "held_out"]
            self.assertEqual(len(train), 200)
            self.assertEqual(len(held_out), 50)
            self.assertEqual(Path(train[0]["path"]).name, "050.safetensors")
            self.assertEqual(Path(train[49]["path"]).name, "001.safetensors")
            self.assertTrue(all(e["dataset"] != "ARC-c" for e in train))
            self.assertTrue(all(e["dataset"] == "ARC-c" for e in held_out))
            # Checkpoint resume compares the JSON manifest with saved state.
            self.assertEqual(manifest, json.loads(json.dumps(manifest)))


if __name__ == "__main__":
    unittest.main()
