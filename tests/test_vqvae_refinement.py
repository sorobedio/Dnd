import tempfile
import unittest
from pathlib import Path

import torch

from workspace.vqvae.train_reconstruction import physical_reconstruction, token_scales, training_manifest
from workspace.vqvae.model import LoRAVQVAE
from workspace.vqvae.audit_reconstruction import weight_errors


class RefinementTests(unittest.TestCase):
    def test_delta_error_matches_explicit_matrix_product(self):
        torch.manual_seed(23)
        a, b = torch.randn(3, 7), torch.randn(5, 3)
        ah, bh = a + .1 * torch.randn_like(a), b + .1 * torch.randn_like(b)
        result = weight_errors({'lora_A.weight': a, 'lora_B.weight': b},
                               {'lora_A.weight': ah, 'lora_B.weight': bh})
        expected = ((bh.double() @ ah.double() - b.double() @ a.double()).norm()
                    / (b.double() @ a.double()).norm()).item()
        self.assertAlmostEqual(result['delta_BA_relative_l2'], expected, places=10)

    def test_reference_codes_decode_without_original_or_task_label(self):
        model = LoRAVQVAE(features=[(12, 4, 8), (4, 2, 4)], codebook_size=8, reference_count=2)
        model.reference_tokens[0].fill_(-2.)
        model.reference_tokens[1].fill_(2.)
        model.reference_scale.fill_(.1)
        model.references_initialized.fill_(True)
        x = model.reference_tokens.clone() + .03 * torch.randn_like(model.reference_tokens)
        model(x)
        model.eval()
        with torch.no_grad():
            prediction, codes, _, _ = model(x)
            self.assertEqual(codes.shape, (2, 9))
            self.assertEqual(codes[:,0].tolist(), [0,1])
            torch.testing.assert_close(model.decode(codes), prediction)
        restored = LoRAVQVAE(**model.config)
        restored.load_state_dict(model.state_dict())
        restored.eval()
        torch.testing.assert_close(restored.decode(codes), prediction)

    def test_relative_weight_loss_and_gradient(self):
        target = torch.zeros(1, 2, 4, 8)
        std, mean = 0.003, 0.002
        pair = torch.tensor([mean * 8, torch.log(torch.tensor(std * .9 + .1)) + 1.6])
        target[:, :, -2:, :] = pair.repeat(4)
        target[:, :, :, -2:] = pair
        target[:, :, :2, :6] = 1.
        target[:, 0, 0, 0] = float('nan')
        m, s = token_scales(target)
        torch.testing.assert_close(m, torch.full_like(m, mean))
        torch.testing.assert_close(s, torch.full_like(s, std), atol=1e-7, rtol=1e-5)
        prediction = torch.nan_to_num(target, nan=0.).clone()
        torch.testing.assert_close(physical_reconstruction(prediction, target), torch.tensor(0.))
        prediction[:, :, :2, :6] += 1.
        prediction.requires_grad_()
        loss = physical_reconstruction(prediction, target)
        torch.testing.assert_close(loss, torch.tensor(1.))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_arc_c_promoted_and_latest_added_without_test_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for task, split in [('ARC-e', 'train'), ('ARC-c', 'held_out')]:
                (root / task).mkdir()
                for number in [1, 9]:
                    (root / task / f'{number}.safetensors').write_text(str(number))
                entries.append(dict(dataset=task, split=split, path=str(root / task / '1.safetensors')))
            source = dict(entries=entries, settings=dict(datasets=['ARC-e'], dataset_tag='ARC-c'))
            result = training_manifest(source)
            self.assertEqual(len(result['entries']), 4)
            self.assertTrue(all(e['split'] == 'train' for e in result['entries']))
            self.assertEqual(result['tokenization_dtype'], 'float32')
            self.assertEqual(source['entries'][1]['split'], 'held_out')


if __name__ == '__main__':
    unittest.main()
