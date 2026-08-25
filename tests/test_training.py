import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from train import InfiniteTalkDataset, InfiniteTalkTrain, validate_training_batch


def make_batch(batch_size=1, latent_frames=5, height=4, width=4, device="cpu"):
    human = torch.zeros(height, width)
    human[1:3, 1:3] = 1
    masks = torch.stack([human, torch.zeros_like(human), 1 - human])
    audio_frames = 1 + (latent_frames - 1) * 4
    return {
        "latents": torch.randn(batch_size, 16, latent_frames, height, width, device=device),
        "context": torch.randn(batch_size, 6, 4096, device=device),
        "clip_fea": torch.randn(batch_size, 257, 1280, device=device),
        "y": torch.randn(batch_size, 20, latent_frames, height, width, device=device),
        "audio": torch.randn(batch_size, audio_frames, 5, 12, 768, device=device),
        "min_padding_len": torch.zeros(batch_size, dtype=torch.long, device=device),
        "ref_target_masks": masks.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device),
    }


class TrainingContractTest(unittest.TestCase):
    def test_reference_mask_uses_latent_frame_count(self):
        mask = InfiniteTalkDataset._build_mask(5, 3, 4)
        self.assertEqual(tuple(mask.shape), (4, 5, 3, 4))
        self.assertTrue(torch.equal(mask[:, 0], torch.ones(4, 3, 4)))
        self.assertEqual(mask[:, 1:].count_nonzero().item(), 0)

    def test_face_box_at_edge_has_no_safe_crop_padding(self):
        dataset = InfiniteTalkDataset.__new__(InfiniteTalkDataset)
        landmarks = np.zeros((17, 203, 2), dtype=np.float32)
        landmarks[:, :, 0] = 2
        landmarks[:, :, 1] = 2
        padding, masks = dataset.get_target_mask(landmarks, 64, 64)
        self.assertEqual(padding, 0)
        self.assertEqual(tuple(masks.shape), (3, 8, 8))

    def test_batch_contract_accepts_single_sample(self):
        self.assertEqual(validate_training_batch(make_batch()), (1, 16, 5, 4, 4))

    def test_batch_contract_rejects_batch_size_two(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            validate_training_batch(make_batch(batch_size=2))

    def test_dataset_rejects_mismatched_audio_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data"
            share = root / "share"
            data.mkdir()
            share.mkdir()
            torch.save(torch.randn(6, 4096), share / "context.pt")
            torch.save(torch.randn(16, 5, 4, 4), data / "latents_001.pth")
            torch.save(torch.randn(16, 5, 4, 4), data / "latent_y_001.pth")
            torch.save(torch.randn(16, 5, 12, 768), data / "audio_001.pth")
            torch.save({"ref_context": torch.randn(257, 1280)}, data / "ref_context_001.pth")
            np.save(data / "lmk203_001.npy", np.zeros((17, 203, 2), dtype=np.float32))
            dataset = InfiniteTalkDataset(str(data), str(share))
            with self.assertRaisesRegex(ValueError, "audio must have shape"):
                dataset[0]

    @unittest.skipUnless(
        os.environ.get("RUN_GPU_TRAINING_SMOKE") == "1" and torch.cuda.is_available(),
        "set RUN_GPU_TRAINING_SMOKE=1 on a CUDA host",
    )
    def test_tiny_training_step_forward_and_backward(self):
        config = {
            "model_type": "i2v",
            "patch_size": (1, 2, 2),
            "text_len": 8,
            "in_dim": 36,
            "dim": 24,
            "ffn_dim": 48,
            "freq_dim": 16,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "window_size": (-1, -1),
            "qk_norm": True,
            "cross_attn_norm": True,
            "output_dim": 12,
            "intermediate_dim": 16,
            "context_tokens": 4,
            "vae_scale": 4,
        }
        with patch.object(InfiniteTalkTrain, "load_model", return_value=None):
            module = InfiniteTalkTrain(config, "unused", None, "unused")
        module.log_dict = lambda *args, **kwargs: None
        module = module.cuda()
        random.seed(0)
        loss = module.training_step(make_batch(device="cuda"), 0)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
