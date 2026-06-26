from pathlib import Path
import sys
import unittest

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from observation_adapter import (  # noqa: E402
    JOINT_NAMES,
    SO101_ACTION_CATEGORY_ID,
    SO101ObservationAdapter,
)


class ObservationAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = SO101ObservationAdapter()

    def test_q99_normalization_and_padding(self) -> None:
        state_at_q01, mask = self.adapter.normalize_and_pad_state(self.adapter.state_q01)
        state_at_q99, _ = self.adapter.normalize_and_pad_state(self.adapter.state_q99)
        midpoint = (self.adapter.state_q01 + self.adapter.state_q99) / 2
        state_at_midpoint, _ = self.adapter.normalize_and_pad_state(midpoint)

        self.assertEqual(len(JOINT_NAMES), 6)
        torch.testing.assert_close(state_at_q01[0, 0, :6], -torch.ones(6))
        torch.testing.assert_close(state_at_q99[0, 0, :6], torch.ones(6))
        torch.testing.assert_close(state_at_midpoint[0, 0, :6], torch.zeros(6))
        self.assertEqual(state_at_q01.shape, (1, 1, 64))
        self.assertEqual(torch.count_nonzero(state_at_q01[0, 0, 6:]).item(), 0)
        self.assertTrue(mask[0, 0, :6].all().item())
        self.assertFalse(mask[0, 0, 6:].any().item())

    def test_complete_batch_contract_and_deterministic_seed(self) -> None:
        source_image = np.zeros((480, 640, 3), dtype=np.uint8)
        joints = (self.adapter.state_q01 + self.adapter.state_q99) / 2

        first = self.adapter.adapt(
            image=source_image,
            prompt="Pick the red cube",
            joint_positions=joints,
        )
        second = self.adapter.adapt(
            image=source_image,
            prompt="Pick the red cube",
            joint_positions=joints,
        )

        self.assertEqual(first["images"].shape, (1, 1, 352, 640, 3))
        self.assertEqual(first["images"].dtype, torch.uint8)
        self.assertEqual(first["state"].shape, (1, 1, 64))
        self.assertEqual(first["text"].shape, (1, 512))
        self.assertEqual(first["text_attention_mask"].shape, (1, 512))
        self.assertEqual(first["text_negative"].shape, (1, 512))
        self.assertEqual(first["text_attention_mask_negative"].shape, (1, 512))
        self.assertEqual(first["action_noise"].shape, (1, 24, 32))
        self.assertEqual(first["action_noise"].dtype, torch.bfloat16)
        self.assertEqual(first["action_seed"].item(), 1140)
        self.assertEqual(first["embodiment_id"].item(), SO101_ACTION_CATEGORY_ID)
        self.assertEqual(SO101_ACTION_CATEGORY_ID, 0)
        torch.testing.assert_close(first["action_noise"], second["action_noise"])

    def test_three_camera_mosaic_layout(self) -> None:
        front = np.full((10, 20, 3), (10, 20, 30), dtype=np.uint8)
        top = np.full((10, 20, 3), (40, 50, 60), dtype=np.uint8)
        gripper = np.full((10, 20, 3), (70, 80, 90), dtype=np.uint8)
        batch = self.adapter.adapt_camera_views(
            images={"front": front, "top": top, "gripper": gripper},
            prompt="Pick the blue cube",
            joint_positions=torch.zeros(6),
        )

        mosaic = batch["images"][0, 0]
        self.assertEqual(tuple(mosaic.shape), (352, 640, 3))
        # WanVAE downsamples by 8 and the DiT patchifies by 2x2:
        # (352 / 16) * (640 / 16) = the checkpoint's frame_seqlen 880.
        self.assertEqual((mosaic.shape[0] // 16) * (mosaic.shape[1] // 16), 880)
        self.assertEqual(mosaic[0, 0].tolist(), [10, 20, 30])
        self.assertEqual(mosaic[0, 320].tolist(), [40, 50, 60])
        self.assertEqual(mosaic[176, 0].tolist(), [70, 80, 90])
        self.assertEqual(mosaic[176, 320].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
