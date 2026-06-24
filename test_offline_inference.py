from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from observation_adapter import JOINT_NAMES  # noqa: E402
from offline_inference import (  # noqa: E402
    denormalize_actions,
    load_action_statistics,
    save_actions,
    save_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "data" / "so101-megamix-v1" / "meta" / "stats.json"


class OfflineInferenceHelpersTest(unittest.TestCase):
    def test_action_denormalization_endpoints(self) -> None:
        q01, q99 = load_action_statistics(STATS_PATH)
        normalized = torch.stack(
            [
                torch.cat([-torch.ones(6), torch.zeros(26)]),
                torch.cat([torch.zeros(6), torch.zeros(26)]),
                torch.cat([torch.ones(6), torch.zeros(26)]),
            ]
        ).unsqueeze(0)
        actions = denormalize_actions(normalized, q01, q99)

        torch.testing.assert_close(actions[0], q01)
        torch.testing.assert_close(actions[1], (q01 + q99) / 2)
        torch.testing.assert_close(actions[2], q99)

    def test_action_and_frame_serialization(self) -> None:
        actions = torch.arange(24 * 6, dtype=torch.float32).reshape(24, 6)
        frames = np.zeros((9, 16, 32, 3), dtype=np.uint8)
        frames[:, :, :, 1] = 127

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            actions_path = save_actions(actions, output_dir)
            frame_paths = save_frames(frames, output_dir)

            self.assertTrue(actions_path.is_file())
            lines = actions_path.read_text().splitlines()
            self.assertEqual(len(lines), 25)
            self.assertEqual(lines[0].split(","), ["step", *JOINT_NAMES])
            self.assertEqual(len(frame_paths), 9)
            self.assertTrue(all(path.is_file() for path in frame_paths))
            for camera in ("front", "top", "gripper"):
                camera_paths = sorted((output_dir / "frames" / camera).glob("*.png"))
                self.assertEqual(len(camera_paths), 9)


if __name__ == "__main__":
    unittest.main()
