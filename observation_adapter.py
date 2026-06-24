#!/usr/bin/env python3
"""Convert an SO-101 observation into DreamZero action-head inputs.

The checkpoint was trained on a 2x2 camera mosaic: front at top-left, top at
top-right, gripper at bottom-left, and a black bottom-right quadrant. Missing
cameras are represented by black quadrants.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_PATH = PROJECT_ROOT / "data" / "so101-megamix-v1" / "meta" / "stats.json"
DEFAULT_INFO_PATH = PROJECT_ROOT / "data" / "so101-megamix-v1" / "meta" / "info.json"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "checkpoints" / "umt5-xxl"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CAMERA_NAMES = ("front", "top", "gripper")

# The checkpoint tensors have category dimension 1, and CausalWanModel maps
# actions/states to category zero during its forward pass. This is deliberately
# not a global DreamZero registry/projector ID.
SO101_ACTION_CATEGORY_ID = 0

DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, "
    "artwork, painting, image, still, grayscale, dull, worst quality, low quality, "
    "JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, "
    "disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, "
    "three legs, many people in the background, walking backwards."
)

ImageInput: TypeAlias = str | Path | Image.Image | np.ndarray | torch.Tensor


class SO101ObservationAdapter:
    """Prepare single-frame SO-101 observations for ``WANPolicyHead`` inference."""

    def __init__(
        self,
        *,
        stats_path: str | Path = DEFAULT_STATS_PATH,
        info_path: str | Path = DEFAULT_INFO_PATH,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        camera_height: int = 176,
        camera_width: int = 320,
        max_length: int = 512,
        max_state_dim: int = 64,
        action_horizon: int = 24,
        action_dim: int = 32,
        action_seed: int = 1140,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.stats_path = Path(stats_path).expanduser().resolve()
        self.info_path = Path(info_path).expanduser().resolve()
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_length = max_length
        self.max_state_dim = max_state_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_seed = action_seed

        if self.max_state_dim < len(JOINT_NAMES):
            raise ValueError(f"max_state_dim must be at least {len(JOINT_NAMES)}")
        if self.action_dim < len(JOINT_NAMES):
            raise ValueError(f"action_dim must be at least {len(JOINT_NAMES)}")

        self._state_q01, self._state_q99 = self._load_and_validate_metadata()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
        )

    def _load_and_validate_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        with self.stats_path.open() as stats_file:
            stats = json.load(stats_file)
        with self.info_path.open() as info_file:
            info = json.load(info_file)

        state_stats = stats.get("observation.state")
        if state_stats is None:
            raise KeyError(f"observation.state is missing from {self.stats_path}")
        missing_stats = [name for name in ("q01", "q99") if name not in state_stats]
        if missing_stats:
            raise KeyError(f"Missing state statistics in {self.stats_path}: {missing_stats}")

        q01 = torch.tensor(state_stats["q01"], dtype=torch.float32)
        q99 = torch.tensor(state_stats["q99"], dtype=torch.float32)
        if q01.shape != (len(JOINT_NAMES),) or q99.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected six q01/q99 values, got q01={tuple(q01.shape)}, "
                f"q99={tuple(q99.shape)}"
            )
        if torch.any(q99 <= q01):
            raise ValueError("Every observation.state q99 value must be greater than q01")

        features = info.get("features", {})
        state_feature = features.get("observation.state")
        if state_feature is None:
            raise KeyError(f"observation.state feature is missing from {self.info_path}")
        metadata_names = tuple(name.removesuffix(".pos") for name in state_feature["names"])
        if metadata_names != JOINT_NAMES:
            raise ValueError(
                "SO-101 state order does not match the model contract: "
                f"expected {JOINT_NAMES}, found {metadata_names}"
            )
        missing_cameras = [
            name for name in CAMERA_NAMES if f"observation.images.{name}" not in features
        ]
        if missing_cameras:
            raise KeyError(f"Dataset metadata is missing camera streams: {missing_cameras}")
        return q01, q99

    @property
    def state_q01(self) -> torch.Tensor:
        return self._state_q01.clone()

    @property
    def state_q99(self) -> torch.Tensor:
        return self._state_q99.clone()

    def normalize_and_pad_state(
        self, joint_positions: Sequence[float] | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize six degree-valued joints to [-1, 1] and zero-pad to 64."""

        joints = torch.as_tensor(joint_positions, dtype=torch.float32).flatten()
        if joints.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected six joints in order {JOINT_NAMES}; got shape {tuple(joints.shape)}"
            )
        if not torch.isfinite(joints).all():
            raise ValueError("Joint positions must all be finite")

        normalized = 2.0 * (joints - self._state_q01) / (
            self._state_q99 - self._state_q01
        ) - 1.0
        normalized = normalized.clamp(-1.0, 1.0)

        state = torch.zeros((1, 1, self.max_state_dim), dtype=torch.float32)
        state[0, 0, : len(JOINT_NAMES)] = normalized
        state_mask = torch.zeros((1, 1, self.max_state_dim), dtype=torch.bool)
        state_mask[0, 0, : len(JOINT_NAMES)] = True
        return state, state_mask

    def prepare_camera_image(self, image: ImageInput) -> np.ndarray:
        """Resize one RGB camera frame to uint8 ``[176,320,3]``."""
        pil_image = self._to_pil_rgb(image)
        pil_image = pil_image.resize(
            (self.camera_width, self.camera_height),
            resample=Image.Resampling.BILINEAR,
        )
        return np.array(pil_image, dtype=np.uint8, copy=True)

    def prepare_camera_mosaic(self, images: Mapping[str, ImageInput]) -> torch.Tensor:
        """Return the trained 2x2 mosaic as ``[1,1,352,640,3]`` uint8."""

        unknown = sorted(set(images) - set(CAMERA_NAMES))
        if unknown:
            raise ValueError(f"Unknown SO-101 camera names: {unknown}; expected {CAMERA_NAMES}")
        if not images:
            raise ValueError("At least one camera image is required")

        height, width = self.camera_height, self.camera_width
        mosaic = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
        placements = {
            "front": (slice(0, height), slice(0, width)),
            "top": (slice(0, height), slice(width, 2 * width)),
            "gripper": (slice(height, 2 * height), slice(0, width)),
        }
        for name, image in images.items():
            rows, columns = placements[name]
            mosaic[rows, columns] = self.prepare_camera_image(image)
        return torch.from_numpy(mosaic).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _to_pil_rgb(image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"Tensor image must be rank 3, got {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        elif isinstance(image, np.ndarray):
            array = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Array image must have shape [H,W,C], got {array.shape}")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("Image contains non-finite values")
            if array.size and array.min() >= 0.0 and array.max() <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        mode = "RGBA" if array.shape[-1] == 4 else "RGB"
        return Image.fromarray(array, mode=mode).convert("RGB")

    def tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        cleaned_prompt = " ".join(prompt.split())
        positive = self.tokenizer(
            [cleaned_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative = self.tokenizer(
            [DEFAULT_NEGATIVE_PROMPT],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {
            "text": positive.input_ids,
            "text_attention_mask": positive.attention_mask,
            "text_negative": negative.input_ids,
            "text_attention_mask_negative": negative.attention_mask,
        }

    def make_action_noise(
        self,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Create the initial 24x32 action noise used by DreamZero denoising."""

        selected_seed = self.action_seed if seed is None else seed
        generator = torch.Generator(device=device).manual_seed(selected_seed)
        return torch.randn(
            (1, self.action_horizon, self.action_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def adapt(
        self,
        *,
        image: ImageInput | None = None,
        images: Mapping[str, ImageInput] | None = None,
        camera_name: str = "front",
        prompt: str,
        joint_positions: Sequence[float] | np.ndarray | torch.Tensor,
        action_noise_device: str | torch.device = "cpu",
        action_noise_dtype: torch.dtype = torch.bfloat16,
        action_seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build one batch from a synchronized camera observation.

        ``image`` is a convenience for a single named camera; use ``images`` to
        provide two or three camera streams. Unprovided quadrants remain black.
        """

        if (image is None) == (images is None):
            raise ValueError("Provide exactly one of image or images")
        if image is not None:
            if camera_name not in CAMERA_NAMES:
                raise ValueError(f"Unknown camera {camera_name!r}; expected {CAMERA_NAMES}")
            selected_images: Mapping[str, ImageInput] = {camera_name: image}
        else:
            assert images is not None
            selected_images = images

        state, state_mask = self.normalize_and_pad_state(joint_positions)
        selected_seed = self.action_seed if action_seed is None else action_seed
        batch = {
            "images": self.prepare_camera_mosaic(selected_images),
            "state": state,
            "state_mask": state_mask,
            "embodiment_id": torch.tensor([SO101_ACTION_CATEGORY_ID], dtype=torch.int64),
            "action_seed": torch.tensor([selected_seed], dtype=torch.int64),
            "action_noise": self.make_action_noise(
                device=action_noise_device,
                dtype=action_noise_dtype,
                seed=selected_seed,
            ),
        }
        batch.update(self.tokenize_prompt(prompt))
        return batch

    def adapt_camera_views(
        self,
        *,
        images: Mapping[str, ImageInput],
        prompt: str,
        joint_positions: Sequence[float] | np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Adapt synchronized cameras into one 2x2 mosaic model batch."""

        return self.adapt(
            images=images,
            prompt=prompt,
            joint_positions=joint_positions,
            **kwargs,
        )

    @staticmethod
    @torch.inference_mode()
    def embed_text(
        policy_head: torch.nn.Module, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed tokenized prompts with the loaded DreamZero text encoder.

        The standard ``lazy_joint_video_action`` path performs this internally;
        this helper is for an offline script that wants to cache embeddings.
        """

        device = next(policy_head.text_encoder.parameters()).device
        prompt_embeddings = policy_head.encode_prompt(
            batch["text"].to(device),
            batch["text_attention_mask"].to(device),
        )
        negative_embeddings = policy_head.encode_prompt(
            batch["text_negative"].to(device),
            batch["text_attention_mask_negative"].to(device),
        )
        return prompt_embeddings, negative_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--joint-positions",
        type=float,
        nargs=6,
        required=True,
        metavar=tuple(name.upper() for name in JOINT_NAMES),
    )
    parser.add_argument("--camera", choices=CAMERA_NAMES, default="front")
    parser.add_argument("--action-seed", type=int, default=1140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SO101ObservationAdapter(action_seed=args.action_seed)
    batch = adapter.adapt(
        image=args.image,
        camera_name=args.camera,
        prompt=args.prompt,
        joint_positions=args.joint_positions,
    )

    print(f"SO-101 {args.camera} observation adapted successfully")
    print(f"  joint order: {JOINT_NAMES}")
    for key, value in batch.items():
        value_range = ""
        if value.is_floating_point() and value.numel():
            value_range = f" range=[{value.min().item():.4f}, {value.max().item():.4f}]"
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}{value_range}")


if __name__ == "__main__":
    main()
