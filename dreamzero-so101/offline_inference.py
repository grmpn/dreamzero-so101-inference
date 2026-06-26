#!/usr/bin/env python3
"""Run one offline DreamZero-SO101 video-and-action inference.

This script is intended for a CUDA GPU large enough to hold the full model
(for example, an H100 80GB). It loads Wan2.1 from the paths in the supplied
config, applies the SO-101 LoRA/action checkpoint, adapts one observation, and
writes denormalized actions plus every decoded frame to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers.feature_extraction_utils import BatchFeature


from path_utils import (  # noqa: E402
    DREAMZERO_ROOT,
    PROJECT_ROOT,
    default_config_path,
    project_path,
)


DEFAULT_CONFIG = default_config_path()
DEFAULT_LORA = PROJECT_ROOT / "checkpoints" / "dreamzero-so101-lora" / "model.safetensors"
DEFAULT_STATS = PROJECT_ROOT / "data" / "so101-megamix-v1" / "meta" / "stats.json"
DEFAULT_INFO = PROJECT_ROOT / "data" / "so101-megamix-v1" / "meta" / "info.json"
DEFAULT_TOKENIZER = PROJECT_ROOT / "checkpoints" / "umt5-xxl"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
sys.path.insert(0, str(DREAMZERO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from observation_adapter import JOINT_NAMES, SO101ObservationAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lora-weights", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--top-image",
        type=Path,
        help="optional synchronized top-camera frame; omitted cameras are black-padded",
    )
    parser.add_argument(
        "--gripper-image",
        type=Path,
        help="optional synchronized gripper-camera frame; omitted cameras are black-padded",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--joint-positions",
        type=float,
        nargs=6,
        required=True,
        metavar=tuple(name.upper() for name in JOINT_NAMES),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-path",
        type=Path,
        help="optional in-memory override for every Wan2.1 asset path in the config",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--info-path", type=Path, default=DEFAULT_INFO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-seed", type=int, default=1140)
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help=(
            "number of 24-action chunks to generate; chunks after the first are "
            "conditioned on the previous chunk's final predicted frame and action"
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--attention-backend",
        choices=("FA2", "FA3", "TE", "torch"),
        default="FA2",
    )
    parser.add_argument(
        "--compile-encoders",
        action="store_true",
        help="torch.compile the text/image/VAE encoders after loading (slower startup)",
    )
    parser.add_argument(
        "--skip-mp4",
        action="store_true",
        help="write PNG frames only",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")


def resolve_lora_path(path: str | Path) -> Path:
    resolved = project_path(path)
    if resolved.is_dir():
        resolved = resolved / "model.safetensors"
    if not resolved.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {resolved}")
    return resolved


def patch_base_model_paths(cfg: DictConfig, base_model_path: str | Path) -> None:
    base = project_path(base_model_path)
    if not base.is_dir():
        raise FileNotFoundError(f"Wan2.1 directory not found: {base}")
    head = cfg.action_head_cfg.config
    head.diffusion_model_cfg.diffusion_model_pretrained_path = str(base)
    head.text_encoder_cfg.text_encoder_pretrained_path = str(
        base / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    head.image_encoder_cfg.image_encoder_pretrained_path = str(
        base / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    head.vae_cfg.vae_pretrained_path = str(base / "Wan2.1_VAE.pth")


def resolve_config_paths(cfg: DictConfig) -> None:
    """Resolve relative config asset paths for Docker/container portability."""

    head = cfg.action_head_cfg.config
    head.diffusion_model_cfg.diffusion_model_pretrained_path = str(
        project_path(head.diffusion_model_cfg.diffusion_model_pretrained_path)
    )
    head.text_encoder_cfg.text_encoder_pretrained_path = str(
        project_path(head.text_encoder_cfg.text_encoder_pretrained_path)
    )
    head.image_encoder_cfg.image_encoder_pretrained_path = str(
        project_path(head.image_encoder_cfg.image_encoder_pretrained_path)
    )
    head.vae_cfg.vae_pretrained_path = str(project_path(head.vae_cfg.vae_pretrained_path))
    if cfg.get("resume_path") is not None:
        cfg.resume_path = str(project_path(cfg.resume_path))
    decode_path = head.get("load_pretrained_det_decode_layer_path")
    if decode_path is not None:
        head.load_pretrained_det_decode_layer_path = str(project_path(decode_path))


def checkpoint_state_for_action_head(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    raw_state = load_file(str(checkpoint), device="cpu")
    prefix = "action_head."
    invalid = [key for key in raw_state if not key.startswith(prefix)]
    if invalid:
        raise ValueError(
            f"Expected every checkpoint key to start with {prefix!r}; first invalid key: "
            f"{invalid[0]}"
        )
    return {key.removeprefix(prefix): value for key, value in raw_state.items()}


def load_policy_head(
    *,
    config_path: str | Path,
    lora_path: str | Path,
    device: torch.device,
    base_model_path: str | Path | None,
    action_seed: int,
    compile_encoders: bool,
) -> tuple[torch.nn.Module, DictConfig]:
    """Instantiate Wan2.1, load the SO-101 checkpoint, and move it to BF16 CUDA."""

    from hydra.utils import instantiate

    config_path = project_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if base_model_path is not None:
        patch_base_model_paths(cfg, base_model_path)
    resolve_config_paths(cfg)

    print("Instantiating DreamZero action head and loading Wan2.1 components...")
    policy_head = instantiate(cfg.action_head_cfg)

    print(f"Loading SO-101 LoRA/action weights from {lora_path}...")
    lora_state = checkpoint_state_for_action_head(lora_path)
    incompatible = policy_head.load_state_dict(lora_state, strict=False)
    if incompatible.unexpected_keys:
        preview = "\n  ".join(incompatible.unexpected_keys[:30])
        raise RuntimeError(f"Unexpected SO-101 checkpoint keys:\n  {preview}")
    print(
        f"Loaded {len(lora_state)} SO-101 tensors; "
        f"{len(incompatible.missing_keys)} frozen base keys were supplied separately."
    )
    del lora_state
    gc.collect()

    policy_head.eval()
    policy_head._device = str(device)
    policy_head.seed = action_seed
    # The constructor stores this as ``num_inference_timesteps`` but the causal
    # inference method reads ``num_inference_steps``.
    policy_head.num_inference_steps = int(cfg.action_head_cfg.config.num_inference_timesteps)

    print(f"Moving the complete action head to {device} as BF16...")
    policy_head.to(device=device, dtype=torch.bfloat16)
    policy_head.trt_engine = None

    if compile_encoders:
        print("Compiling text, image, and VAE encoders...")
        policy_head.text_encoder.forward = torch.compile(
            policy_head.text_encoder.forward,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        policy_head.image_encoder.model.visual.forward = torch.compile(
            policy_head.image_encoder.model.visual.forward,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        policy_head.vae.model.encode = torch.compile(
            policy_head.vae.model.encode,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
    torch.cuda.empty_cache()
    return policy_head, cfg


def move_model_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for name, tensor in batch.items():
        if tensor.is_floating_point():
            moved[name] = tensor.to(device=device, dtype=torch.bfloat16)
        else:
            moved[name] = tensor.to(device=device)
    return moved


def load_action_statistics(stats_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = project_path(stats_path)
    with path.open() as stats_file:
        stats = json.load(stats_file)
    if "action" not in stats:
        raise KeyError(f"action statistics are missing from {path}")
    missing = [name for name in ("q01", "q99") if name not in stats["action"]]
    if missing:
        raise KeyError(f"Missing action statistics in {path}: {missing}")
    q01 = torch.tensor(stats["action"]["q01"], dtype=torch.float32)
    q99 = torch.tensor(stats["action"]["q99"], dtype=torch.float32)
    if q01.shape != (6,) or q99.shape != (6,):
        raise ValueError(f"Expected six action q01/q99 values, got {q01.shape} and {q99.shape}")
    return q01, q99


def denormalize_actions(
    normalized_actions: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
) -> torch.Tensor:
    """Convert the first six normalized channels back to SO-101 joint positions."""

    if normalized_actions.ndim == 3:
        if normalized_actions.shape[0] != 1:
            raise ValueError("Offline demo currently expects batch size one")
        normalized_actions = normalized_actions[0]
    if normalized_actions.ndim != 2 or normalized_actions.shape[-1] < 6:
        raise ValueError(
            f"Expected normalized actions [T,D>=6], got {tuple(normalized_actions.shape)}"
        )
    normalized_joints = normalized_actions[:, :6].float().cpu()
    return (normalized_joints + 1.0) / 2.0 * (q99 - q01) + q01


def decode_video_frames(
    policy_head: torch.nn.Module,
    video_latents: torch.Tensor,
) -> np.ndarray:
    """Decode ``[1,C,T,H,W]`` Wan latents to uint8 ``[T,H,W,3]`` frames."""

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = policy_head.vae.decode(
            video_latents,
            tiled=policy_head.tiled,
            tile_size=(policy_head.tile_size_height, policy_head.tile_size_width),
            tile_stride=(policy_head.tile_stride_height, policy_head.tile_stride_width),
        )
    frames = decoded.permute(0, 2, 3, 4, 1)[0]
    return ((frames.float() + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()


def split_mosaic_frame(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split a decoded 2x2 SO-101 mosaic frame back into camera quadrants."""

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected mosaic frame [H,W,3], got {frame.shape}")
    height, width = frame.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Mosaic dimensions must be even, got {height}x{width}")
    half_h, half_w = height // 2, width // 2
    return {
        "front": np.array(frame[:half_h, :half_w], copy=True),
        "top": np.array(frame[:half_h, half_w:], copy=True),
        "gripper": np.array(frame[half_h:, :half_w], copy=True),
    }


def concatenate_rollout_frames(chunks: Sequence[np.ndarray]) -> np.ndarray:
    """Join decoded chunk frames, dropping repeated conditioning frames."""

    if not chunks:
        raise ValueError("At least one decoded frame chunk is required")
    selected = []
    for index, frames in enumerate(chunks):
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 frames [T,H,W,3], got {frames.shape} {frames.dtype}"
            )
        selected.append(frames if index == 0 else frames[1:])
    return np.concatenate(selected, axis=0)


def run_policy_chunk(
    policy_head: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return policy_head.lazy_joint_video_action(
            BatchFeature(data={}),
            BatchFeature(data=model_inputs),
        )


def save_actions(actions: torch.Tensor, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predicted_actions.csv"
    action_array = actions.detach().cpu().numpy()
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("step", *JOINT_NAMES))
        for step, positions in enumerate(action_array):
            writer.writerow((step, *(f"{float(value):.8f}" for value in positions)))
    return path


def save_frames(frames: np.ndarray, output_dir: str | Path) -> list[Path]:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames [T,H,W,3], got {frames.shape} {frames.dtype}")
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera_dirs = {name: frames_dir / name for name in ("front", "top", "gripper")}
    for directory in camera_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = frames_dir / f"frame_{index:03d}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        paths.append(path)
        height, width = frame.shape[:2]
        if height % 2 == 0 and width % 2 == 0:
            half_h, half_w = height // 2, width // 2
            camera_frames = {
                "front": frame[:half_h, :half_w],
                "top": frame[:half_h, half_w:],
                "gripper": frame[half_h:, :half_w],
            }
            for name, camera_frame in camera_frames.items():
                camera_path = camera_dirs[name] / f"frame_{index:03d}.png"
                Image.fromarray(camera_frame, mode="RGB").save(camera_path)
    return paths


def print_actions(actions: torch.Tensor) -> None:
    print("\nGenerated SO-101 joint positions:")
    print("step  " + "  ".join(f"{name:>15}" for name in JOINT_NAMES))
    for step, positions in enumerate(actions.tolist()):
        values = "  ".join(f"{value:15.6f}" for value in positions)
        print(f"{step:>4}  {values}")


def write_metadata(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    frame_count: int,
    latent_shapes: Sequence[tuple[int, ...]],
    action_shape: tuple[int, ...],
    num_inference_steps: int,
    chunk_count: int,
    action_seeds: Sequence[int],
) -> Path:
    metadata: dict[str, Any] = {
        "config": str(project_path(args.config)),
        "lora_weights": str(resolve_lora_path(args.lora_weights)),
        "input_image": str(args.image.expanduser().resolve()),
        "input_top_image": (
            str(args.top_image.expanduser().resolve()) if args.top_image is not None else None
        ),
        "input_gripper_image": (
            str(args.gripper_image.expanduser().resolve())
            if args.gripper_image is not None
            else None
        ),
        "prompt": args.prompt,
        "input_joint_positions": dict(zip(JOINT_NAMES, args.joint_positions)),
        "action_seed": args.action_seed,
        "action_seeds": list(action_seeds),
        "chunk_count": chunk_count,
        "actions_per_chunk": 24,
        "num_inference_steps": num_inference_steps,
        "action_shape": list(action_shape),
        "video_latent_shapes": [list(shape) for shape in latent_shapes],
        "decoded_frame_count": frame_count,
        "frame_zero_includes_conditioning_image": True,
        "subsequent_chunks_drop_conditioning_frame": True,
    }
    path = output_dir / "metadata.json"
    with path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return path


def validate_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("DreamZero offline inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this Python process")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    print(
        f"Using {properties.name} with "
        f"{properties.total_memory / 1024**3:.1f} GiB VRAM"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ["ATTENTION_BACKEND"] = args.attention_backend
    os.environ.setdefault("ENABLE_TENSORRT", "False")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    validate_cuda_device(device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = resolve_lora_path(args.lora_weights)

    adapter = SO101ObservationAdapter(
        stats_path=args.stats_path,
        info_path=args.info_path,
        tokenizer_path=args.tokenizer_path,
        action_seed=args.action_seed,
    )
    camera_images = {"front": args.image}
    if args.top_image is not None:
        camera_images["top"] = args.top_image
    if args.gripper_image is not None:
        camera_images["gripper"] = args.gripper_image
    cpu_batch = adapter.adapt(
        images=camera_images,
        prompt=args.prompt,
        joint_positions=args.joint_positions,
        action_noise_device="cpu",
    )

    policy_head, _ = load_policy_head(
        config_path=args.config,
        lora_path=lora_path,
        device=device,
        base_model_path=args.base_model_path,
        action_seed=args.action_seed,
        compile_encoders=args.compile_encoders,
    )
    action_q01, action_q99 = load_action_statistics(args.stats_path)
    action_chunks: list[torch.Tensor] = []
    frame_chunks: list[np.ndarray] = []
    latent_shapes: list[tuple[int, ...]] = []
    action_seeds: list[int] = []

    for chunk_index in range(args.num_chunks):
        chunk_seed = args.action_seed + chunk_index
        action_seeds.append(chunk_seed)
        policy_head.seed = chunk_seed

        model_inputs = move_model_inputs(cpu_batch, device)
        print(
            f"Running chunk {chunk_index + 1}/{args.num_chunks} with "
            f"{policy_head.num_inference_steps} denoising steps for prompt: "
            f"{args.prompt!r}"
        )
        prediction = run_policy_chunk(policy_head, model_inputs)

        normalized_actions = prediction["action_pred"].float()
        video_latents = prediction["video_pred"]
        actions = denormalize_actions(normalized_actions, action_q01, action_q99)
        frames = decode_video_frames(policy_head, video_latents)

        action_chunks.append(actions)
        frame_chunks.append(frames)
        latent_shapes.append(tuple(video_latents.shape))

        if chunk_index + 1 < args.num_chunks:
            next_joint_positions = actions[-1].tolist()
            next_camera_images = split_mosaic_frame(frames[-1])
            cpu_batch = adapter.adapt(
                images=next_camera_images,
                prompt=args.prompt,
                joint_positions=next_joint_positions,
                action_noise_device="cpu",
                action_seed=chunk_seed + 1,
            )
            del prediction, normalized_actions, video_latents, model_inputs
            torch.cuda.empty_cache()

    actions = torch.cat(action_chunks, dim=0)
    frames = concatenate_rollout_frames(frame_chunks)

    print_actions(actions)
    actions_path = save_actions(actions, output_dir)
    frame_paths = save_frames(frames, output_dir)
    metadata_path = write_metadata(
        output_dir=output_dir,
        args=args,
        frame_count=len(frame_paths),
        latent_shapes=latent_shapes,
        action_shape=tuple(actions.shape),
        num_inference_steps=policy_head.num_inference_steps,
        chunk_count=args.num_chunks,
        action_seeds=action_seeds,
    )

    video_path = output_dir / "predicted_video.mp4"
    if not args.skip_mp4:
        try:
            imageio.mimsave(video_path, list(frames), fps=args.fps, codec="libx264")
        except Exception as error:
            print(f"Warning: PNG frames were saved, but MP4 creation failed: {error}")

    print(f"\nSaved actions: {actions_path}")
    print(f"Saved {len(frame_paths)} composite frames: {output_dir / 'frames'}")
    print("Saved front/top/gripper crops in matching frames/ subdirectories")
    if video_path.exists():
        print(f"Saved video: {video_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        "Note: frame_000 is the initial conditioning mosaic; subsequent chunk "
        "conditioning frames are omitted from the saved sequence to avoid duplicate "
        "frames. Mosaic layout is front/top on the first row and gripper/black on "
        "the second row."
    )


if __name__ == "__main__":
    main()
