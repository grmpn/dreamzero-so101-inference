#!/usr/bin/env python3
"""Instantiate and validate the DreamZero SO-101 action head.

The default ``meta`` mode constructs the complete architecture without allocating
the 14B backbone, injects the configured LoRA modules, and materializes the
DreamZero checkpoint tensors. Use ``full`` only on a machine with enough CPU RAM
for DreamZero's current non-streaming Wan2.1 loader.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import os
import sys
from unittest.mock import patch

import torch
from accelerate import init_empty_weights
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.nn.modules.module import _IncompatibleKeys


from path_utils import (  # noqa: E402
    DREAMZERO_ROOT,
    PROJECT_ROOT,
    default_config_path,
    project_path,
)


DEFAULT_CONFIG = default_config_path()
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "dreamzero-so101-lora" / "model.safetensors"
)

# Keep Hugging Face from trying to write outside this project.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
sys.path.insert(0, str(DREAMZERO_ROOT))

from hydra.utils import instantiate  # noqa: E402
# Import the Hydra target before entering ``init_empty_weights``. Importing
# Transformers/PEFT for the first time inside PyTorch's meta-device context can
# otherwise trigger a torch._dynamo circular initialization.
from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (  # noqa: E402,F401
    WANPolicyHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mode",
        choices=("meta", "full"),
        default="meta",
        help="meta validates architecture/checkpoint compatibility; full loads all base weights",
    )
    parser.add_argument(
        "--parameter-preview-limit",
        type=int,
        default=30,
        help="number of LoRA/action/state parameters to print; use -1 for all or 0 for none",
    )
    return parser.parse_args()


def _configured_asset_paths(cfg: DictConfig) -> dict[str, Path]:
    head = cfg.action_head_cfg.config
    return {
        "Wan2.1 directory": project_path(head.diffusion_model_cfg.diffusion_model_pretrained_path),
        "UMT5 encoder": project_path(head.text_encoder_cfg.text_encoder_pretrained_path),
        "CLIP image encoder": project_path(head.image_encoder_cfg.image_encoder_pretrained_path),
        "Wan VAE": project_path(head.vae_cfg.vae_pretrained_path),
        "DreamZero checkpoint": project_path(cfg.resume_path) / "model.safetensors",
    }


def resolve_config_paths(cfg: DictConfig) -> None:
    """Resolve relative config asset paths before Hydra instantiation."""

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


def validate_local_assets(cfg: DictConfig, checkpoint: Path) -> None:
    assets = _configured_asset_paths(cfg)
    assets["checkpoint argument"] = checkpoint
    missing = [f"{name}: {path}" for name, path in assets.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing configured assets:\n  " + "\n  ".join(missing))

    wan_dir = assets["Wan2.1 directory"]
    index = wan_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Missing Wan2.1 shard index: {index}")
    with index.open() as index_file:
        shard_index = json.load(index_file)
    shard_names = set(shard_index["weight_map"].values())
    missing_shards = sorted(name for name in shard_names if not (wan_dir / name).is_file())
    if missing_shards:
        raise FileNotFoundError("Missing Wan2.1 shards:\n  " + "\n  ".join(missing_shards))


@contextmanager
def suppress_constructor_weight_loading():
    """Let the upstream monolithic constructor build on meta without file loads.

    DreamZero currently loads every component inside ``WANPolicyHead.__init__``.
    In meta mode those loads are deliberately deferred; checkpoint tensors are
    assigned after the complete LoRA-wrapped structure exists.
    """

    original_tensor = torch.tensor

    def meta_safe_tensor(*args, **kwargs):
        if str(kwargs.get("device")) == "cuda":
            kwargs["device"] = "meta"
        return original_tensor(*args, **kwargs)

    def no_weight_file_load(*args, **kwargs):
        return {}

    def accept_empty_state_dict(module, state_dict, *args, **kwargs):
        return _IncompatibleKeys([], [])

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch, "tensor", meta_safe_tensor))
        stack.enter_context(patch.object(torch, "load", no_weight_file_load))
        stack.enter_context(patch.object(torch.nn.Module, "load_state_dict", accept_empty_state_dict))
        yield


def checkpoint_state_for_action_head(checkpoint: Path) -> dict[str, torch.Tensor]:
    raw_state = load_file(str(checkpoint), device="cpu")
    prefix = "action_head."
    invalid = [key for key in raw_state if not key.startswith(prefix)]
    if invalid:
        sample = ", ".join(invalid[:3])
        raise ValueError(f"Checkpoint contains keys outside {prefix!r}: {sample}")
    return {key.removeprefix(prefix): value for key, value in raw_state.items()}


def verify_checkpoint_assignment(
    policy_head: torch.nn.Module, expected_state: dict[str, torch.Tensor]
) -> int:
    """Verify every provided checkpoint tensor against the instantiated module."""

    loaded_state = policy_head.state_dict(keep_vars=True)
    errors: list[str] = []
    verified = 0
    for name, expected in expected_state.items():
        loaded = loaded_state.get(name)
        if loaded is None:
            errors.append(f"{name}: key is absent after load")
            continue
        if loaded.is_meta:
            errors.append(f"{name}: tensor is still on the meta device")
            continue
        if loaded.shape != expected.shape:
            errors.append(f"{name}: shape {tuple(loaded.shape)} != {tuple(expected.shape)}")
            continue
        if loaded.dtype != expected.dtype:
            errors.append(f"{name}: dtype {loaded.dtype} != {expected.dtype}")
            continue
        if not torch.equal(loaded.detach().cpu(), expected):
            errors.append(f"{name}: loaded values differ from checkpoint")
            continue
        verified += 1

    if errors:
        preview = "\n  ".join(errors[:30])
        raise RuntimeError(
            f"Failed to verify {len(errors)} of {len(expected_state)} checkpoint tensors:\n  "
            f"{preview}"
        )
    return verified


def instantiate_meta(cfg: DictConfig, checkpoint: Path):
    # Avoid the upstream constructor's all-shards-at-once DiT load in this mode.
    cfg.action_head_cfg.config.skip_component_loading = True
    with init_empty_weights(include_buffers=True), suppress_constructor_weight_loading():
        policy_head = instantiate(cfg.action_head_cfg)

    state = checkpoint_state_for_action_head(checkpoint)
    incompatible = policy_head.load_state_dict(state, strict=False, assign=True)
    if incompatible.unexpected_keys:
        unexpected = "\n  ".join(incompatible.unexpected_keys[:20])
        raise RuntimeError(f"Checkpoint does not match the reconstructed action head:\n  {unexpected}")
    return policy_head, state, incompatible


def instantiate_full(cfg: DictConfig, checkpoint: Path):
    policy_head = instantiate(cfg.action_head_cfg)
    state = checkpoint_state_for_action_head(checkpoint)
    incompatible = policy_head.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        unexpected = "\n  ".join(incompatible.unexpected_keys[:20])
        raise RuntimeError(f"Checkpoint does not match the loaded action head:\n  {unexpected}")
    return policy_head, state, incompatible


def parameter_summary(module: torch.nn.Module) -> tuple[int, int, int, int]:
    tensors = list(module.parameters())
    materialized = [tensor for tensor in tensors if not tensor.is_meta]
    return (
        len(tensors),
        sum(tensor.numel() for tensor in tensors),
        len(materialized),
        sum(tensor.numel() for tensor in materialized),
    )


def print_parameter_preview(policy_head: torch.nn.Module, limit: int) -> None:
    selected = [
        (name, parameter)
        for name, parameter in policy_head.named_parameters()
        if any(token in name.lower() for token in ("lora", "action", "state"))
    ]
    print(f"  LoRA/action/state parameter tensors: {len(selected):,}")
    if limit == 0:
        return
    preview = selected if limit < 0 else selected[:limit]
    for name, parameter in preview:
        location = "meta" if parameter.is_meta else str(parameter.device)
        print(f"    {name} {tuple(parameter.shape)} {parameter.dtype} {location}")
    if 0 < limit < len(selected):
        print(f"    ... {len(selected) - limit:,} more (use --parameter-preview-limit -1 for all)")


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    checkpoint_path = project_path(args.checkpoint)
    cfg = OmegaConf.load(config_path)
    resolve_config_paths(cfg)
    validate_local_assets(cfg, checkpoint_path)

    if args.mode == "meta":
        policy_head, checkpoint_state, incompatible = instantiate_meta(cfg, checkpoint_path)
    else:
        policy_head, checkpoint_state, incompatible = instantiate_full(cfg, checkpoint_path)

    policy_head.eval()
    verified_tensors = verify_checkpoint_assignment(policy_head, checkpoint_state)
    tensor_count, parameter_count, real_tensor_count, real_parameter_count = parameter_summary(
        policy_head
    )

    print("DreamZero SO-101 action head instantiated successfully")
    print(f"  mode: {args.mode}")
    print(f"  policy head: {type(policy_head).__module__}.{type(policy_head).__name__}")
    print(f"  diffusion model: {type(policy_head.model).__name__}")
    print(f"  text encoder: {type(policy_head.text_encoder).__name__}")
    print(f"  image encoder: {type(policy_head.image_encoder).__name__}")
    print(f"  VAE: {type(policy_head.vae).__name__}")
    print("  checkpoint prefix removed for action-head-only load: action_head.")
    print(f"  checkpoint tensors loaded and value-verified: {verified_tensors}")
    print(f"  missing keys: {len(incompatible.missing_keys):,}")
    print(f"  unexpected keys: {len(incompatible.unexpected_keys):,}")
    print(f"  first missing: {incompatible.missing_keys[:30]}")
    print(f"  first unexpected: {incompatible.unexpected_keys[:30]}")
    print(f"  parameter tensors: {tensor_count:,} ({parameter_count:,} parameters)")
    print(
        f"  materialized tensors: {real_tensor_count:,} "
        f"({real_parameter_count:,} parameters)"
    )
    print_parameter_preview(policy_head, args.parameter_preview_limit)


if __name__ == "__main__":
    main()
