# DreamZero SO-101 Inference Handoff

## Project goal

Build a standalone inference path for the `dreamzero-so101` LoRA checkpoint on
top of the DreamZero architecture and the Wan2.1-I2V-14B-480P backbone. The
intended end state is offline inference from synchronized camera observations,
a natural-language instruction, and the current six SO-101 joint positions,
producing both a 24-step joint trajectory and predicted video frames.

## Scope and repository rules

- Do not modify anything under `dreamzero/` unless the user explicitly asks.
  Treat it as upstream reference code.
- Put project-specific code, tests, documentation, and compatibility work under
  `dreamzero-so101/`.
- Preserve user changes and unrelated files. The upstream `dreamzero/` worktree
  already contains an untracked `cuda-keyring_1.1-1_all.deb`; it is unrelated.
- Checkpoints and datasets are large assets and must not be rewritten or copied.
- Prefer the local `.venv/bin/python` environment.
- The current configs contain absolute paths. On another host, pass
  `--base-model-path` or patch the config paths deliberately.

## Important paths

- Project config: `dreamzero-so101/config.json`
- Original checkpoint config: `checkpoints/dreamzero-so101-lora/config.json`
- SO-101 checkpoint: `checkpoints/dreamzero-so101-lora/model.safetensors`
- Wan2.1 backbone: `checkpoints/Wan2.1-I2V-14B-480P/`
- UMT5 tokenizer: `checkpoints/umt5-xxl/`
- Dataset metadata: `data/so101-megamix-v1/meta/`
- Upstream DreamZero source: `dreamzero/`

## Completed work

### Config path repair

The stale `/workspace/checkpoints/...` paths in the LoRA config were replaced
with local checkpoint paths for the Wan DiT, UMT5 encoder, CLIP image encoder,
Wan VAE, and LoRA resume directory.

### Architecture and checkpoint validation

`instantiate_action_head.py` reconstructs the real DreamZero `WANPolicyHead`.
Its default meta-device mode avoids allocating the 14B backbone while still:

- constructing all 40 Wan layers;
- injecting the configured rank-4 LoRA modules;
- loading the SO-101 action heads and LoRA tensors;
- calling `eval()`;
- verifying checkpoint values, shapes, and dtypes.

Verified checkpoint result:

- 814 SO-101 tensors loaded and value-verified;
- 0 unexpected keys;
- 2,132 missing keys, all expected frozen base-component weights not supplied
  by the LoRA-only checkpoint.

The checkpoint keys begin with `action_head.`. Because the script instantiates
the action head directly rather than the outer VLA wrapper, that prefix must be
removed before `load_state_dict`.

Run the validation with:

```bash
.venv/bin/python dreamzero-so101/instantiate_action_head.py
```

### Observation adapter

`observation_adapter.py` converts raw SO-101 observations into the tensors used
by `WANPolicyHead.lazy_joint_video_action`.

Joint order is fixed and must never be rearranged:

1. `shoulder_pan`
2. `shoulder_lift`
3. `elbow_flex`
4. `wrist_flex`
5. `wrist_roll`
6. `gripper`

State normalization uses the dataset's `observation.state.q01` and `q99`:

```text
normalized = 2 * (value - q01) / (q99 - q01) - 1
```

Values are clipped to `[-1, 1]`. The six normalized joints are placed in the
first six channels of a zero-padded `[1, 1, 64]` float tensor. A matching
boolean state mask marks those six valid channels.

The dataset contains all required state and action q01/q99 statistics. No
normalization statistics are currently missing.

#### Camera layout

This checkpoint does not use three independent inference calls. It was trained
on one synchronized 2x2 mosaic:

```text
+-------------------+-------------------+
| front             | top               |
+-------------------+-------------------+
| gripper           | black padding     |
+-------------------+-------------------+
```

Each camera is resized to width 320, height 176. The complete model input is
therefore `[B=1, T=1, H=352, W=640, C=3]` uint8. This is checkpoint-critical:
WanVAE downsamples by 8 and the DiT patchifies by 2x2, giving
`(352 / 16) * (640 / 16) = 880` tokens, matching `frame_seqlen=880`.

Missing camera streams are black-padded. A single `image` input is treated as
the front camera by default. When available, supply synchronized front, top,
and gripper frames.

#### Text, embodiment, and action seed

- Positive and DreamZero negative prompts are tokenized with local UMT5-XXL to
  `[1, 512]` IDs and attention masks.
- `embed_text()` can cache positive and negative text embeddings after the
  policy head is loaded. Standard inference embeds the tokens internally.
- The correct action/state category ID is `0`, not a global DreamZero registry
  ID. The checkpoint action/state tensors have category dimension one, and the
  current `CausalWanModel` explicitly selects category zero.
- The action seed defaults to DreamZero's `1140`.
- Seeded action noise has shape `[1, 24, 32]` and BF16 dtype.

Run adapter tests with:

```bash
.venv/bin/python dreamzero-so101/test_observation_adapter.py -v
```

### Offline inference demo

`offline_inference.py` is the H100-targeted entry point. It:

1. loads the supplied config and optionally overrides Wan paths;
2. instantiates the full Wan2.1/DreamZero action head;
3. loads the 814 SO-101 tensors;
4. moves the completed model to CUDA as BF16;
5. runs the observation adapter;
6. executes causal joint video/action inference;
7. denormalizes the first six action channels with `action.q01/q99`;
8. prints all 24 generated joint positions;
9. decodes and saves every returned video frame.

Example:

```bash
.venv/bin/python dreamzero-so101/offline_inference.py \
  --config dreamzero-so101/config.json \
  --lora-weights checkpoints/dreamzero-so101-lora/model.safetensors \
  --base-model-path checkpoints/Wan2.1-I2V-14B-480P \
  --image front.jpg \
  --top-image top.jpg \
  --gripper-image gripper.jpg \
  --prompt "Pick the red cube" \
  --joint-positions -0.47 -99.23 95.37 67.74 -1.64 1.99 \
  --output-dir outputs/demo
```

Generated output:

- `predicted_actions.csv`: all 24 denormalized six-joint commands;
- `predicted_video.mp4`: decoded composite rollout;
- `frames/frame_XXX.png`: every composite frame;
- `frames/front/`, `frames/top/`, `frames/gripper/`: native camera crops;
- `metadata.json`: inputs, shapes, seed, and output details.

The current causal method returns three latent frames on its first call, which
decode to nine RGB frames. Frame zero contains the conditioning mosaic; the
remaining frames are the generated continuation. Do not claim that a single
causal call returns the 33-frame training window.

The upstream class stores the config value as `num_inference_timesteps`, while
the causal method reads `num_inference_steps`. The offline script deliberately
copies the configured value (4) to the attribute used during inference.

Run offline helper tests with:

```bash
.venv/bin/python dreamzero-so101/test_offline_inference.py -v
```

## Hardware and execution constraints

- The host RTX 4090 works and is visible to PyTorch outside the Codex sandbox.
  Normal sandboxed commands hide `/dev/nvidia*`; GPU commands may require
  elevated execution.
- The 4090 has 24 GB VRAM and cannot hold this full unquantized model.
- Full inference is intended to be verified on an H100 80GB.
- The upstream constructor is memory-heavy: it builds FP32 modules and
  accumulates sharded Wan state before moving the model to BF16 CUDA. The H100
  host also needs substantial system RAM.
- The complete H100 inference path has not yet been executed in this workspace.
  Syntax, adapter behavior, LoRA compatibility, normalization, frame layout,
  and output serialization have been tested locally.
- `--compile-encoders` is optional and increases startup time. Leave it off for
  the first correctness run.
- `--attention-backend FA2` is the default. `FA3`, Transformer Engine (`TE`),
  and PyTorch attention are optional alternatives when installed and working.

## Current test status

All five standard-library tests pass:

- three observation-adapter tests;
- two offline-inference helper tests.

The tests cover q99 endpoints and midpoint, state padding/masks, deterministic
action noise, tokenizer tensor shapes, category ID zero, camera mosaic layout,
the 880-token spatial contract, action denormalization, CSV serialization, and
composite/per-camera frame output.

## Recommended next steps

1. Move or mount this repository and checkpoints on the H100 host.
2. Confirm all absolute config paths, or use `--base-model-path`.
3. Run `instantiate_action_head.py` meta validation first.
4. Run one offline inference with all three synchronized cameras.
5. Check peak host RAM and H100 VRAM during construction and inference.
6. Confirm that 24 actions, nine decoded frames, and the MP4 are produced.
7. Compare predicted actions against a known dataset sample before attempting
   any real-robot control.
8. Keep physical deployment out of scope until explicit safety limits,
   controller rate handling, and action validation are added.

