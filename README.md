# DreamZero SO101 Inference Quick Start

Quick setup for running DreamZero SO101 inference on Ubuntu.

## Required versions

* **Ubuntu:** 22.04 or 24.04 recommended
* **Python:** 3.11
* **CUDA Toolkit:** 12.9
* **PyTorch:** 2.8.0 + CUDA 12.9
* **torchvision:** 0.23.0
* **torchaudio:** 2.8.0
* **FlashAttention:** installed from `flash-attn`

Required Hugging Face downloads:

* `Wan-AI/Wan2.1-I2V-14B-480P`
* `google/umt5-xxl`
* `Vizuara/dreamzero-so101-lora`
* `whosricky/so101-megamix-v1`

Expected repo structure:

```text
dreamzero-so101-inference/
├── checkpoints/
├── data/
├── dreamzero-so101/
├── images/
└── outputs/
```

## 1. Install system dependencies

```bash
sudo apt-get update

sudo apt-get install -y --no-install-recommends \
  git \
  curl \
  wget \
  ca-certificates \
  build-essential \
  python3-dev \
  cmake \
  pkg-config \
  ninja-build \
  ffmpeg \
  libglib2.0-0 \
  libgl1 \
  libsm6 \
  libxext6 \
  libxrender1

sudo update-ca-certificates
```

## 2. Install CUDA Toolkit 12.9

Skip this section if `nvcc --version` already reports CUDA 12.9.

```bash
. /etc/os-release
UBUNTU_TAG=ubuntu${VERSION_ID/./}

wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_TAG}/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-9
```

Set CUDA environment variables:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_PATH=/usr/local/cuda-12.9
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Optional: persist them in `~/.bashrc`.

```bash
cat >> ~/.bashrc <<'EOF'

export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_PATH=/usr/local/cuda-12.9
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
```

Verify:

```bash
nvcc --version
nvidia-smi
```

## 3. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL
uv self update
uv --version
```

## 4. Clone this repo

```bash
mkdir -p ~/dev
cd ~/dev

git clone https://github.com/grmpn/dreamzero-so101-inference.git
cd dreamzero-so101-inference
```

## 5. Create the Python environment

```bash
uv python install 3.11
uv venv .venv --python 3.11

source .venv/bin/activate

python --version
```

## 6. Install PyTorch CUDA 12.9

```bash
uv pip uninstall -y torch torchvision torchaudio triton || true

uv pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

Verify:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Expected:

```text
torch: 2.8.0+cu129
torch CUDA: 12.9
cuda available: True
```

## 7. Install DreamZero SO101

```bash
cd ~/dev/dreamzero-so101-inference/dreamzero-so101

uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

Re-pin PyTorch after install, in case dependencies changed it:

```bash
uv pip install --reinstall \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

## 8. Install FlashAttention

```bash
cd ~/dev/dreamzero-so101-inference
source .venv/bin/activate

uv pip install -U packaging psutil ninja wheel setuptools

MAX_JOBS=4 \
CUDA_HOME="$CUDA_HOME" \
CUDA_PATH="$CUDA_PATH" \
CUDACXX="$CUDACXX" \
uv pip install --no-build-isolation --no-cache-dir flash-attn
```

Verify:

```bash
python - <<'PY'
import flash_attn
print("flash_attn:", flash_attn.__version__)
PY
```

## 9. Download required checkpoints and data

```bash
cd ~/dev/dreamzero-so101-inference
source .venv/bin/activate

mkdir -p checkpoints data .hf-cache

export HF_HOME="$PWD/.hf-cache"
export HF_HUB_CACHE="$PWD/.hf-cache/hub"
export HF_XET_CACHE="$PWD/.hf-cache/xet"

uv pip install -U "huggingface_hub[cli]"
```

Download the required files:

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

hf download google/umt5-xxl \
  --local-dir ./checkpoints/umt5-xxl

hf download Vizuara/dreamzero-so101-lora \
  --local-dir ./checkpoints/dreamzero-so101-lora

hf download whosricky/so101-megamix-v1 \
  --repo-type dataset \
  --local-dir ./data/so101-megamix-v1
```

## 10. Final sanity check

```bash
cd ~/dev/dreamzero-so101-inference
source .venv/bin/activate

python - <<'PY'
import torch, flash_attn, shutil
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("flash_attn:", flash_attn.__version__)
print("CUDA_HOME:", CUDA_HOME)
print("nvcc:", shutil.which("nvcc"))
PY
```

## 11. Run inference

Run from the directory containing `offline_inference.py`.

```bash
python offline_inference.py \
  --config config.json \
  --lora-weights checkpoints/dreamzero-so101-lora/model.safetensors \
  --base-model-path checkpoints/Wan2.1-I2V-14B-480P \
  --image images/front1.jpg \
  --gripper-image images/gripper1.jpg \
  --top-image images/top1.jpg \
  --prompt "Pick up the red cube" \
  --joint-positions -0.47 -99.23 95.37 67.74 -1.64 1.99 \
  --output-dir outputs \
  --num-chunks 30
```

If `--config config.json` does not work, use the absolute path to the config file instead, for example:

```bash
--config /absolute/path/to/dreamzero-so101/config.json
```

If running from the repo root and `offline_inference.py` is inside `dreamzero-so101/`, use:

```bash
python dreamzero-so101/offline_inference.py \
  --config dreamzero-so101/config.json \
  --lora-weights checkpoints/dreamzero-so101-lora/model.safetensors \
  --base-model-path checkpoints/Wan2.1-I2V-14B-480P \
  --image images/front1.jpg \
  --gripper-image images/gripper1.jpg \
  --top-image images/top1.jpg \
  --prompt "Pick up the red cube" \
  --joint-positions -0.47 -99.23 95.37 67.74 -1.64 1.99 \
  --output-dir outputs \
  --num-chunks 30
```
