#!/bin/bash
# Set up a RunPod pod for valid-json-wrong-answer experiments.
# Run this after SSH-ing into the pod.
#
# Usage:
#   bash scripts/setup_runpod.sh <GIT_REPO_URL> <HF_TOKEN>
#
# Example:
#   bash scripts/setup_runpod.sh https://github.com/youruser/valid-json-wrong-answer.git hf_abc123...

set -e

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/setup_runpod.sh <GIT_REPO_URL> <HF_TOKEN>"
    echo ""
    echo "  GIT_REPO_URL  — HTTPS or SSH URL to the valid-json-wrong-answer repo"
    echo "  HF_TOKEN      — HuggingFace token (needed to download Qwen models)"
    echo ""
    echo "Example:"
    echo "  bash scripts/setup_runpod.sh https://github.com/youruser/valid-json-wrong-answer.git hf_abc123..."
    exit 1
fi

GIT_REPO_URL="$1"
HF_TOKEN="$2"

echo "=== Setting up valid-json-wrong-answer on RunPod ==="
echo "Repo: $GIT_REPO_URL"

export HF_HOME=/workspace/hf_cache
export HF_TOKEN="$HF_TOKEN"
mkdir -p $HF_HOME

# Persist env vars for future sessions
echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc
echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc

cd /workspace

# Clone repo (or pull if exists)
if [ -d "valid-json-wrong-answer" ]; then
    echo "Repo exists, pulling latest..."
    cd valid-json-wrong-answer && git pull && cd ..
else
    echo "Cloning repo..."
    git clone "$GIT_REPO_URL" valid-json-wrong-answer
fi

cd valid-json-wrong-answer

# Install dependencies
echo "Installing Python dependencies..."
pip install -q torch transformers peft datasets llguidance slotloss

# Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch, transformers, peft
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'Transformers {transformers.__version__}')
print(f'PEFT {peft.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd /workspace/valid-json-wrong-answer"
echo "  bash scripts/smoketest.sh"
echo "  bash scripts/run_experiment.sh 32b   # or 7b, 05b, all"
