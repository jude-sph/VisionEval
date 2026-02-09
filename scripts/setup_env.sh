#!/bin/bash
# Setup script for VisionEval project
# Run this on each machine after cloning the repo

set -e

echo "=== VisionEval Environment Setup ==="

# Detect GPU configuration
echo ""
echo "--- GPU Detection ---"
if command -v nvidia-smi &> /dev/null; then
    echo "GPUs found:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
    echo ""

    # Check for specific GPUs
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "3090"; then
        echo "Detected: Machine B (RTX 3090)"
        echo "Config: FP16, single GPU"
    elif nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "TITAN X"; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        if [[ "$COMPUTE_CAP" == "6.1" ]]; then
            echo "Detected: Machine A (Titan X Pascal, compute 6.1)"
            echo "Config: INT8 tensor parallel (2 GPUs per instance)"
        else
            echo "Warning: Titan X with compute $COMPUTE_CAP (Maxwell?) — may not support INT8"
        fi
    else
        echo "Unknown GPU configuration"
    fi
else
    echo "No NVIDIA GPUs detected (CPU/MPS mode)"
fi

# Create conda environment
echo ""
echo "--- Environment Setup ---"
if command -v conda &> /dev/null; then
    echo "Creating conda environment 'visioneval'..."
    conda create -n visioneval python=3.11 -y 2>/dev/null || echo "Environment already exists"
    echo "Activate with: conda activate visioneval"
    echo ""
    echo "Then install dependencies:"
    echo "  pip install -e '.[dev]'"
else
    echo "conda not found. Create a virtual environment manually:"
    echo "  python3.11 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e '.[dev]'"
fi

# Clone Cambrian
echo ""
echo "--- Cambrian Setup ---"
CAMBRIAN_DIR="${CAMBRIAN_DIR:-$HOME/cambrian}"
if [ -d "$CAMBRIAN_DIR" ]; then
    echo "Cambrian already cloned at $CAMBRIAN_DIR"
else
    echo "Cloning Cambrian..."
    git clone https://github.com/cambrian-mllm/cambrian.git "$CAMBRIAN_DIR"
fi
echo "Install Cambrian with:"
echo "  cd $CAMBRIAN_DIR && pip install -e ."

# Verify critical packages
echo ""
echo "--- Verification ---"
echo "After installing, verify with:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo "  python -c \"import transformers; print(f'Transformers: {transformers.__version__}')\""
echo "  python -c \"import accelerate; print(f'Accelerate: {accelerate.__version__}')\""
echo "  python -c \"from cambrian.model.builder import load_pretrained_model; print('Cambrian: OK')\""

# INT8 test (Machine A only)
echo ""
echo "--- INT8 Compatibility Test (Machine A) ---"
echo "Run this on the Pascal machine to verify INT8 works:"
echo "  python scripts/run_single.py --benchmark mmmu --condition normal --max_samples 5 --gpu_ids 0,1 --load_8bit"

echo ""
echo "=== Setup complete ==="
