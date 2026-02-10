# Setup

## Requirements

- Linux with NVIDIA GPU (24GB+ VRAM, tested on RTX 3090)
- Python 3.10+
- CUDA 12.x

## Install

```bash
git clone https://github.com/jude-sph/VisionEval.git
cd VisionEval

# Clone Cambrian into project root
git clone https://github.com/cambrian-mllm/cambrian.git

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e cambrian/

# Flash Attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

## Run

```bash
# Everything (benchmarks + noise optimization) in background
bash scripts/start.sh

# Quick test (10 samples per benchmark)
bash scripts/start.sh --max_samples 10

# Single benchmark + condition
python scripts/run_single.py --benchmark mmmu --condition normal --max_samples 10

# Noise optimization only
python scripts/optimize_noise.py --benchmark mmmu --max_samples 50
```

## Monitor

```bash
tmux attach -t visioneval        # live output
tail -f logs/eval.log             # log file
python scripts/check_progress.py  # summary
```

## Results

```
results/raw/           # per-question JSONL (one file per benchmark_condition)
results/aggregated/    # summary CSV + JSON
results/optimization/  # noise optimization results
```
