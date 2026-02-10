#!/bin/bash
# Launch noise optimization in a persistent session that survives SSH disconnect.
#
# Usage:
#   bash scripts/start_noise.sh                           # Default: MMMU, 50 samples, 4 GPUs
#   bash scripts/start_noise.sh --benchmark mmmu --max_samples 100
#   bash scripts/start_noise.sh --max_samples 1           # Smoke test
#
# The process will keep running after you disconnect SSH.
# Check progress anytime with:
#   python scripts/check_progress.py
#
# View live logs with:
#   tail -f logs/optimize_noise.log
#
# To stop the run:
#   tmux kill-session -t noise

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/optimize_noise.log"
PID_FILE="$LOG_DIR/noise.pid"

# Forward all arguments to optimize_noise.py
ARGS="$@"

echo ""
echo "=== Noise Optimization Launcher ==="
echo "Project: $PROJECT_DIR"
echo "Log:     $LOG_FILE"
echo "Args:    $ARGS"
echo ""

# Try tmux first (best option: you can reattach later)
if command -v tmux &> /dev/null; then
    echo "Using tmux (recommended)"
    echo ""
    echo "The optimization is running in tmux session 'noise'."
    echo ""
    echo "Useful commands:"
    echo "  tmux attach -t noise              # Reattach to see live output"
    echo "  tmux kill-session -t noise        # Stop the run"
    echo "  python scripts/check_progress.py  # Check progress from anywhere"
    echo "  tail -f $LOG_FILE                 # Follow log file"
    echo ""

    # Kill existing session if any
    tmux kill-session -t noise 2>/dev/null || true

    tmux new-session -d -s noise \
        "cd $PROJECT_DIR && python scripts/optimize_noise.py $ARGS 2>&1 | tee $LOG_FILE; echo ''; echo 'Optimization complete. Press enter to close.'; read"

    echo "Session started. You can safely disconnect SSH now."

# Fall back to nohup
else
    echo "Using nohup (tmux not found)"
    echo ""
    echo "Useful commands:"
    echo "  tail -f $LOG_FILE                 # Follow log"
    echo "  kill \$(cat $PID_FILE)              # Stop the run"
    echo "  python scripts/check_progress.py  # Check progress"
    echo ""

    cd "$PROJECT_DIR"
    nohup bash -c "python scripts/optimize_noise.py $ARGS" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "PID: $(cat $PID_FILE)"
    echo "Process started. You can safely disconnect SSH now."
fi

echo ""
echo "=== Check progress anytime with: python scripts/check_progress.py ==="
