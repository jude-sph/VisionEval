#!/bin/bash
# Launch bottleneck optimization in a persistent session that survives SSH disconnect.
#
# Usage:
#   bash scripts/start_bottleneck.sh                           # Default: MMMU, 50 samples, 1 token
#   bash scripts/start_bottleneck.sh --benchmark mmmu --max_samples 200
#   bash scripts/start_bottleneck.sh --max_samples 1 --num_steps 5  # Smoke test
#
# The process will keep running after you disconnect SSH.
# Check progress anytime with:
#   python scripts/check_progress.py
#
# View live logs with:
#   tail -f logs/optimize_bottleneck.log
#
# To stop the run:
#   tmux kill-session -t bottleneck

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/optimize_bottleneck.log"
PID_FILE="$LOG_DIR/bottleneck.pid"

# Find python — prefer venv, then python3, then python
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "ERROR: No python found. Activate your venv or install python3."
    exit 1
fi

# Forward all arguments to optimize_bottleneck.py
ARGS="$@"

echo ""
echo "=== Bottleneck Optimization Launcher ==="
echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON"
echo "Log:     $LOG_FILE"
echo "Args:    $ARGS"
echo ""

# Try tmux first (best option: you can reattach later)
if command -v tmux &> /dev/null; then
    echo "Using tmux (recommended)"
    echo ""
    echo "The optimization is running in tmux session 'bottleneck'."
    echo ""
    echo "Useful commands:"
    echo "  tmux attach -t bottleneck         # Reattach to see live output"
    echo "  tmux kill-session -t bottleneck   # Stop the run"
    echo "  python scripts/check_progress.py  # Check progress from anywhere"
    echo "  tail -f $LOG_FILE                 # Follow log file"
    echo ""

    # Kill existing session if any
    tmux kill-session -t bottleneck 2>/dev/null || true

    tmux new-session -d -s bottleneck \
        "cd $PROJECT_DIR && $PYTHON scripts/optimize_bottleneck.py $ARGS 2>&1 | tee $LOG_FILE; echo ''; echo 'Optimization complete. Press enter to close.'; read"

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
    nohup bash -c "$PYTHON scripts/optimize_bottleneck.py $ARGS" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "PID: $(cat $PID_FILE)"
    echo "Process started. You can safely disconnect SSH now."
fi

echo ""
echo "=== Check progress anytime with: python scripts/check_progress.py ==="
