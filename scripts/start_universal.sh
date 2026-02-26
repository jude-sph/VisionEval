#!/bin/bash
# Launch universal bottleneck optimization in a persistent tmux session.
#
# Usage:
#   bash scripts/start_universal.sh                                            # Default: 10/class, 10 steps
#   bash scripts/start_universal.sh --max_samples_per_class 20 --num_steps 15
#
# Check progress:
#   tmux attach -t universal
#   tail -f logs/optimize_universal.log
#
# Stop:
#   tmux kill-session -t universal

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/optimize_universal.log"

# Find python
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "ERROR: No python found."
    exit 1
fi

ARGS="$@"

echo ""
echo "=== Universal Bottleneck Launcher ==="
echo "Python:  $PYTHON"
echo "Log:     $LOG_FILE"
echo "Args:    $ARGS"
echo ""

if command -v tmux &> /dev/null; then
    tmux kill-session -t universal 2>/dev/null || true
    tmux new-session -d -s universal \
        "cd $PROJECT_DIR && $PYTHON scripts/optimize_universal.py $ARGS 2>&1 | tee $LOG_FILE; echo ''; echo 'Done. Press enter to close.'; read"
    echo "Running in tmux session 'universal'."
    echo "  tmux attach -t universal           # View live output"
    echo "  tmux kill-session -t universal     # Stop"
    echo "  tail -f $LOG_FILE                  # Follow log"
else
    cd "$PROJECT_DIR"
    nohup bash -c "$PYTHON scripts/optimize_universal.py $ARGS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
fi
echo ""
