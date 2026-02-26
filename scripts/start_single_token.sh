#!/bin/bash
# Launch single-token experiment in a persistent tmux session.
#
# Usage:
#   bash scripts/start_single_token.sh
#   bash scripts/start_single_token.sh --max_samples 50 --num_steps 30
#
# Monitor:
#   tmux attach -t single_token
#   tail -f logs/optimize_single_token.log
#
# Stop:
#   tmux kill-session -t single_token

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/optimize_single_token.log"

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
echo "=== Single Token Experiment Launcher ==="
echo "Python:  $PYTHON"
echo "Log:     $LOG_FILE"
echo "Args:    $ARGS"
echo ""

if command -v tmux &> /dev/null; then
    tmux kill-session -t single_token 2>/dev/null || true
    tmux new-session -d -s single_token \
        "cd $PROJECT_DIR && $PYTHON scripts/optimize_single_token.py $ARGS 2>&1 | tee $LOG_FILE; echo ''; echo 'Done. Press enter to close.'; read"
    echo "Running in tmux session 'single_token'."
    echo "  tmux attach -t single_token        # View live output"
    echo "  tmux kill-session -t single_token  # Stop"
    echo "  tail -f $LOG_FILE                  # Follow log"
else
    cd "$PROJECT_DIR"
    nohup bash -c "$PYTHON scripts/optimize_single_token.py $ARGS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
fi
echo ""
