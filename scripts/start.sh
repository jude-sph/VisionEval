#!/bin/bash
# Launch evaluation in a persistent session that survives SSH disconnect.
#
# Usage:
#   bash scripts/start.sh              # Auto-detect machine, run all jobs
#   bash scripts/start.sh --machine B  # Force machine B
#   bash scripts/start.sh --max_samples 10  # Quick test
#
# The process will keep running after you disconnect SSH.
# Check progress anytime with:
#   python scripts/check_progress.py
#
# View live logs with:
#   tail -f logs/eval.log
#
# To stop the run:
#   If using tmux:  tmux kill-session -t visioneval
#   If using nohup: kill $(cat logs/eval.pid)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/eval.log"
PID_FILE="$LOG_DIR/eval.pid"

# Forward all arguments to run_machine.py
ARGS="$@"

echo "=== VisionEval Launcher ==="
echo "Project: $PROJECT_DIR"
echo "Log:     $LOG_FILE"
echo "Args:    $ARGS"
echo ""

# Try tmux first (best option: you can reattach later)
if command -v tmux &> /dev/null; then
    echo "Using tmux (recommended)"
    echo ""
    echo "The evaluation is running in tmux session 'visioneval'."
    echo ""
    echo "Useful commands:"
    echo "  tmux attach -t visioneval     # Reattach to see live output"
    echo "  tmux kill-session -t visioneval  # Stop the run"
    echo "  python scripts/check_progress.py # Check progress from anywhere"
    echo "  tail -f $LOG_FILE             # Follow log file"
    echo ""

    # Kill existing session if any
    tmux kill-session -t visioneval 2>/dev/null || true

    # Create new session running the evaluation
    tmux new-session -d -s visioneval \
        "cd $PROJECT_DIR && python scripts/run_machine.py $ARGS 2>&1 | tee $LOG_FILE; echo ''; echo 'Evaluation complete. Press enter to close.'; read"

    echo "Session started. You can safely disconnect SSH now."

# Fall back to screen
elif command -v screen &> /dev/null; then
    echo "Using screen (tmux not found)"
    echo ""
    echo "Useful commands:"
    echo "  screen -r visioneval          # Reattach"
    echo "  screen -X -S visioneval quit  # Stop the run"
    echo "  python scripts/check_progress.py # Check progress"
    echo ""

    screen -dmS visioneval bash -c \
        "cd $PROJECT_DIR && python scripts/run_machine.py $ARGS 2>&1 | tee $LOG_FILE"

    echo "Session started. You can safely disconnect SSH now."

# Last resort: nohup
else
    echo "Using nohup (tmux and screen not found)"
    echo ""
    echo "Useful commands:"
    echo "  tail -f $LOG_FILE             # Follow log"
    echo "  kill \$(cat $PID_FILE)          # Stop the run"
    echo "  python scripts/check_progress.py # Check progress"
    echo ""

    cd "$PROJECT_DIR"
    nohup python scripts/run_machine.py $ARGS > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "PID: $(cat $PID_FILE)"
    echo "Process started. You can safely disconnect SSH now."
fi

echo ""
echo "=== Check progress anytime with: python scripts/check_progress.py ==="
