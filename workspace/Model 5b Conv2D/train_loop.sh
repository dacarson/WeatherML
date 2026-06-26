#!/bin/bash
# Restart training automatically after each per-epoch clean stop.
#
# Exit codes from train_model_conv2D.py:
#   0  — training complete (EarlyStopping fired or max_epochs reached)
#   42 — per-epoch clean stop (MaxEpochsPerRun); restart immediately
#   99 — hang detected by watchdog; restart after brief pause
#   *  — unexpected error; stop and report

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/${1:-train_model_conv2D.py}"
RUN=0

while true; do
    RUN=$((RUN + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀  Run #${RUN} — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python3 "$TRAIN_SCRIPT"
    CODE=$?

    case $CODE in
        0)
            echo ""
            echo "✅  Training complete. Exiting loop."
            exit 0
            ;;
        42)
            echo ""
            echo "🔄  Per-epoch stop (Metal reset). Restarting immediately..."
            continue
            ;;
        99)
            echo ""
            echo "⚠️   Hang detected (exit 99). Pausing 10 s then restarting..."
            sleep 10
            continue
            ;;
        *)
            echo ""
            echo "❌  Unexpected exit code $CODE. Stopping loop."
            exit $CODE
            ;;
    esac
done
