#!/bin/bash
# Watchdog daemon: auto-runs extended test suite after 1 hour of idle.
# Usage: ./scripts/test-watchdog.sh &
#
# Polls every 60 seconds for file changes in src/, python/, tests/.
# After 60 consecutive idle checks (1 hour), runs extended tests if
# anything changed since the last completed run.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
MARKER="/tmp/geoprover-last-change"
LAST_RUN_HASH="/tmp/geoprover-last-extended-hash"
POLL_INTERVAL=60
IDLE_THRESHOLD=60  # 60 polls × 60s = 1 hour

mkdir -p "$LOG_DIR"

# Initialize marker if missing
if [ ! -f "$MARKER" ]; then
    touch "$MARKER"
fi

idle_count=0

echo "[watchdog] Started. PID=$$, polling every ${POLL_INTERVAL}s, idle threshold=${IDLE_THRESHOLD} polls (1 hour)"

while true; do
    sleep "$POLL_INTERVAL"

    # Check for file changes newer than marker
    changed_files=$(find "$REPO_DIR/src" "$REPO_DIR/python" "$REPO_DIR/tests" -newer "$MARKER" -type f 2>/dev/null | head -1)

    if [ -n "$changed_files" ]; then
        # Activity detected — reset idle counter and update marker
        touch "$MARKER"
        idle_count=0
    else
        idle_count=$((idle_count + 1))
    fi

    if [ "$idle_count" -ge "$IDLE_THRESHOLD" ]; then
        # Check if repo state changed since last run
        current_hash=$(cd "$REPO_DIR" && find src/ python/ tests/ -type f -exec stat -c '%Y %n' {} + 2>/dev/null | sort | md5sum | cut -d' ' -f1)
        last_hash=""
        if [ -f "$LAST_RUN_HASH" ]; then
            last_hash=$(cat "$LAST_RUN_HASH")
        fi

        if [ "$current_hash" != "$last_hash" ]; then
            timestamp=$(date +%Y%m%d-%H%M%S)
            logfile="$LOG_DIR/extended-test-${timestamp}.log"
            echo "[watchdog] Idle for 1 hour. Running extended tests → $logfile"

            if (cd "$REPO_DIR" && bash scripts/test-extended.sh) > "$logfile" 2>&1; then
                echo "[watchdog] Extended tests PASSED at $(date)"
                echo "$current_hash" > "$LAST_RUN_HASH"
            else
                echo "[watchdog] Extended tests FAILED at $(date). See $logfile"
            fi
        else
            echo "[watchdog] No changes since last run. Skipping."
        fi

        # Reset idle counter after running (or skipping)
        idle_count=0
        touch "$MARKER"
    fi
done
