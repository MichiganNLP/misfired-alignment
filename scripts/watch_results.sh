#!/bin/bash
# Watch results/ for new or updated result files and re-run live_analyze.py.
# Polls every INTERVAL seconds (default 300 = 5 min).
#
# Usage:
#   bash scripts/watch_results.sh            # poll every 5 min
#   bash scripts/watch_results.sh 60         # poll every 60 s
#   bash scripts/watch_results.sh 300 --plot # also regenerate plots each cycle

set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="$PROJ_DIR/results"
REPORT="$RESULTS_DIR/live_report.txt"
INTERVAL="${1:-300}"
EXTRA_ARGS="${2:-}"

echo "=== watch_results.sh started (interval=${INTERVAL}s) ==="
echo "Report will be written to: $REPORT"

last_snapshot=""

while true; do
    # Snapshot: sorted list of (file, size, mtime) for all result files
    snapshot=$(find "$RESULTS_DIR" -maxdepth 1 \( -name "*_results.json" -o -name "*_results.jsonl" \) \
        -printf "%f %s %T@\n" 2>/dev/null | sort)

    if [[ "$snapshot" != "$last_snapshot" ]]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Change detected — running analysis..."
        "$PYTHON" "$PROJ_DIR/scripts/live_analyze.py" $EXTRA_ARGS \
            2>&1 | tee "$REPORT"
        last_snapshot="$snapshot"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report written to $REPORT"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes. Next check in ${INTERVAL}s..."
    fi

    sleep "$INTERVAL"
done
