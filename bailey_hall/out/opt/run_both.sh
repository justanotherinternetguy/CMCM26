#!/usr/bin/env bash
# Crowd-manager then LED-strip greedy search, expected ~80 min, whole thing capped at 90 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=../.venv/bin/python
T0=$(date +%s)
COMMON="--profile concert --metric t100 --max-items 6 --top-spots 8 --exits-per-spot 2
        --screen-seeds 8 --confirm-seeds 32 --confirm-top 3 --shortlist 10 --seed0 1000"
echo "== managers (start $(date))"
$PY optimize.py managers --leds none --tag managers6 --deadline $((T0 + 42*60)) $COMMON
MGR=$(ls -d out/opt/managers6_* | tail -1)/result.json
echo "== LEDs with managers from $MGR ($(date))"
$PY optimize.py leds --managers "$MGR" --tag leds6 --deadline $((T0 + 84*60)) $COMMON
echo "== done in $(( ($(date +%s) - T0) / 60 )) min"
