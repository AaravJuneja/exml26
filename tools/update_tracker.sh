#!/usr/bin/env bash
# Copy training logs for every run listed in docs/tracker_status.json, then render the HTML.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
for run in $($PY -c "import json;print(' '.join(r['name'] for r in json.load(open('docs/tracker_status.json'))['runs']))"); do
  mkdir -p "runs_cache/$run"
  for f in log.jsonl eval.jsonl; do
    if [ -f "runs/$run/$f" ]; then
      cp "runs/$run/$f" "runs_cache/$run/"
    fi
  done
done
$PY tools/tracker.py
