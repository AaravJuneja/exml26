#!/usr/bin/env bash
# Fetch training logs for every run listed in docs/tracker_status.json, then render the HTML.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
for run in $($PY -c "import json;print(' '.join(r['name'] for r in json.load(open('docs/tracker_status.json'))['runs']))"); do
  tools/remote.sh fetch "$run/log.jsonl" "runs_cache/$run/" 2>/dev/null || true
  tools/remote.sh fetch "$run/eval.jsonl" "runs_cache/$run/" 2>/dev/null || true
done
$PY tools/tracker.py
