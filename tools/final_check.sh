#!/bin/bash
# Final verification from a fresh clone of origin/main (laptop).
set -u
PY=python
D=/tmp/exml-final-clone
rm -rf $D && git clone -q "$(git -C . remote get-url origin)" $D
cd $D || exit 1
echo "== commit $(git log --oneline -1)"
echo "== engine/rpc/main.py unchanged vs ../game-env:"
for f in main.py $(cd ../game-env && find engine rpc -name '*.py' | sort); do
  cmp -s "$f" "../game-env/$f" || echo "   DIFFERS: $f"
done
echo "== release reports and model hashes"
$PY - <<'PY'
import json, hashlib
from pathlib import Path
for t in ("exml_rl", "exml_rl_pure"):
    r = json.loads(Path(f"teams/{t}/release_report.json").read_text())
    h = hashlib.sha256(Path(f"teams/{t}/model.py").read_bytes()).hexdigest()[:16]
    print(f"   {t}: {r['status']} {r['checks']} model.py {h} decode={r['policy'].get('decode')}")
PY
echo "== submission/ files identical to the releases"
cmp -s submission/hybrid/model.py teams/exml_rl/model.py && echo "   hybrid == teams/exml_rl" || echo "   MISMATCH: submission/hybrid"
cmp -s submission/pure/model.py teams/exml_rl_pure/model.py && echo "   pure == teams/exml_rl_pure" || echo "   MISMATCH: submission/pure"
echo "== official main.py matches (real engine + RPC workers)"
for pair in "submission/hybrid/model.py teams/baseline/model.py" "submission/pure/model.py teams/example_team/model.py" "submission/hybrid/model.py submission/pure/model.py"; do
  set -- $pair
  printf '[{"name": "red", "model_path": "%s", "color": "#ff4d4d"}, {"name": "blue", "model_path": "%s", "color": "#3d9bff"}]\n' $1 $2 > teams.json
  timeout 900 $PY main.py 7 2>&1 | grep -E "match over|Error|Traceback|timed out" | head -3 | sed "s#^#   $1 vs $2: #"
done
git checkout -q teams.json
echo "== full test suite"
mkdir -p $D.tmp && TMPDIR=$D.tmp $PY -m pytest -q -p no:cacheprovider --basetemp=$D.tmp/base ctf/tests 2>&1 | tail -n 1; rm -rf $D.tmp
