"""Render docs/tracker.html from docs/tracker_status.json and fetched training logs.

    tools/remote.sh fetch RUN/log.jsonl runs_cache/RUN/   (done by tools/update_tracker.sh)
    python tools/tracker.py
"""

import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "tracker_status.json"
CACHE = ROOT / "runs_cache"
OUT = ROOT / "docs" / "tracker.html"

METRICS = [
    ("win_rate", "Train win rate", "#5fd38d"),
    ("goals_for", "Goals for / match", "#5fb3f9"),
    ("goals_against", "Goals against / match", "#f2716b"),
    ("wall_deaths_pp", "Wall deaths / player / match", "#f2c14e"),
    ("pickups", "Pickups / match", "#c38df2"),
    ("entropy", "Policy entropy", "#8b98a8"),
]


def load_log(run):
    path = CACHE / run / "log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def sparkline(xs, ys, color, width=300, height=110):
    if len(ys) < 2:
        return "<div class='muted'>not enough data</div>"
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    x0, x1 = xs[0], xs[-1] if xs[-1] > xs[0] else xs[0] + 1
    pts = " ".join(
        f"{6 + (x - x0) / (x1 - x0) * (width - 12):.1f},{height - 18 - (y - lo) / (hi - lo) * (height - 30):.1f}"
        for x, y in zip(xs, ys))
    return (f"<svg viewBox='0 0 {width} {height}' width='100%'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
            f"<text x='6' y='{height - 4}' class='ax'>{x0 / 1e6:.0f}M</text>"
            f"<text x='{width - 6}' y='{height - 4}' class='ax' text-anchor='end'>{x1 / 1e6:.0f}M samples</text>"
            f"<text x='6' y='12' class='ax'>{hi:.3g}</text>"
            f"<text x='6' y='{height - 20}' class='ax'>{lo:.3g}</text></svg>")


def smooth(values, k=5):
    out = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def run_section(run):
    rows = load_log(run["name"])
    head = (f"<h3>{html.escape(run['name'])} <span class='tag'>{html.escape(run.get('state', ''))}</span></h3>"
            f"<p>{html.escape(run.get('description', ''))}</p>")
    if not rows:
        return head + "<p class='muted'>No log fetched yet.</p>"
    last = rows[-1]
    xs = [r["samples_total"] for r in rows]
    cards = "".join(
        f"<div class='chart'><div class='label'>{label} <b>{last.get(key, 0):.3g}</b></div>"
        f"{sparkline(xs, smooth([r.get(key, 0) for r in rows]), color)}</div>"
        for key, label, color in METRICS)
    opp = last.get("by_opponent", {})
    opp_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{v['n']}</td>"
        f"<td class='num'>{v['win_rate']:.2f}</td><td class='num'>{v['goal_diff']:+.2f}</td></tr>"
        for name, v in opp.items())
    hours = (last["time"] - rows[0]["time"]) / 3600 if len(rows) > 1 else 0
    meta = (f"<p class='muted'>update {last['update']} · {last['samples_total'] / 1e6:.0f}M samples · "
            f"{hours:.1f} h · {last.get('ticks_per_s', 0)} ticks/s</p>")
    table = ("<table><tr><th>Training opponent</th><th class='num'>matches</th>"
             "<th class='num'>win rate</th><th class='num'>goal diff</th></tr>" + opp_rows + "</table>"
             if opp_rows else "")
    return head + meta + f"<div class='charts'>{cards}</div>" + table


def render():
    status = json.loads(STATUS.read_text())
    milestones = "".join(
        f"<tr><td><b>{html.escape(m['id'])}</b></td><td>{html.escape(m['title'])}</td>"
        f"<td><span class='st {m['state']}'>{html.escape(m['state'])}</span></td>"
        f"<td>{html.escape(m.get('note', ''))}</td></tr>" for m in status["milestones"])
    updates = "".join(
        f"<li><span class='muted'>{html.escape(u['time'])}</span> {html.escape(u['text'])}</li>"
        for u in reversed(status["updates"]))
    evals = "".join(
        f"<tr>{''.join(f'<td>{html.escape(str(c))}</td>' for c in row)}</tr>"
        for row in status.get("eval_rows", []))
    eval_head = "".join(f"<th>{html.escape(c)}</th>" for c in status.get("eval_header", []))
    runs = "".join(run_section(run) for run in status["runs"])
    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>ExML CTF — progress tracker</title><style>
body{{margin:0;background:#0f1419;color:#dce3ea;font:15px/1.55 -apple-system,'Segoe UI',Roboto,Arial,sans-serif}}
main{{max-width:1150px;margin:0 auto;padding:28px 22px 70px}} h1{{margin:0 0 4px;font-size:28px}}
h2{{font-size:20px;margin:36px 0 10px;border-bottom:1px solid #2a3644;padding-bottom:6px}} h3{{color:#5fb3f9;margin:22px 0 4px}}
.muted{{color:#8b98a8}} table{{width:100%;border-collapse:collapse;background:#161d26;margin:8px 0 14px;font-size:14px}}
th,td{{padding:8px 11px;border-bottom:1px solid #2a3644;text-align:left;vertical-align:top}} th{{background:#1c2531;color:#8b98a8;font-size:12px;text-transform:uppercase}}
.num{{text-align:right}} .st{{padding:1px 9px;border-radius:999px;font-size:12px;border:1px solid #2a3644}}
.st.done{{color:#5fd38d;border-color:#2f6b49}} .st.active{{color:#5fb3f9;border-color:#2d4560}} .st.todo{{color:#8b98a8}} .st.blocked{{color:#f2716b;border-color:#6b302d}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}} .chart{{background:#161d26;border:1px solid #2a3644;border-radius:8px;padding:8px 10px}}
.label{{font-size:13px;color:#8b98a8}} .label b{{color:#dce3ea}} .ax{{fill:#8b98a8;font-size:10px}} .tag{{font-size:12px;color:#8b98a8;font-weight:400}}
.box{{background:#18273a;border:1px solid #2d4560;border-radius:10px;padding:12px 18px}} li{{margin:5px 0}}
</style></head><body><main>
<h1>ExML Capture the Flag — progress tracker</h1>
<div class='muted'>Last updated {time.strftime('%Y-%m-%d %H:%M')} · branch <code>main</code></div>
<h2>Where we are</h2><div class='box'>{status['summary']}</div>
<h2>Milestones</h2><table><tr><th>#</th><th>Milestone</th><th>State</th><th>Notes</th></tr>{milestones}</table>
<h2>Evaluation (held-out seeds, both sides)</h2>
{f"<table><tr>{eval_head}</tr>{evals}</table>" if evals else "<p class='muted'>No held-out evaluations yet.</p>"}
<h2>Training runs</h2>{runs}
<h2>Update log</h2><ul>{updates}</ul>
</main></body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
