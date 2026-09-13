#!/usr/bin/env python
"""JAX-profiler-style memory report from a torch CUDA memory-history snapshot.

Produces ONE self-contained HTML per snapshot with exactly two things
(TensorBoard memory_viewer style — no timeline lanes, no trace UI):

  1. the memory-allocation curve (allocated MB vs time), peak marked;
  2. a breakdown of the PEAK: every block alive at the peak instant, grouped
     by allocating source (innermost model-code frame), sorted by bytes,
     with expandable rows showing block sizes and the full python stack.

Optionally crop away a leading phase first (e.g. the ESM-C language model)
with --crop-after REGEX — the peak is then the peak of the remaining
(ESMFold) region.

Usage:
  python scripts/memsnap_peak_report.py SNAP.pickle [SNAP2 ...]
      [--crop-after 'compute_lm_hidden_states|esmc'] [--min-mb 0]
Output: <snap>.memreport.html (or <snap>.fold.memreport.html when cropped)
"""

from __future__ import annotations

import argparse
import html
import pickle
import re
from pathlib import Path

from memsnap_to_trace import _pick_name, _py_frames, _stack_lines  # noqa: E402

_GB = 2**30
_MB = 2**20


def build_report(snap_path: Path, out_path: Path, crop_after: str | None,
                 min_mb: float, device: int) -> None:
    snap = pickle.load(open(snap_path, "rb"))
    trace = snap["device_traces"][device]
    t0 = min(e["time_us"] for e in trace)

    t_crop = 0
    if crop_after:
        pat = re.compile(crop_after)
        for e in trace:
            if e["action"] == "alloc" and any(
                    pat.search(f["filename"]) or pat.search(f["name"])
                    for f in _py_frames(e.get("frames") or [])):
                t_crop = max(t_crop, e["time_us"] - t0 + 1)

    # Replay the stream: allocation curve + live set; find the peak in the
    # (possibly cropped) region.
    allocated = 0
    curve: list[tuple[int, int]] = []  # (ts, allocated)
    live: dict[int, dict] = {}
    peak = (-1, -1)  # (bytes, ts)
    peak_live: list[dict] | None = None
    for e in trace:
        ts = e["time_us"] - t0
        if e["action"] == "alloc":
            allocated += e["size"]
            live[e["addr"]] = e
        elif e["action"] == "free_completed":
            allocated -= e["size"]
            live.pop(e["addr"], None)
        else:
            continue
        if ts >= t_crop:
            curve.append((ts, allocated))
            if allocated > peak[0]:
                peak = (allocated, ts)
                peak_live = list(live.values())
    assert peak_live is not None, "empty region after crop"
    peak_b, peak_ts = peak

    # Group the live-at-peak blocks by source.
    groups: dict[str, dict] = {}
    for e in peak_live:
        name = _pick_name(e.get("frames") or [])
        g = groups.setdefault(name, {"bytes": 0, "blocks": [], "frames": None})
        g["bytes"] += e["size"]
        g["blocks"].append(e["size"])
        if g["frames"] is None or e["size"] > max(g["blocks"][:-1], default=0):
            g["frames"] = e.get("frames") or []
    rows = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
    min_b = int(min_mb * _MB)

    # Downsample the curve to <= 4000 points, keeping local maxima.
    if len(curve) > 4000:
        bucket = len(curve) // 2000
        ds = []
        for i in range(0, len(curve), bucket):
            chunk = curve[i:i + bucket]
            ds.append(chunk[0])
            mx = max(chunk, key=lambda p: p[1])
            if mx is not chunk[0]:
                ds.append(mx)
        curve = ds

    # ---- SVG chart -------------------------------------------------------- #
    W, H, PAD = 1000, 260, 45
    ts0, ts1 = curve[0][0], curve[-1][0]
    ymax = max(p[1] for p in curve) * 1.05
    def sx(t): return PAD + (t - ts0) / max(ts1 - ts0, 1) * (W - 2 * PAD)
    def sy(v): return H - PAD + (-v / ymax) * (H - 2 * PAD)
    pts = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in curve)
    gridlines = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        v = ymax * frac
        gridlines.append(
            f'<line x1="{PAD}" y1="{sy(v):.1f}" x2="{W-PAD}" y2="{sy(v):.1f}" class="grid"/>'
            f'<text x="{PAD-6}" y="{sy(v)+4:.1f}" class="ylab">{v/_GB:.1f} GB</text>')
    xticks = []
    for i in range(6):
        t = ts0 + (ts1 - ts0) * i / 5
        xticks.append(f'<text x="{sx(t):.1f}" y="{H-PAD+16}" class="xlab">{t/1e6:.1f}s</text>')
    svg = f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  {''.join(gridlines)}{''.join(xticks)}
  <polyline points="{pts}" fill="none" stroke="var(--line)" stroke-width="1.4"/>
  <line x1="{sx(peak_ts):.1f}" y1="{PAD}" x2="{sx(peak_ts):.1f}" y2="{H-PAD}" class="peakline"/>
  <text x="{min(sx(peak_ts)+6, W-260):.1f}" y="{PAD+14}" class="peaklab">peak {peak_b/_GB:.2f} GB @ {peak_ts/1e6:.2f}s</text>
</svg>"""

    # ---- breakdown table ---------------------------------------------------#
    body_rows = []
    shown = 0
    for name, g in rows:
        if g["bytes"] < min_b:
            continue
        shown += g["bytes"]
        blocks = sorted(g["blocks"], reverse=True)
        blk_txt = ", ".join(f"{b/_MB:.1f}" for b in blocks[:12])
        if len(blocks) > 12:
            blk_txt += f", … (+{len(blocks)-12})"
        stack = "\n".join(html.escape(s) for s in _stack_lines(g["frames"], 30))
        pct = 100.0 * g["bytes"] / peak_b
        body_rows.append(f"""
<tr><td class="num">{g['bytes']/_MB:,.1f}</td><td class="num">{pct:.1f}%</td>
<td class="num">{len(blocks)}</td>
<td><details><summary><code>{html.escape(name)}</code>
<span class="bar" style="width:{pct*4:.0f}px"></span></summary>
<div class="det">block sizes (MB): {blk_txt}<pre>{stack}</pre></div></details></td></tr>""")
    other = peak_b - shown

    title = f"{snap_path.stem}{' · ESM-C cropped' if t_crop else ''}"
    out_path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ --line:#2563eb; --fg:#111; --mut:#666; color-scheme: light; }}
body {{ font: 13px/1.45 system-ui, sans-serif; margin: 24px auto; max-width: 1040px; color: var(--fg); background:#fafafa; }}
h1 {{ font-size: 17px; }} h2 {{ font-size: 14px; margin-top: 28px; }}
.grid {{ stroke:#ddd; stroke-width:1; }} .ylab,.xlab {{ font-size:10px; fill:#666; }}
.ylab {{ text-anchor:end; }} .xlab {{ text-anchor:middle; }}
.peakline {{ stroke:#dc2626; stroke-dasharray:4 3; }} .peaklab {{ fill:#dc2626; font-size:11px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ padding: 3px 8px; border-bottom: 1px solid #e5e5e5; vertical-align: top; text-align:left; }}
td.num, th.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.bar {{ display:inline-block; height:9px; background:#93c5fd; margin-left:8px; vertical-align:middle; }}
summary {{ cursor: pointer; list-style:none; }} summary::-webkit-details-marker {{ display:none; }}
.det {{ color: var(--mut); margin: 4px 0 8px 0; }} pre {{ font-size: 11px; overflow-x:auto; background:#f1f1f1; padding:6px; }}
.meta {{ color: var(--mut); }}
</style>
<h1>GPU memory report — {html.escape(title)}</h1>
<p class="meta">window {ts0/1e6:.1f}–{ts1/1e6:.1f} s{f' (cropped at {t_crop/1e6:.1f} s: everything before the last match of <code>{html.escape(crop_after)}</code> removed)' if t_crop else ''}
 · <b>peak allocated {peak_b/_GB:.2f} GB</b> at {peak_ts/1e6:.2f} s · {len(peak_live)} live blocks at peak</p>
{svg}
<h2>What makes up the peak (live blocks at {peak_ts/1e6:.2f} s, grouped by allocating source — click a row for block sizes + full stack)</h2>
<table>
<tr><th class="num">MB</th><th class="num">% of peak</th><th class="num">#blocks</th><th>source</th></tr>
{''.join(body_rows)}
<tr><td class="num">{other/_MB:,.1f}</td><td class="num">{100.0*other/peak_b:.1f}%</td><td class="num">—</td><td class="meta">everything below {min_mb} MB per source</td></tr>
</table>
""", encoding="utf-8")
    print(f"{snap_path.name}: peak {peak_b/_GB:.2f} GB @ {peak_ts/1e6:.2f}s, "
          f"{len(peak_live)} live blocks, {len(body_rows)} sources shown -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path, nargs="+")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--crop-after", default=None, metavar="REGEX")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="hide sources totalling less than this at peak")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    if args.out is not None and len(args.snapshot) > 1:
        raise SystemExit("-o only valid with a single snapshot")
    for sp in args.snapshot:
        suffix = ".fold.memreport.html" if args.crop_after else ".memreport.html"
        out = args.out or sp.with_suffix(suffix)
        build_report(sp, out, args.crop_after, args.min_mb, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
