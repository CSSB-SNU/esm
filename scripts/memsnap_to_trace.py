#!/usr/bin/env python
"""Convert a torch CUDA memory-history snapshot (.pickle from
``torch.cuda.memory._dump_snapshot``) into a Perfetto/Chrome trace.

Why: pytorch.org/memory_viz shows the same data but zooming is clumsy and the
allocating source is hard to read per block. Perfetto gives free pan/zoom
(WASD), search, and per-block args on click.

What the trace contains
  - Counter track  "GPU memory"  (allocated_MB / reserved_MB) — memory over
    time, fed by every alloc/free/segment event.
  - One "X" (complete) event PER ALLOCATION with size >= --min-mb, laid out on
    lanes so overlapping lifetimes stack like a flame graph:
        row  = lifetime of one allocation (alloc -> free_completed)
        name = the most model-relevant python frame (func (file:line))
        args = size, stream, full python stack (click a block to see it)
    Small allocations are left out of the lane view (they would be millions of
    slivers) but still shape the counter track.

Open the output at https://ui.perfetto.dev ("Open trace file"; .json.gz is
accepted directly) or chrome://tracing.

Usage:
  python scripts/memsnap_to_trace.py SNAP.pickle [SNAP2.pickle ...]
      [-o OUT.json.gz] [--min-mb 1] [--device 0]
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import pickle
from pathlib import Path

# Frames whose filename matches these are what the user thinks of as "the
# source" — innermost model code wins over torch/library internals. NOTE: do
# not use a plain "/esm/" hint — the repo root itself is named esm, so every
# .venv path under it would match.
_MODEL_HINTS = ("modeling_esmfold2", "miniworld", "/esm/models/", "/esm/utils/",
                "/scripts/", "/kernels/")


def _py_frames(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f.get("filename", "").endswith(".py")]


def _is_model_frame(f: dict) -> bool:
    fn = f["filename"]
    if "/torch/" in fn:
        return False
    if any(h in fn for h in _MODEL_HINTS):
        return True
    return "site-packages" not in fn and ".venv" not in fn  # user code


def _pick_name(frames: list[dict]) -> str:
    py = _py_frames(frames)
    for f in py:  # innermost model-code frame
        if _is_model_frame(f):
            return f"{f['name']} ({Path(f['filename']).name}:{f['line']})"
    if py:  # else innermost python frame (torch internals)
        f = py[0]
        return f"{f['name']} ({Path(f['filename']).name}:{f['line']})"
    return "<no python frame>"


def _stack_lines(frames: list[dict], limit: int = 25) -> list[str]:
    return [f"{f['filename']}:{f['line']}  {f['name']}"
            for f in _py_frames(frames)[:limit]]


def convert(snap_path: Path, out_path: Path, min_mb: float, device: int) -> None:
    snap = pickle.load(open(snap_path, "rb"))
    trace = snap["device_traces"][device]
    if not trace:
        raise SystemExit(f"device {device} has no events in {snap_path}")

    t0 = min(e["time_us"] for e in trace)
    events: list[dict] = [
        {"ph": "M", "pid": 0, "name": "process_name",
         "args": {"name": f"GPU{device} memory ({snap_path.name})"}},
        {"ph": "M", "pid": 1, "name": "process_name",
         "args": {"name": f"allocations >= {min_mb} MB (lanes = concurrent lifetimes)"}},
    ]

    # ---- counter track (every event) ------------------------------------- #
    allocated = reserved = peak = 0
    live: dict[int, dict] = {}  # addr -> block
    blocks: list[dict] = []
    for e in trace:
        a, ts = e["action"], e["time_us"] - t0
        if a == "alloc":
            allocated += e["size"]
            peak = max(peak, allocated)
            live[e["addr"]] = {"ts": ts, "size": e["size"],
                               "frames": e.get("frames") or [],
                               "stream": e["stream"]}
        elif a == "free_completed":
            allocated -= e["size"]
            b = live.pop(e["addr"], None)
            if b is not None:
                b["end"] = ts
                blocks.append(b)
        elif a in ("segment_alloc", "segment_map"):
            reserved += e["size"]
        elif a in ("segment_free", "segment_unmap"):
            reserved -= e["size"]
        else:
            continue
        events.append({"ph": "C", "pid": 0, "name": "GPU memory", "ts": ts,
                       "args": {"allocated_MB": round(allocated / 2**20, 1),
                                "reserved_MB": round(reserved / 2**20, 1)}})
    t_end = trace[-1]["time_us"] - t0
    for b in live.values():  # still live at snapshot end
        b["end"] = t_end
        blocks.append(b)

    # ---- per-allocation lane events --------------------------------------- #
    min_bytes = int(min_mb * 2**20)
    big = sorted((b for b in blocks if b["size"] >= min_bytes),
                 key=lambda b: b["ts"])
    busy: list[tuple[int, int]] = []  # heap of (end_ts, lane)
    free_lanes: list[int] = []
    next_lane = 0
    for b in big:
        while busy and busy[0][0] <= b["ts"]:
            heapq.heappush(free_lanes, heapq.heappop(busy)[1])
        lane = heapq.heappop(free_lanes) if free_lanes else next_lane
        if lane == next_lane:
            next_lane += 1
        heapq.heappush(busy, (b["end"], lane))
        events.append({
            "ph": "X", "pid": 1, "tid": lane,
            "ts": b["ts"], "dur": max(b["end"] - b["ts"], 1),
            "name": _pick_name(b["frames"]),
            "args": {"size_MB": round(b["size"] / 2**20, 2),
                     "stream": b["stream"],
                     "stack": _stack_lines(b["frames"])},
        })

    payload = json.dumps({"traceEvents": events,
                          "displayTimeUnit": "ms"}).encode()
    opener = gzip.open if out_path.suffix == ".gz" else open
    with opener(out_path, "wb") as fh:
        fh.write(payload)
    print(f"{snap_path.name}: {len(trace)} events -> {len(blocks)} allocations "
          f"({len(big)} >= {min_mb} MB on {next_lane} lanes), "
          f"peak allocated {peak / 2**20:.0f} MB")
    print(f"  -> {out_path} ({out_path.stat().st_size / 2**20:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path, nargs="+")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (single input only); "
                         "default <snap>.trace.json.gz")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="lane view includes allocations >= this size (MB)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    if args.out is not None and len(args.snapshot) > 1:
        raise SystemExit("-o only valid with a single snapshot")
    for sp in args.snapshot:
        out = args.out or sp.with_suffix(".trace.json.gz")
        convert(sp, out, args.min_mb, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
