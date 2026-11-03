"""D3: arrival detectors compared on dev-A against the calibrated image arrivals.

    <echo-lattice>/.venv/Scripts/python.exe -X utf8 d3_frontends.py

F0 is the frozen E10 picker. F1 picks peaks of the envelope of the response correlated with its own direct
sound, F2 peaks of the plain envelope; both are tried with several peak spacings and floors, with and
without ignoring the first millisecond after the direct sound (where the loudspeaker rings). Times are
taken relative to the detected direct sound. For every response:

- share: fraction of the picked peaks (direct excluded) within 2 cm of an image arrival of the calibrated
  geometry (up to fifth order per axis, the M7 lattice);
- baseline: the share expected if the same number of peaks were spread uniformly over the same span,
  i.e. the length of the union of the +-2 cm intervals inside the span divided by the span;
- recall: fraction of first-order arrivals from reflective surfaces that have a pick within 2 cm (and 5 cm).
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

import common as cm

TOL_M = 0.02


def union_length(centres: np.ndarray, half: float, lo: float, hi: float) -> float:
    a = np.sort(np.clip(centres - half, lo, hi))
    b = np.sort(np.clip(centres + half, lo, hi))
    total, cur_a, cur_b = 0.0, None, None
    for x, y in zip(a, b):
        if cur_a is None:
            cur_a, cur_b = x, y
        elif x <= cur_b:
            cur_b = max(cur_b, y)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = x, y
    if cur_a is not None:
        total += cur_b - cur_a
    return total


def detectors() -> dict:
    out = {"F0_frozen": None}
    for kind, sep, floor, mask in itertools.product(("F1_matched", "F2_envelope"), (6, 12, 24), (0.02, 0.05, 0.1), (0.0, 1.0)):
        out[f"{kind}_sep{sep}_floor{floor}_mask{mask:g}ms"] = (kind, sep, floor, mask)
    return out


def picks_for(h: np.ndarray, spec) -> np.ndarray:
    if spec is None:
        return cm.frozen_pick(h)
    kind, sep, floor, mask = spec
    signal = cm.matched_envelope(h)[0] if kind == "F1_matched" else cm.raw_envelope(h)
    p = cm.pick(signal, n_max=200, floor=floor, separation=sep)
    if mask > 0 and p.size > 1:
        keep = np.concatenate([[True], p[1:] - p[0] > mask * cm.RATE / 1000.0])
        p = p[keep]
    return p[:120]


def main() -> int:
    dev = cm.load_dev(cm.HERE / "data" / "dev_responses.npz")
    geo = cm.geometry()
    room = geo["room"]
    sel = np.flatnonzero(dev["split"] == "dev-A")
    specs = detectors()
    rows = []
    for n, idx in enumerate(sel):
        h = dev["rirs"][idx]
        code, j, i = str(dev["room_code"][idx]), int(dev["src_id"][idx]), int(dev["mic_id"][idx])
        mic, src = geo["mics"][i], geo["sources"][j]
        refl = cm.reflective(code)
        lattice = cm.images(room, mic, src, max_order=5, max_reflections=None)
        d0 = lattice[0]["distance"]
        rel_img = np.array([im["distance"] - d0 for im in lattice[1:]])
        first_refl = np.array([im["distance"] - d0 for im in lattice if im["order"] == 1 and all(refl[w] for w in im["walls"])])
        for name, spec in specs.items():
            p = picks_for(h, spec)
            rel = cm.to_m(p[1:] - p[0])
            if rel.size == 0:
                continue
            span = float(rel.max())
            gap = np.min(np.abs(rel[:, None] - rel_img[None, :]), axis=1)
            share = float(np.mean(gap <= TOL_M))
            baseline = union_length(rel_img[rel_img <= span + TOL_M], TOL_M, 0.0, span) / max(span, 1e-9)
            if first_refl.size:
                g = np.min(np.abs(first_refl[:, None] - rel[None, :]), axis=1)
                recall2, recall5 = float(np.mean(g <= 0.02)), float(np.mean(g <= 0.05))
            else:
                recall2 = recall5 = np.nan
            rows.append({"response": int(idx), "room_code": code, "src_id": j, "mic_id": i, "detector": name,
                         "n_picks": int(p.size), "span_ms": 1000 * span / cm.SPEED, "share": share,
                         "baseline": baseline, "ratio": share / baseline if baseline > 0 else np.nan,
                         "recall_2cm": recall2, "recall_5cm": recall5})
        if (n + 1) % 60 == 0:
            print(f"{n + 1}/{len(sel)}", flush=True)
    t = pd.DataFrame(rows)
    out = cm.HERE / "results"
    t.to_csv(out / "d3_frontends_dev-A.csv", index=False)
    summary = (t.groupby("detector")
                .agg(n=("response", "size"), picks=("n_picks", "median"), span_ms=("span_ms", "median"),
                     share=("share", "median"), baseline=("baseline", "median"), ratio=("ratio", "median"),
                     recall_2cm=("recall_2cm", "mean"), recall_5cm=("recall_5cm", "mean"))
                .sort_values("ratio", ascending=False))
    summary.to_csv(out / "d3_summary_dev-A.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary.round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
