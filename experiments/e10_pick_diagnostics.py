"""Exploratory: what the frozen echo picker sees in the measured responses (not part of the pre-specified analysis).

    <project>/.venv/Scripts/python.exe e10_pick_diagnostics.py --rirs <npz> [--out results]

For every eleventh response of the E10 subset (54 responses), with the frozen E10 configuration and the
same artificial latency as e10_run.py: how many peaks are picked, how long after the first pick the last
one comes, how many image arrivals of the calibrated geometry (order 5 per axis) fall in that span, and
which share of the picks lies within 2 cm of one. Writes E10_pick_diagnostics.json. Changes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import e10_run as e
import harness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=e.HERE / "results")
    args = parser.parse_args()
    from echo_lattice import lattice_inversion as li

    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    config = harness.base_config(3).override(freeze["e10"]["config_overrides"])
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        codes, srcs, mics = z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()
    latencies = e.artificial_latencies(len(codes))
    lattice = li._Lattice(3, 5)
    rows = []
    for k in range(0, len(codes), 11):
        shift = round(latencies[k][0] * e.RATE / 1000.0)
        response = np.concatenate([np.zeros(shift), rirs[k]])[:4800]
        picks = np.sort(harness.picked_lengths(response, config))
        theta = np.concatenate([geo["room"], geo["mics"][mics[k]], geo["sources"][srcs[k]], [shift / e.RATE * speed]])
        images = np.sort(lattice.lengths(theta)[0])
        inside = images[(images >= picks[0] - 0.02) & (images <= picks[-1] + 0.02)]
        near = np.array([np.min(np.abs(inside - x)) for x in picks]) if inside.size else np.full(picks.size, np.inf)
        rows.append({"index": k, "room_code": codes[k], "src_id": srcs[k], "mic_id": mics[k], "picks": int(picks.size),
                     "span_ms": float((picks[-1] - picks[0]) / speed * 1000.0), "images_in_span": int(inside.size),
                     "share_within_2cm": float(np.mean(near < 0.02))})
    # the same quantities on simulated 3-D responses (the first 20 scenes of E2) for contrast
    simulated = []
    for scene in harness.scenes_for("E2", 20, 3):
        cfg = harness.scene_config(harness.base_config(3), scene)
        weights, shift, _ = harness.measure(cfg, scene)
        picks = np.sort(harness.picked_lengths(weights, cfg))
        c = cfg.acoustics.speed
        theta = np.concatenate([scene.size, scene.mic, scene.src, [shift / cfg.acoustics.sample_rate * c]])
        images = np.sort(lattice.lengths(theta)[0])
        inside = images[(images >= picks[0] - 0.02) & (images <= picks[-1] + 0.02)]
        near = np.array([np.min(np.abs(inside - x)) for x in picks])
        simulated.append({"picks": int(picks.size), "span_ms": float((picks[-1] - picks[0]) / c * 1000.0),
                          "images_in_span": int(inside.size), "share_within_2cm": float(np.mean(near < 0.02))})

    def summarize(items):
        return {key: {"median": float(np.median([r[key] for r in items])),
                      "q25": float(np.percentile([r[key] for r in items], 25)),
                      "q75": float(np.percentile([r[key] for r in items], 75))}
                for key in ("picks", "span_ms", "images_in_span", "share_within_2cm")}

    summary = {"measured_e10": summarize(rows), "simulated_e2_first_20": summarize(simulated)}
    out = {"note": "exploratory, not pre-specified", "responses": len(rows), "summary": summary, "rows": rows,
           "simulated_rows": simulated}
    (args.out / "E10_pick_diagnostics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
