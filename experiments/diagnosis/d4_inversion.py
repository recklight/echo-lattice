"""D4: the lattice inversion end to end on dev-A with other arrival detectors (confirmatory; G2 already failed).

    <echo-lattice>/.venv/Scripts/python.exe -X utf8 d4_inversion.py --workers 14

Runs M1 (latency estimated, 3-D settings of E10) on the dev-A responses of the four loudspeakers facing
the room, with the picks of the detectors listed below. The frozen detection model is used, and once with
a larger clutter count. An artificial latency U(0, 20) ms is added to every response's path lengths with a
fixed seed, as in E10. Success is the summed error of the sorted sides below 5 cm and below 10 cm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import common as cm

RUNS = {
    "F0_frozen/clutter3": ("F0", 3.0),
    "F2_envelope_sep6_floor0.02_mask1ms/clutter3": ("F2", 3.0),
    "F2_envelope_sep6_floor0.02_mask1ms/clutter60": ("F2", 60.0),
    "F1_matched_sep6_floor0.02_mask1ms/clutter3": ("F1", 3.0),
}


def picks(h, kind):
    if kind == "F0":
        return cm.frozen_pick(h)
    signal = cm.matched_envelope(h)[0] if kind == "F1" else cm.raw_envelope(h)
    p = cm.pick(signal, n_max=200, floor=0.02, separation=6)
    keep = np.concatenate([[True], p[1:] - p[0] > cm.RATE / 1000.0])
    return p[keep][:120]


def job(spec):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import echo_lattice as pkg
    from echo_lattice.lattice_inversion import EchoModel, invert_lattice

    config = cm.e10_run.harness.base_config(3).override({"acoustics.speed_of_sound_m_s": cm.SPEED, "acoustics.temperature_c": None})
    base = pkg.echo_model(config)
    room = np.asarray(spec["room"])
    p = picks(np.asarray(spec["h"]), spec["kind"])
    r = cm.to_m(p) + spec["latency_m"]
    model = EchoModel(relative_floor=base.relative_floor, reflection=base.reflection, clutter=spec["clutter"],
                      max_detection=base.max_detection, resolution_m=base.resolution_m)
    row = {k: spec[k] for k in ("run", "response", "room_code", "src_id", "mic_id")}
    started = time.perf_counter()
    try:
        est = invert_lattice(r, n_dims=3, model=model, max_order=config.lattice.max_order)
        err = float(np.abs(np.sort(est.size_m)[::-1] - np.sort(room)[::-1]).sum())
        row.update(error_m=err, explained=est.explained_fraction, missed=est.missed_fraction, reliable=est.reliable,
                   est_sides=" x ".join(f"{s:.3f}" for s in est.size_m))
    except Exception as exc:  # no hypothesis and the like: a failure
        row.update(error_m=np.inf, explained=np.nan, missed=np.nan, reliable=False, error=f"{type(exc).__name__}: {exc}")
    row["time_s"] = time.perf_counter() - started
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    dev = cm.load_dev(cm.HERE / "data" / "dev_responses.npz")
    geo = cm.geometry()
    sel = [k for k in np.flatnonzero(dev["split"] == "dev-A") if int(dev["src_id"][k]) <= 3]
    rng = np.random.default_rng(20261103)
    latency = {int(k): float(rng.uniform(0.0, 20.0)) * 1e-3 * cm.SPEED for k in sel}
    specs = []
    for run, (kind, clutter) in RUNS.items():
        for k in sel:
            specs.append({"run": run, "kind": kind, "clutter": clutter, "response": int(k), "h": dev["rirs"][k][: cm.WINDOW],
                          "room_code": str(dev["room_code"][k]), "src_id": int(dev["src_id"][k]), "mic_id": int(dev["mic_id"][k]),
                          "room": geo["room"].tolist(), "latency_m": latency[int(k)]})
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, s) for s in specs]
        for n, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if n % 60 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} fits, {time.perf_counter() - started:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    out = cm.HERE / "results"
    t.to_csv(out / "d4_inversion_dev-A.csv", index=False)
    summary = {}
    for run, u in t.groupby("run"):
        ok5, ok10 = u.error_m < 0.05, u.error_m < 0.10
        rel = u.reliable.astype(bool)
        summary[run] = {"n": int(len(u)), "median_error_m": float(u.error_m.median()), "success_5cm": float(ok5.mean()),
                        "success_10cm": float(ok10.mean()), "no_solution": int(np.isinf(u.error_m).sum()),
                        "flagged_reliable": int(rel.sum()), "flag_precision_10cm": float(ok10[rel].mean()) if rel.any() else None}
    (out / "d4_summary_dev-A.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
