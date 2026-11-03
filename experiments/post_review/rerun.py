"""Post-review reruns PR2, PR3, PR6 and PR7: rebuild the original responses and rerun M1.

Every scene and response comes from harness.plan and harness.measure with the original
master seed, so nothing is redrawn. Nothing frozen is changed. For each response this
records the M1 estimate (to compare with the original CSV), the latency standard error,
the spurious and missed picks against the true lattice, and the explained fraction that
uniformly redrawn picks would reach (random baseline).

    <echo-lattice>/.venv/Scripts/python.exe post_review/rerun.py --experiment E1 --workers 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402

CONDITIONS = {"E1": None, "E2": None, "E3": {"snr_0dB", "snr_10dB"}, "E5": {"R4_spurious_pulses"}}
WITH_M1 = {"E1", "E2", "E3"}
N_RANDOM = 200


def quality_counts(r, theta, lattice, model):
    """Spurious picks and missed strong arrivals of one lattice, as counts."""
    from echo_lattice import lattice_inversion as li

    rho, distance = lattice.lengths(theta)
    n_dims = lattice.n_dims
    low, high, _ = li._window(r, model)
    inside = (rho > low) & (rho < high)
    tol = max(2 * model.resolution_m, 0.02)
    q0 = float(np.linalg.norm(theta[2 * n_dims:3 * n_dims] - theta[n_dims:2 * n_dims]))
    prob = li.detection_probability(lattice.n_reflections, distance, q0, model)
    strong = np.sort(rho[inside & (prob >= 0.5 * model.max_detection)])
    near = li._nearest_gap(np.sort(rho[inside]), r) < tol if inside.any() else np.zeros(r.size, bool)
    missed = li._nearest_gap(r, strong) >= tol if strong.size else np.zeros(0, bool)
    return int(r.size - near.sum()), int(strong.size), int(missed.sum()), int(inside.sum())


def random_explained(r, theta, lattice, model, rng, n=N_RANDOM):
    """Mean explained fraction of picks redrawn uniformly over the same pick span."""
    from echo_lattice import lattice_inversion as li

    lo, hi = float(r.min()), float(r.max())
    vals = []
    for _ in range(n):
        fake = np.sort(np.concatenate([[lo], rng.uniform(lo, hi, r.size - 1)]))
        vals.append(li._fit_quality(fake, theta, lattice, model)[0])
    return float(np.mean(vals))


def run(spec: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import echo_lattice as pkg
    from echo_lattice import lattice_inversion as li

    scene = harness.Scene(**spec["scene"])
    n_dims = len(scene.size)
    config = harness.scene_config(harness.base_config(n_dims), scene, spec.get("config"))
    mismatch = spec.get("mismatch", {})
    weights, shift, _ = harness.measure(config, scene, spikes=mismatch.get("spikes", 0))
    a, s = config.acoustics, config.lattice
    true_latency_ms = 1000.0 * shift / a.sample_rate
    true_latency_m = 1e-3 * true_latency_ms * a.speed
    model = pkg.echo_model(config)
    lattice = li._Lattice(n_dims, s.max_order)
    r = np.sort(harness.picked_lengths(weights, config))
    truth = np.concatenate([scene.size, scene.mic, scene.src, [true_latency_m]])
    rng = np.random.default_rng([harness.MASTER_SEED, 77, spec["index"]])
    spurious, n_strong, n_missed, n_inside = quality_counts(r, truth, lattice, model)
    row = {"experiment": spec["experiment"], "condition": spec["condition"], "scene": spec["index"],
           "n_picks": int(r.size), "truth_spurious": spurious, "truth_strong": n_strong,
           "truth_missed_strong": n_missed, "truth_inside": n_inside,
           "truth_explained": li._fit_quality(r, truth, lattice, model)[0],
           "random_explained_truth": random_explained(r, truth, lattice, model, rng)}
    if spec["experiment"] in WITH_M1:
        started = time.perf_counter()
        try:
            est = harness.run_lattice(weights, config)
            row.update(harness.lattice_row(est, scene, a.speed, true_latency_ms))
            theta = np.concatenate([est.size_m, est.microphone_m, est.source_m, [est.latency_m]])
            row["latency_sd_m"] = float(est.latency_sd_m)
            row["latency_z"] = (est.latency_m - true_latency_m) / est.latency_sd_m if est.latency_sd_m > 0 else float("nan")
            row["random_explained_fit"] = random_explained(r, theta, lattice, model, rng)
        except Exception as exc:
            row.update(room_error_m=float("inf"), success=False, reliable=False, error=f"{type(exc).__name__}: {exc}")
        row["time_s"] = time.perf_counter() - started
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "post_review")
    args = parser.parse_args()
    import pandas as pd

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    keep = CONDITIONS[args.experiment]
    jobs = [j for j in harness.plan(args.experiment) if keep is None or j["condition"] in keep]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": args.experiment, "jobs": len(jobs), "master_seed": harness.MASTER_SEED,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": harness.code_identity(),
            "workers": args.workers,
            "n_random": N_RANDOM, "note": "post-review rerun, not pre-registered; no setting changed"}
    rows, failures = [], 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                failures += 1
                rows.append({"experiment": args.experiment, "error": repr(exc)})
            if k % 25 == 0 or k == len(futures):
                print(f"{args.experiment}: {k}/{len(futures)}, {time.perf_counter() - started:.0f} s", flush=True)
    table = pd.DataFrame(rows).sort_values(["condition", "scene"], na_position="last")
    table.to_csv(args.out / f"{args.experiment}_rerun.csv", index=False)
    meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                wall_s=time.perf_counter() - started, job_failures=failures)
    (args.out / f"{args.experiment}_rerun.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", args.out / f"{args.experiment}_rerun.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
