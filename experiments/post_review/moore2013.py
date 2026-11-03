"""Post-review experiment PR1: Moore, Brookes and Naylor (2013) on the E1 responses.

Moore et al. (EUSIPCO 2013) estimate a planar rectangular room from the labelled arrival
times of one impulse response with an unknown time offset. Their Section 3.4 minimises
the squared differences between all measured and predicted time differences of arrival
(their eq. 26) over theta = (x0, y0, la, lb, lc, ld): a grid of candidate source positions
(x0, y0) relative to the microphone, and for each the four wall distances by a
trust-region reflective least-squares solve. This script follows that recipe on the same
500 identified E1 responses as M1. The labels are those of M7 (picks matched to the true
images within max(2 samples, 2 cm)), restricted to the 13 images of their Table 1 (two
reflections or fewer). The wall-distance starts come from their eqs. (16)-(17) given the
grid point; the best grid point is then refined on the same cost with all six parameters
free. Their eq. (15) gives the source-microphone distance from an adjacent second-order
reflection and is recorded as a check. Success is the paper's criterion (sorted side
error < 5 cm). Post-review experiment, not pre-registered; nothing frozen is changed.

    <echo-lattice>/.venv/Scripts/python.exe post_review/moore2013.py --workers 14
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

GRID_STEP_M = 0.10
GRID_RADIUS_M = 3.0


def labelled_images(r, truth, lattice, model):
    """M7's assignment: each pick to its nearest true image within tolerance, one pick per image."""
    rho, _ = lattice.lengths(truth)
    tol = max(2 * model.resolution_m, 0.02)
    gap = np.abs(r[:, None] - rho[None, :])
    nearest = np.argmin(gap, axis=1)
    ok = gap[np.arange(r.size), nearest] < tol
    echoes, images = np.flatnonzero(ok), nearest[ok]
    keep = np.unique(images, return_index=True)[1]
    return echoes[keep], images[keep]


def to_theta(x0, y0, la, lb, lc, ld):
    """Moore's microphone-centred parameters to (L_x, L_y, m_x, m_y, s_x, s_y, 0)."""
    return np.array([la + lc, lb + ld, la, lb, la + x0, lb + y0, 0.0])


def moore_fit(r_lab, images, lattice):
    from scipy.optimize import least_squares

    k = lattice.mult[images] - np.where(lattice.sign[images] < 0, 1.0, 0.0)  # recover the index
    pairs = np.array([(i, j) for i in range(r_lab.size) for j in range(i + 1, r_lab.size)])
    measured = r_lab[pairs[:, 0]] - r_lab[pairs[:, 1]]

    def predicted(th6):
        return lattice.lengths(to_theta(*th6))[1][images]

    def res(th6):
        p = predicted(th6)
        return (p[pairs[:, 0]] - p[pairs[:, 1]]) - measured

    direct = np.flatnonzero((k == 0).all(axis=1))
    d = {}
    if direct.size:
        r0 = r_lab[direct[0]]
        for idx in range(images.size):
            key = tuple(int(v) for v in k[idx])
            d[key] = r_lab[idx] - r0
    best = (np.inf, None)
    lo4, hi4 = np.full(4, 0.05), np.full(4, 30.0)
    for x0 in np.arange(-GRID_RADIUS_M, GRID_RADIUS_M + 1e-9, GRID_STEP_M):
        for y0 in np.arange(-GRID_RADIUS_M, GRID_RADIUS_M + 1e-9, GRID_STEP_M):
            R0 = float(np.hypot(x0, y0))
            if R0 < 0.05 or R0 > GRID_RADIUS_M:
                continue
            start = []
            # eq. (16)-(17): (d_w + R0)^2 = (2 l_w +/- c)^2 + c_perp^2 for the wall on each side
            for key, along, perp, sign in (((-1, 0), x0, y0, +1), ((0, -1), y0, x0, +1), ((1, 0), x0, y0, -1), ((0, 1), y0, x0, -1)):
                if key in d and (d[key] + R0) ** 2 > perp ** 2:
                    start.append(max(0.06, 0.5 * (np.sqrt((d[key] + R0) ** 2 - perp ** 2) - sign * along)))
                else:
                    start.append(2.0)
            sol = least_squares(lambda w: res(np.concatenate([[x0, y0], w])), np.clip(start, lo4 + 1e-3, hi4 - 1e-3),
                                method="trf", bounds=(lo4, hi4), max_nfev=200)
            if sol.cost < best[0]:
                best = (sol.cost, np.concatenate([[x0, y0], sol.x]))
    lo6 = np.concatenate([[-GRID_RADIUS_M - 1, -GRID_RADIUS_M - 1], lo4])
    hi6 = np.concatenate([[GRID_RADIUS_M + 1, GRID_RADIUS_M + 1], hi4])
    sol = least_squares(res, np.clip(best[1], lo6 + 1e-6, hi6 - 1e-6), method="trf", bounds=(lo6, hi6), max_nfev=400)
    return sol.x, float(sol.cost), float(best[0]), d


def eq15_r0(d):
    """Moore eq. (15) from any adjacent second-order reflection and its two first-order ones."""
    out = []
    for (kx, ky) in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        a, b, ab = d.get((kx, 0)), d.get((0, ky)), d.get((kx, ky))
        if a is not None and b is not None and ab is not None and abs(ab - a - b) > 1e-9:
            out.append(0.5 * (a * a + b * b - ab * ab) / (ab - a - b))
    return out


def run(spec: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import echo_lattice as pkg
    from echo_lattice import lattice_inversion as li

    scene = harness.Scene(**spec["scene"])
    config = harness.scene_config(harness.base_config(2), scene, spec.get("config"))
    weights, shift, _ = harness.measure(config, scene)
    a, s = config.acoustics, config.lattice
    model = pkg.echo_model(config)
    lattice = li._Lattice(2, s.max_order)
    r = np.sort(harness.picked_lengths(weights, config))
    true_latency_m = 1e-3 * (1000.0 * shift / a.sample_rate) * a.speed
    truth = np.concatenate([scene.size, scene.mic, scene.src, [true_latency_m]])
    echoes, images = labelled_images(r, truth, lattice, model)
    low_order = lattice.n_reflections[images] <= 2
    echoes, images = echoes[low_order], images[low_order]
    row = {"experiment": "PR1_moore2013", "scene": spec["index"], "true_sides": " x ".join(f"{x:.3f}" for x in scene.size),
           "n_labelled_order2": int(images.size), "has_direct": bool((lattice.n_reflections[images] == 0).any())}
    started = time.perf_counter()
    try:
        if images.size < 5 or not row["has_direct"]:
            raise ValueError("fewer than five labelled arrivals up to second order, or no direct sound")
        th6, cost, grid_cost, d = moore_fit(r[echoes], images, lattice)
        theta = to_theta(*th6)
        err = pkg.room_size_error(theta[:2], scene.size)
        r0s = eq15_r0(d)
        true_r0 = float(np.linalg.norm(np.subtract(scene.src, scene.mic)))
        row.update(est_sides=" x ".join(f"{x:.4f}" for x in np.sort(theta[:2])[::-1]), room_error_m=float(err),
                   success=bool(err < harness.SUCCESS_M), final_cost=cost, grid_cost=grid_cost,
                   n_adjacent=len(r0s), eq15_r0_error_m=(float(np.median(np.abs(np.array(r0s) - true_r0))) if r0s else float("nan")))
    except Exception as exc:
        row.update(room_error_m=float("inf"), success=False, error=f"{type(exc).__name__}: {exc}")
    row["time_s"] = time.perf_counter() - started
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "post_review")
    args = parser.parse_args()
    import pandas as pd

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    jobs = [{"index": i, "scene": sc.__dict__, "config": {}} for i, sc in enumerate(harness.scenes_for("E1", args.n or 500, 2))]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": "PR1_moore2013", "jobs": len(jobs), "master_seed": harness.MASTER_SEED,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": harness.code_identity(),
            "grid_step_m": GRID_STEP_M,
            "grid_radius_m": GRID_RADIUS_M, "workers": args.workers,
            "note": "post-review experiment, not pre-registered; labels from M7, images up to second order"}
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if k % 25 == 0 or k == len(futures):
                print(f"PR1: {k}/{len(futures)}, {time.perf_counter() - started:.0f} s", flush=True)
    stem = "PR1_moore2013" + (f"_n{args.n}" if args.n else "")
    pd.DataFrame(rows).sort_values("scene").to_csv(args.out / f"{stem}.csv", index=False)
    meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=time.perf_counter() - started)
    (args.out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", args.out / f"{stem}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
