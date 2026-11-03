"""Post-review E10 analyses PR4, PR5 and the E10 part of PR6. Exploratory, not pre-registered.

    <echo-lattice>/.venv/Scripts/python.exe post_review/e10_post_review.py --rirs <e10_responses.npz>

PR4: the near-image share of the picks of the 54 diagnostic responses (every eleventh response of the
E10 subset, as in e10_pick_diagnostics.py) at 2 cm, 0.1 ms and 0.5 ms, with and without the three
uncalibrated omnidirectional loudspeakers, against two random baselines (uniform redraw over the pick
span, circular shift of the picks within the span; 1,000 draws each); the same on the first 20 simulated
E2 responses. PR5 and PR6: for each of the 540 primary responses, the frozen model's Poisson
log-likelihood and the two flag fractions at the calibrated truth and at the stored M1, M2 and M7 fits,
and the explained fraction uniformly redrawn picks would reach. A stored fit is rebuilt from its sides,
positions and the latency sign that reproduces its stored fractions. No parameter is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e10_run as e  # noqa: E402
import harness  # noqa: E402

N_DRAWS = 1000
OMNI = {6, 7, 8}


def near_share(picks, images, tol):
    inside = images[(images >= picks[0] - tol) & (images <= picks[-1] + tol)]
    if not inside.size:
        return 0.0
    gap = np.abs(picks[:, None] - inside[None, :]).min(axis=1)
    return float(np.mean(gap < tol))


def baselines(picks, images, tol, rng):
    lo, hi = picks[0], picks[-1]
    uni = [near_share(np.sort(np.concatenate([[lo], rng.uniform(lo, hi, picks.size - 1)])), images, tol) for _ in range(N_DRAWS)]
    span = hi - lo
    shifted = []
    for _ in range(N_DRAWS):
        off = rng.uniform(0, span)
        moved = np.sort(lo + np.mod(picks - lo + off, span))
        shifted.append(near_share(moved, images, tol))
    return float(np.mean(uni)), float(np.mean(shifted))


def parse(vec):
    return np.array([float(x) for x in str(vec).replace("x", " ").split()]) if isinstance(vec, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "post_review")
    args = parser.parse_args()
    import pandas as pd
    import echo_lattice as pkg
    from echo_lattice import lattice_inversion as li

    args.out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    config = harness.base_config(3).override(freeze["e10"]["config_overrides"])
    model = pkg.echo_model(config)
    s = config.lattice
    lattice = li._Lattice(3, s.max_order)
    diag_lattice = li._Lattice(3, 5)
    width = li.FIT_WIDTHS_M[-1]
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        codes, srcs, mics = z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()
    latencies = e.artificial_latencies(len(codes))
    rng = np.random.default_rng([harness.MASTER_SEED, 4, 10])
    tols = {"2cm": 0.02, "0.1ms": 1e-4 * speed, "0.5ms": 5e-4 * speed}

    def response_picks(k):
        shift = round(latencies[k][0] * e.RATE / 1000.0)
        response = np.concatenate([np.zeros(shift), rirs[k]])[:4800]
        return np.sort(harness.picked_lengths(response, config)), shift

    def truth_theta(k, shift):
        return np.concatenate([geo["room"], geo["mics"][mics[k]], geo["sources"][srcs[k]], [shift / e.RATE * speed]])

    # ---------------- PR4: the 54 diagnostic responses
    pr4 = []
    for k in range(0, len(codes), 11):
        picks, shift = response_picks(k)
        images = np.sort(diag_lattice.lengths(truth_theta(k, shift))[0])
        row = {"index": k, "room_code": codes[k], "src_id": srcs[k], "mic_id": mics[k], "omni": srcs[k] in OMNI,
               "furnished": codes[k] == "020002", "picks": int(picks.size),
               "images_in_span": int(((images >= picks[0] - 0.02) & (images <= picks[-1] + 0.02)).sum())}
        for name, tol in tols.items():
            row[f"share_{name}"] = near_share(picks, images, tol)
            row[f"uniform_{name}"], row[f"shift_{name}"] = baselines(picks, images, tol, rng)
        pr4.append(row)
    sim = []
    for i, scene in enumerate(harness.scenes_for("E2", 20, 3)):
        cfg = harness.scene_config(harness.base_config(3), scene)
        weights, shift, _ = harness.measure(cfg, scene)
        picks = np.sort(harness.picked_lengths(weights, cfg))
        c = cfg.acoustics.speed
        images = np.sort(diag_lattice.lengths(np.concatenate([scene.size, scene.mic, scene.src, [shift / cfg.acoustics.sample_rate * c]]))[0])
        row = {"scene": i, "picks": int(picks.size)}
        for name, tol in {"2cm": 0.02, "0.1ms": 1e-4 * c, "0.5ms": 5e-4 * c}.items():
            row[f"share_{name}"] = near_share(picks, images, tol)
            row[f"uniform_{name}"], row[f"shift_{name}"] = baselines(picks, images, tol, rng)
        sim.append(row)
    pd.DataFrame(pr4).to_csv(args.out / "PR4_E10_near_image.csv", index=False)
    pd.DataFrame(sim).to_csv(args.out / "PR4_E2_near_image.csv", index=False)

    # ---------------- PR5 / PR6: the 540 primary responses
    res = pd.read_csv(HERE.parent / "results" / "E10.csv")
    res = res[res.condition.astype(str).str.zfill(6) != "020002"]
    order = e.subset_order(freeze)
    key = {(c, j, i): n for n, (c, j, i) in enumerate(order)}
    rows = []
    for _, g in res.groupby("scene"):
        first = g.iloc[0]
        code = str(first.room_code).zfill(6)
        k = key[(code, int(first.src_id), int(first.mic_id))]
        picks, shift = response_picks(k)
        truth = truth_theta(k, shift)
        out = {"scene": int(first.scene), "room_code": code, "src_id": int(first.src_id), "mic_id": int(first.mic_id),
               "source_type": first.source_type, "n_picks": int(picks.size)}

        def score(theta):
            rho, distance = lattice.lengths(theta)
            q0 = float(np.linalg.norm(theta[6:9] - theta[3:6]))
            prob = li.detection_probability(lattice.n_reflections, distance, q0, model)
            ll = li.point_process_loglik(picks, rho, prob, width, model)
            ex, mi = li._fit_quality(picks, theta, lattice, model)
            return ll, ex, mi

        out["truth_loglik"], out["truth_explained"], out["truth_missed"] = score(truth)
        for _, m in g.iterrows():
            tag = m.method
            out[f"{tag}_success"] = bool(m.success)
            out[f"{tag}_room_error_m"] = float(m.room_error_m)
            sides, mic, src = parse(m.est_sides), parse(m.est_microphone_m), parse(m.est_source_m)
            if sides is None or mic is None or src is None or not np.isfinite(m.latency_error_ms):
                continue
            best = None
            for sign in (+1, -1):
                lat_ms = float(m.true_latency_ms) + sign * float(m.latency_error_ms)
                theta = np.concatenate([sides, mic, src, [lat_ms / 1000.0 * speed]])
                ll, ex, mi = score(theta)
                gap = abs(ex - m.explained) + abs(mi - m.missed)
                if best is None or gap < best[0]:
                    best = (gap, theta, ll, ex, mi)
            gap, theta, ll, ex, mi = best
            out.update({f"{tag}_loglik": ll, f"{tag}_explained": ex, f"{tag}_missed": mi, f"{tag}_rebuild_gap": gap})
            if tag == "M1":
                lo, hi = picks[0], picks[-1]
                draws = [li._fit_quality(np.sort(np.concatenate([[lo], rng.uniform(lo, hi, picks.size - 1)])), theta, lattice, model)[0]
                         for _ in range(200)]
                out["M1_random_explained"] = float(np.mean(draws))
        rows.append(out)
    pd.DataFrame(rows).to_csv(args.out / "PR5_E10_scores.csv", index=False)
    meta = {"code": harness.code_identity(), "rirs_sha256": hashlib.sha256(args.rirs.read_bytes()).hexdigest(),
            "speed_m_s": speed, "tolerances_m": tols, "n_draws": N_DRAWS,
            "note": "post-review exploratory analysis, not pre-registered; dataset echo annotations (pickle) not loaded"}
    (args.out / "PR4_PR5_E10.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote PR4/PR5 outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
