"""D1 timing of measured echoes against the calibrated geometry, and D2 the fit with labels given.

    <echo-lattice>/.venv/Scripts/python.exe -X utf8 d1_d2_timing.py [dev-A|dev-B]

D1: for every response of the split and every image up to second order, the measured arrival is the
maximum within +-0.25 ms of the predicted one, in |h| and in the envelope of the response correlated with
its own direct sound; offsets are taken relative to the direct sound, so a common transducer delay drops
out. A random baseline repeats the search at uniformly drawn times. The direct-sound times of all responses
are regressed on the calibrated distances to check the speed of sound and the time origin.

D2: with the labels taken from the geometry, the room, microphone, source and latency are fitted to the
measured arrivals by least squares from five starts near the truth (as M7), and the summed error of the
sorted sides is reported. V0 fits the exact predicted arrivals and checks the fitting code.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import common as cm

HALF = 12  # search half-width, samples (0.25 ms, 8.6 cm of path)


def window_max(signal: np.ndarray, centre: float, half: int = HALF) -> tuple[float, float]:
    lo, hi = max(0, int(round(centre)) - half), min(len(signal) - 1, int(round(centre)) + half)
    i = lo + int(np.argmax(signal[lo:hi + 1]))
    return float(cm.parabolic(signal, np.array([i]))[0]), float(signal[i])


def predicted_lengths(theta: np.ndarray, ks: np.ndarray) -> np.ndarray:
    size, mic, src, delta = theta[:3], theta[3:6], theta[6:9], theta[9]
    disp = np.where(ks % 2 == 1, -1.0, 1.0) * src + (ks + ks % 2) * size - mic
    return np.linalg.norm(disp, axis=1) + delta


def fit_with_labels(r_obs: np.ndarray, ks: np.ndarray, truth: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.concatenate([np.full(3, 0.3), np.zeros(6), [-np.inf]])
    upper = np.concatenate([np.full(3, 60.0), np.full(6, 60.0), [np.inf]])
    best = None
    for _ in range(5):
        start = truth.copy()
        start[:9] += rng.uniform(-0.10, 0.10, 9)
        start[9] += rng.uniform(-0.03, 0.03)
        start = np.clip(start, lower + 1e-6, upper - 1e-6)
        sol = least_squares(lambda th: predicted_lengths(th, ks) - r_obs, start, method="trf",
                            bounds=(lower, upper), max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x


def side_error(theta: np.ndarray, room: np.ndarray) -> float:
    return float(np.abs(np.sort(theta[:3])[::-1] - np.sort(room)[::-1]).sum())


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev-A"
    dev = cm.load_dev(cm.HERE / "data" / "dev_responses.npz")
    geo = cm.geometry()
    room = geo["room"]
    sel = np.flatnonzero(dev["split"] == split)
    rng = np.random.default_rng(20261103)
    rows, fits, direct_rows = [], [], []
    to_cm = 100.0 * cm.SPEED / cm.RATE
    for n, idx in enumerate(sel):
        h = dev["rirs"][idx]
        code, j, i = str(dev["room_code"][idx]), int(dev["src_id"][idx]), int(dev["mic_id"][idx])
        mic, src = geo["mics"][i], geo["sources"][j]
        refl = cm.reflective(code)
        imgs = cm.images(room, mic, src, max_order=2, max_reflections=2)
        d0 = imgs[0]["distance"]
        mag = np.abs(h)
        env_mf, _ = cm.matched_envelope(h)
        signals = {"abs": mag, "mf": env_mf}
        nds = {name: float(cm.parabolic(s, np.array([cm.direct_peak(s)]))[0]) for name, s in signals.items()}
        direct_rows.append({"distance_m": d0, "direct_s": nds["abs"] / cm.RATE, "src_id": j, "mic_id": i, "room_code": code})
        measured = {name: [] for name in signals}
        for img in imgs:
            rel = (img["distance"] - d0) / cm.SPEED * cm.RATE
            row = {"split": split, "response": int(idx), "room_code": code, "src_id": j, "mic_id": i,
                   "k": "%d,%d,%d" % img["k"], "order": img["order"], "walls": "+".join(img["walls"]),
                   "all_reflective": all(refl[w] for w in img["walls"]),
                   "any_reflective": any(refl[w] for w in img["walls"]), "pred_rel_samples": rel}
            for name, s in signals.items():
                t, v = window_max(s, nds[name] + rel)
                row[f"off_cm_{name}"] = (t - nds[name] - rel) * to_cm
                row[f"amp_{name}"] = v / s[int(round(nds[name]))]
                measured[name].append(t)
            # random baseline: the same search at a uniform time inside the span of the order-2 images
            span = (imgs[-1]["distance"] - d0) / cm.SPEED * cm.RATE
            t_rand = rng.uniform(2 * HALF, span)
            for name, s in signals.items():
                t, _ = window_max(s, nds[name] + t_rand)
                row[f"rand_off_cm_{name}"] = (t - nds[name] - t_rand) * to_cm
            rows.append(row)
        ks = np.array([img["k"] for img in imgs])
        truth = np.concatenate([room, mic, src, [0.0]])
        exact = np.array([img["distance"] for img in imgs])
        fit_row = {"split": split, "response": int(idx), "room_code": code, "src_id": j, "mic_id": i}
        theta = fit_with_labels(exact, ks, truth, seed=int(idx))
        fit_row["V0_exact"] = side_error(theta, room)
        for name in signals:
            r_obs = np.array(measured[name]) * cm.SPEED / cm.RATE
            theta = fit_with_labels(r_obs, ks, truth, seed=int(idx))
            fit_row[f"V_{name}_all"] = side_error(theta, room)
            keep = np.array([all(refl[w] for w in img["walls"]) for img in imgs])
            if keep.sum() >= 12:
                theta = fit_with_labels(r_obs[keep], ks[keep], truth, seed=int(idx))
                fit_row[f"V_{name}_reflective"] = side_error(theta, room)
            fit_row[f"n_reflective_images"] = int(keep.sum())
        fits.append(fit_row)
        if (n + 1) % 60 == 0:
            print(f"{n + 1}/{len(sel)}", flush=True)

    out = cm.HERE / "results"
    t = pd.DataFrame(rows)
    f = pd.DataFrame(fits)
    t.to_csv(out / f"d1_timing_{split}.csv", index=False)
    f.to_csv(out / f"d2_label_fits_{split}.csv", index=False)

    d = pd.DataFrame(direct_rows)
    a = np.vstack([d["distance_m"], np.ones(len(d))]).T
    (inv_c, t0), *_ = np.linalg.lstsq(a, d["direct_s"].to_numpy(), rcond=None)
    resid = d["direct_s"].to_numpy() - a @ np.array([inv_c, t0])

    def share(x, lim):
        return float(np.mean(np.abs(x) <= lim))

    summary = {"split": split, "responses": int(len(sel)),
               "direct_regression": {"speed_m_s": float(1 / inv_c), "t0_ms": float(1000 * t0),
                                     "residual_sd_ms": float(1000 * resid.std()), "n": int(len(d))}}
    classes = {
        "first_order_reflective": (t["order"] == 1) & t["all_reflective"],
        "first_order_absorbing": (t["order"] == 1) & ~t["all_reflective"],
        "second_order_all_reflective": (t["order"] == 2) & t["all_reflective"],
        "second_order_other": (t["order"] == 2) & ~t["all_reflective"],
    }
    for label, mask in classes.items():
        sub = t[mask]
        entry = {"n": int(len(sub))}
        for name in ("abs", "mf"):
            entry[name] = {"median_abs_off_cm": float(np.median(np.abs(sub[f"off_cm_{name}"]))),
                           "within_1cm": share(sub[f"off_cm_{name}"], 1.0), "within_2cm": share(sub[f"off_cm_{name}"], 2.0),
                           "random_within_2cm": share(sub[f"rand_off_cm_{name}"], 2.0),
                           "median_amp": float(np.median(sub[f"amp_{name}"]))}
        summary[label] = entry
    for col in [c for c in f.columns if c.startswith("V")]:
        v = f[col].dropna()
        summary[f"D2_{col}"] = {"n": int(len(v)), "median_cm": float(100 * np.median(v)),
                                "success_5cm": float(np.mean(v < 0.05)), "success_10cm": float(np.mean(v < 0.10))}
    (out / f"d1_d2_summary_{split}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
