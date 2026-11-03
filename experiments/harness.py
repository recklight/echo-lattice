"""Evaluation harness for the lattice inversion, experiments E1-E9 of the paper.

Run with the project's Python 3.10 environment, for example::

    <project>/.venv/Scripts/python.exe harness.py --experiment E1 --workers 12 --out results

Every experiment draws its scenes from a fixed master seed, simulates one
measurement per scene and condition, identifies the impulse response once,
and runs every method on that same response (paired design). Ablations and
the random multi-start baseline call the package's internal functions; the
package itself is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MASTER_SEED = 20261024
SUCCESS_M = 0.05
PROJECT = Path(__file__).resolve().parents[1]  # repository root; code_identity() hashes its src/

PLANAR = {
    "acoustics.max_order": 4,
    "acoustics.reflection": 0.85,
    "acoustics.delay": "fractional",
    "inversion.method": "lattice",
}
SPATIAL = {
    **PLANAR,
    "acoustics.sample_rate": 48_000,
    "acoustics.max_order": 5,
    "acoustics.rir_taps": 4800,
    "identification.n_taps": 4800,
    "identification.method": "least_squares",
    "synthetic.duration_s": 1.0,
    "lattice.dimensions": 3,
    "lattice.max_order": 5,
    "lattice.max_echoes": 120,
}


# ----------------------------------------------------------------------- scenes
@dataclass
class Scene:
    size: list[float]
    mic: list[float]
    src: list[float]
    latency_ms: float
    noise_seed: int
    extra: dict = field(default_factory=dict)


def draw_scene(
    rng: np.random.Generator,
    n_dims: int,
    *,
    low=(3.0, 3.0, 2.4),
    high=(10.0, 8.0, 4.0),
    margin=(0.3, 0.3),
    distance=(0.6, 2.5),
    mic_margin=None,
) -> tuple[list[float], list[float], list[float]]:
    """Room sides, microphone and source (same rule as the project's benchmark)."""
    low, high = np.array(low)[:n_dims], np.array(high)[:n_dims]
    while True:
        size = np.round(rng.uniform(low, high), 3)
        if mic_margin is None:
            mic = np.round(rng.uniform(margin[0], size - margin[0]), 3)
        else:  # microphone close to a wall on every axis
            near = rng.uniform(mic_margin[0], mic_margin[1], n_dims)
            side = rng.integers(0, 2, n_dims)
            mic = np.round(np.where(side == 0, near, size - near), 3)
        src = np.round(rng.uniform(margin[1], size - margin[1]), 3)
        if distance[0] <= np.linalg.norm(src - mic) <= distance[1]:
            return size.tolist(), mic.tolist(), src.tolist()


def scenes_for(experiment: str, n: int, n_dims: int, **kw) -> list[Scene]:
    """Deterministic scenes: one SeedSequence child per scene (scene draw and noise)."""
    root = np.random.SeedSequence([MASTER_SEED, sum(map(ord, experiment))])
    out = []
    for child in root.spawn(n):
        rng = np.random.default_rng(child)
        size, mic, src = draw_scene(rng, n_dims, **kw)
        latency = float(np.round(rng.uniform(0.0, 20.0), 3))
        out.append(Scene(size, mic, src, latency, int(rng.integers(0, 2**31 - 1))))
    return out


# ------------------------------------------------------------------ measurement
def measure(config, scene: Scene, *, truth_rir=None, skew_ppm: float = 0.0, spikes: int = 0):
    """Simulate one measurement and return (identified response, latency taps, truth)."""
    import echo_lattice as pkg
    from echo_lattice.recording import Recording, simulate_recording

    rng = np.random.default_rng(scene.noise_seed)
    truth = pkg.room_estimation.simulate_truth(config) if truth_rir is None else truth_rir
    rir = truth.rir.copy()
    rate = config.acoustics.sample_rate
    if spikes:
        taps = truth.arrival_taps()
        lo, hi = int(taps[0]) + 2, int(min(rir.size - 1, taps[min(len(taps) - 1, 39)]))
        where = rng.integers(lo, max(lo + 1, hi), spikes)
        rir[where] += rng.choice([-1.0, 1.0], spikes) * rng.uniform(0.03, 0.10, spikes)
    shift = pkg.latency_taps(config)
    delayed = pkg.delay_response(rir, shift)
    n_samples = round(config.synthetic.duration_s * rate)
    signal = pkg.make_excitation(config.synthetic.excitation, n_samples, rate, rng=rng)
    level = config.synthetic.snr_db
    clean = simulate_recording(signal, delayed, snr_db=None, sample_rate=rate, rng=rng)
    if skew_ppm:
        clean = fractional_resample(clean, 1.0 + 1e-6 * skew_ppm)
    microphone = pkg.add_noise(clean, level, kind=config.synthetic.noise, sample_rate=rate, rng=rng)
    recording = Recording(excitation=signal, microphone=microphone, sample_rate=rate)
    weights = pkg.room_estimation.identify_recording(recording, config)
    return weights, shift, truth


def fractional_resample(y: np.ndarray, ratio: float, half: int = 16) -> np.ndarray:
    """y sampled by a clock running ``ratio`` times faster (Hann-windowed sinc, same length)."""
    n = np.arange(y.size, dtype=np.float64)
    t = n / ratio
    base = np.floor(t).astype(np.int64)
    out = np.zeros_like(y)
    for m in range(-half + 1, half + 1):
        idx = base + m
        ok = (idx >= 0) & (idx < y.size)
        x = t - idx
        w = np.sinc(x) * 0.5 * (1 + np.cos(np.pi * np.clip(x / half, -1, 1)))
        out[ok] += y[idx[ok]] * w[ok]
    return out


# ---------------------------------------------------------------------- metrics
def canonical_direction(size, mic, src) -> np.ndarray:
    order = np.argsort(-np.asarray(size), kind="stable")
    offset = np.abs(np.subtract(src, mic))[order]
    return offset / np.linalg.norm(offset)


def estimate_direction(bearing_deg) -> np.ndarray:
    if len(bearing_deg) == 1:
        a = np.deg2rad(bearing_deg[0])
        return np.array([np.cos(a), np.sin(a)])
    az, el = np.deg2rad(bearing_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def lattice_row(estimate, scene: Scene, speed: float, true_latency_ms: float) -> dict:
    import echo_lattice as pkg

    err = pkg.room_size_error(estimate.size_m, scene.size)
    truth_dir = canonical_direction(scene.size, scene.mic, scene.src)
    found = estimate_direction(estimate.bearing_deg)
    angle = float(np.degrees(np.arccos(np.clip(float(truth_dir @ found), -1.0, 1.0))))
    sorted_true = np.sort(scene.size)[::-1]
    est = np.asarray(estimate.size_m)
    sd = np.asarray(estimate.size_sd_m)
    z = (est - sorted_true) / np.where(sd > 0, sd, np.nan)
    return {
        "est_sides": " x ".join(f"{s:.4f}" for s in est),
        "room_error_m": float(err),
        "success": bool(err < SUCCESS_M),
        "latency_error_ms": abs(estimate.latency_ms(speed) - true_latency_ms),
        "bearing_error_deg": angle,
        "reliable": bool(estimate.reliable),
        "explained": float(estimate.explained_fraction),
        "missed": float(estimate.missed_fraction),
        "n_echoes": int(estimate.n_echoes),
        "side_sd_m": " ".join(f"{s:.5f}" for s in sd),
        "side_z": " ".join(f"{v:.3f}" for v in z),
    }


# ----------------------------------------------------------------------- methods
def run_lattice(weights, config, overrides=None, patch=None):
    """Lattice inversion of one response; ``patch`` swaps internals for an ablation."""
    import echo_lattice as pkg
    from echo_lattice import lattice_inversion as li

    saved = {}
    if patch:
        for name, value in patch.items():
            saved[name] = getattr(li, name)
            setattr(li, name, value)
    try:
        cfg = config.override(overrides or {})
        return pkg.invert_response(weights, cfg).estimate
    finally:
        for name, value in saved.items():
            setattr(li, name, value)


def run_first_order(weights, config, bearing_deg, advance_taps=0):
    import echo_lattice as pkg

    response = np.asarray(weights, dtype=np.float64)
    if advance_taps:
        response = np.concatenate([response[advance_taps:], np.zeros(advance_taps)])
    cfg = config.override({"inversion.method": "first_order", "inversion.source_bearing_deg": bearing_deg})
    result = pkg.invert_response(response, cfg)
    return result.estimate, result.echo_taps


def first_order_hits(echo_taps, scene: Scene, config, advance_taps: int, shift: int) -> int:
    """How many of the 9 true first-order arrivals (direct included) are among the picks."""
    import echo_lattice as pkg

    a = config.acoustics
    first = pkg.image_source_rir(scene.size, scene.src, scene.mic, sample_rate=a.sample_rate,
                                 n_taps=a.rir_taps, speed_of_sound_m_s=a.speed, max_order=1,
                                 delay=a.delay).arrival_taps() + (shift - advance_taps)
    picks = np.asarray(echo_taps, dtype=np.float64)
    return int(sum(np.min(np.abs(picks - t)) <= 2 for t in first)) if picks.size else 0


def picked_lengths(weights, config):
    """The echoes the lattice inversion sees, as path lengths."""
    import echo_lattice as pkg

    s, a = config.lattice, config.acoustics
    magnitude = pkg.prune_small_taps(weights, config.identification.prune_threshold)
    taps = pkg.pick_arrivals(magnitude, s.max_echoes, relative_floor=s.relative_floor,
                             min_separation_taps=s.min_separation_taps)
    return pkg.taps_to_distances(taps, a.sample_rate, a.speed)


def _finish(theta, r, n_dims, lattice, model, loglik, cov, min_explained, max_missed):
    """LatticeEstimate from fitted parameters, as invert_lattice builds it."""
    from echo_lattice import lattice_inversion as li

    canonical, order = li._canonical(theta, n_dims)
    size, mic, src = canonical[:n_dims], canonical[n_dims:2 * n_dims], canonical[2 * n_dims:3 * n_dims]
    side_sd = np.sqrt(np.clip(np.diag(cov)[:n_dims], 0.0, None))[order]
    offset = src - mic
    if n_dims == 2:
        bearing = (float(np.degrees(np.arctan2(offset[1], offset[0]))),)
    else:
        bearing = (float(np.degrees(np.arctan2(offset[1], offset[0]))),
                   float(np.degrees(np.arctan2(offset[2], np.hypot(offset[0], offset[1])))))
    explained, missed = li._fit_quality(r, theta, lattice, model)
    return li.LatticeEstimate(
        size_m=tuple(map(float, size)), microphone_m=tuple(map(float, mic)),
        source_m=tuple(map(float, src)), latency_m=float(theta[-1]),
        size_sd_m=tuple(map(float, side_sd)), latency_sd_m=float(np.sqrt(max(cov[-1, -1], 0.0))),
        source_distance_m=float(np.linalg.norm(offset)), bearing_deg=bearing,
        log_likelihood=float(loglik), n_echoes=int(r.size), explained_fraction=explained,
        missed_fraction=missed, reliable=bool(explained >= min_explained and missed <= max_missed))


def run_random_multistart(weights, config, budget_s: float, seed: int):
    """M6: the same score and EM fit, started from random points, for the same wall time."""
    import echo_lattice as pkg
    from echo_lattice import lattice_inversion as li

    s = config.lattice
    n_dims = s.dimensions
    model = pkg.echo_model(config)
    r = np.sort(picked_lengths(weights, config))
    lattice = li._Lattice(n_dims, s.max_order)
    rng = np.random.default_rng(seed)
    hi = np.array([12.0, 10.0, 5.0])[:n_dims]
    lo = np.array([2.0, 2.0, 2.0])[:n_dims]
    best = (-np.inf, None, None)
    started = time.perf_counter()
    tries = 0
    while time.perf_counter() - started < budget_s or tries < 1:
        size = rng.uniform(lo, hi)
        mic = rng.uniform(0.05 * size, 0.95 * size)
        src = rng.uniform(0.05 * size, 0.95 * size)
        delta = r[0] - float(np.linalg.norm(src - mic))
        theta, loglik, cov = li.fit_lattice(r, np.concatenate([size, mic, src, [delta]]),
                                            max_order=s.max_order, model=model)
        tries += 1
        if loglik > best[0]:
            best = (loglik, theta, cov)
    min_e = 0.9 if s.min_explained is None else s.min_explained
    max_m = (0.2 if n_dims == 2 else 0.1) if s.max_missed is None else s.max_missed
    est = _finish(best[1], r, n_dims, lattice, model, best[0], best[2], min_e, max_m)
    return est, tries


def run_oracle_labels(weights, config, scene: Scene, true_latency_m: float, seed: int, starts: int = 5):
    """M7: echoes assigned to their true images; same least-squares step as the polish."""
    import echo_lattice as pkg
    from echo_lattice import lattice_inversion as li
    from scipy.optimize import least_squares

    s = config.lattice
    n_dims = s.dimensions
    model = pkg.echo_model(config)
    r = np.sort(picked_lengths(weights, config))
    lattice = li._Lattice(n_dims, s.max_order)
    truth = np.concatenate([scene.size, scene.mic, scene.src, [true_latency_m]])
    rho, _ = lattice.lengths(truth)
    tol = max(2 * model.resolution_m, 0.02)
    gap = np.abs(r[:, None] - rho[None, :])
    nearest = np.argmin(gap, axis=1)
    ok = gap[np.arange(r.size), nearest] < tol
    echoes, images = np.flatnonzero(ok), nearest[ok]
    keep = np.unique(images, return_index=True)[1]
    echoes, images = echoes[keep], images[keep]
    lower = np.concatenate([np.full(n_dims, 0.3), np.zeros(2 * n_dims), [-np.inf]])
    upper = np.concatenate([np.full(n_dims, 60.0), np.full(2 * n_dims, 60.0), [np.inf]])

    def res(th):
        return r[echoes] - lattice.lengths(th)[0][images]

    def jac(th):
        return -lattice.jacobian(th, images)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        start = truth.copy()
        start[: 3 * n_dims] += rng.uniform(-0.10, 0.10, 3 * n_dims)
        start[-1] += rng.uniform(-0.03, 0.03)
        start = li._clip_inside(np.clip(start, lower + 1e-6, upper - 1e-6), n_dims)
        sol = least_squares(res, start, jac=jac, method="trf", bounds=(lower, upper), max_nfev=200)
        if best is None or sol.cost < best.cost:
            best = sol
    theta = li._clip_inside(best.x, n_dims)
    cov = li._laplace_covariance(r, theta, lattice, li.FIT_WIDTHS_M[-1], model)
    loglik = 0.0
    min_e = 0.9 if s.min_explained is None else s.min_explained
    max_m = (0.2 if n_dims == 2 else 0.1) if s.max_missed is None else s.max_missed
    est = _finish(theta, r, n_dims, lattice, model, loglik, cov, min_e, max_m)
    return est, int(echoes.size), int(r.size)


# --------------------------------------------------------------------- ablations
def _single_chain_fit(observed_m, theta, *, max_order=4, model=None, widths_m=None, iterations=4):
    from echo_lattice import lattice_inversion as li

    model = model or li.EchoModel()
    r = np.sort(np.asarray(observed_m, dtype=np.float64))
    start = np.asarray(theta, dtype=np.float64).copy()
    n_dims = (start.size - 1) // 3
    lattice = li._Lattice(n_dims, max_order)
    floor = 0.5 * model.resolution_m
    widths = tuple(max(w, floor) for w in (widths_m or li.FIT_WIDTHS_M))
    lower = np.concatenate([np.full(n_dims, 0.3), np.zeros(2 * n_dims), [-np.inf]])
    upper = np.concatenate([np.full(n_dims, 60.0), np.full(2 * n_dims, 60.0), [np.inf]])
    theta = li._clip_inside(np.clip(start, lower + 1e-6, upper - 1e-6), n_dims)
    for width in widths:
        theta = li._em_rounds(r, theta, lattice, model, width, iterations, (lower, upper))
        theta = li._polish(r, theta, lattice, model, widths[-1], (lower, upper))
    rho, distance = lattice.lengths(theta)
    q0 = float(np.linalg.norm(theta[2 * n_dims:3 * n_dims] - theta[n_dims:2 * n_dims]))
    prob = li.detection_probability(lattice.n_reflections, distance, q0, model)
    ll = li.point_process_loglik(r, rho, prob, widths[-1], model)
    return theta, ll, li._laplace_covariance(r, theta, lattice, widths[-1], model)


def _no_expectation_batch(r, rho, prob, width_m, model):
    """Score without the expected-count term: only the explained echoes are rewarded."""
    from echo_lattice import lattice_inversion as li

    G = rho.shape[0]
    low, high, span = li._window(r, model)
    inside = (rho > low) & (rho < high)
    big = 1e7
    masked = np.where(inside, rho, big)
    order = np.argsort(masked, axis=1)
    ts = np.take_along_axis(masked, order, axis=1)
    ps = np.take_along_axis(np.where(inside, prob, 0.0), order, axis=1)
    offset = np.arange(G)[:, None] * 4 * big
    flat = (ts + offset).ravel()
    pflat = ps.ravel()
    query = (r[None, :] + offset).ravel()
    pos = np.searchsorted(flat, query)
    lo = np.clip(pos - 1, 0, flat.size - 1)
    hi = np.clip(pos, 0, flat.size - 1)
    near = np.exp(-0.5 * ((query - flat[lo]) / width_m) ** 2) * pflat[lo]
    near += np.exp(-0.5 * ((query - flat[hi]) / width_m) ** 2) * pflat[hi]
    near /= width_m * np.sqrt(2 * np.pi)
    return np.log(model.clutter / span + near.reshape(G, -1)).sum(axis=1) - model.clutter


def _no_expectation_loglik(observed_m, predicted_m, probability, width_m, model):
    from echo_lattice import lattice_inversion as li

    r = np.sort(np.asarray(observed_m, dtype=np.float64))
    rho = np.asarray(predicted_m, dtype=np.float64)
    prob = np.asarray(probability, dtype=np.float64)
    low, high, span = li._window(r, model)
    inside = (rho > low) & (rho < high)
    rho, prob = rho[inside], prob[inside]
    density = np.exp(-0.5 * ((r[:, None] - rho[None, :]) / width_m) ** 2)
    density = density @ prob / (width_m * np.sqrt(2 * np.pi))
    return float(np.log(model.clutter / span + density).sum() - model.clutter)


def _identity_polish(r, theta, lattice, model, width, bounds):
    return theta


def _no_rescan(r, theta, loglik, scorer, **kw):
    return theta, loglik


ABLATIONS = {
    "A1_no_missed_term": {"patch": {"_batch_loglik": _no_expectation_batch,
                                    "point_process_loglik": _no_expectation_loglik}},
    "A2_single_chain": {"patch": {"fit_lattice": _single_chain_fit}},
    "A3_no_polish": {"patch": {"_polish": _identity_polish}},
    "A4_no_rescan": {"patch": {"_rescan_axes": _no_rescan}},
    "A5_image_groups_only": {"overrides": {"lattice.searches": ["image_groups"]}},
    "A6_axis_scan_only": {"overrides": {"lattice.searches": ["axis_scan"]}},
}


# ------------------------------------------------------------------ experiments
def base_config(n_dims: int):
    import echo_lattice as pkg

    return pkg.Config().override(PLANAR if n_dims == 2 else SPATIAL)


def scene_config(base, scene: Scene, extra: dict | None = None):
    settings = {"room.size_m": scene.size, "room.microphone_m": scene.mic,
                "room.source_m": scene.src, "synthetic.latency_ms": scene.latency_ms}
    settings.update(extra or {})
    return base.override(settings)


def true_bearing_deg(scene: Scene) -> float:
    return float(np.degrees(np.arctan2(scene.src[1] - scene.mic[1], scene.src[0] - scene.mic[0])))


def job(spec: dict) -> list[dict]:
    """One scene and condition: measure once, run every listed method."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import echo_lattice as pkg
    from echo_lattice.image_source import image_source_rir, speed_of_sound

    scene = Scene(**spec["scene"])
    n_dims = len(scene.size)
    base = base_config(n_dims)
    config = scene_config(base, scene, spec.get("config"))
    truth = None
    mismatch = spec.get("mismatch", {})
    a = config.acoustics
    if mismatch:
        speed = speed_of_sound(mismatch["temperature_c"]) if "temperature_c" in mismatch else a.speed
        truth = image_source_rir(
            scene.size, scene.src, scene.mic, sample_rate=a.sample_rate, n_taps=a.rir_taps,
            speed_of_sound_m_s=speed, max_order=mismatch.get("sim_order", a.max_order),
            order_mode=a.order_mode, reflection=mismatch.get("reflection", a.reflection), delay=a.delay)
    rows = []
    base_row = {"experiment": spec["experiment"], "condition": spec["condition"], "scene": spec["index"],
                "true_sides": " x ".join(f"{x:.3f}" for x in scene.size),
                "true_latency_ms": None}
    try:
        weights, shift, truth = measure(config, scene, truth_rir=truth,
                                        skew_ppm=mismatch.get("skew_ppm", 0.0),
                                        spikes=mismatch.get("spikes", 0))
    except Exception as exc:  # recorded, never silently dropped
        return [{**base_row, "method": "measurement", "error": repr(exc)}]
    rate, speed = a.sample_rate, a.speed
    true_latency_ms = 1000.0 * shift / rate
    base_row["true_latency_ms"] = true_latency_ms
    m1_time = None
    for method in spec["methods"]:
        row = {**base_row, "method": method}
        started = time.perf_counter()
        try:
            if method == "M1":
                est = run_lattice(weights, config)
            elif method == "M2":
                est = run_lattice(weights, config, {"lattice.latency_ms": true_latency_ms})
            elif method in ABLATIONS:
                spec_a = ABLATIONS[method]
                est = run_lattice(weights, config, spec_a.get("overrides"), spec_a.get("patch"))
            elif method == "M6":
                est, tries = run_random_multistart(weights, config, m1_time or 5.0, scene.noise_seed)
                row["tries"] = tries
            elif method == "M7":
                est, used, total = run_oracle_labels(weights, config, scene, 1e-3 * true_latency_ms * speed,
                                                     scene.noise_seed)
                row["labelled_echoes"], row["picked_echoes"] = used, total
            elif method in ("M3", "M4", "M5"):
                bearing = 45.0 if method != "M5" else true_bearing_deg(scene)
                advance = 0 if method == "M3" else shift
                est, echo_taps = run_first_order(weights, config, bearing, advance)
                err = pkg.room_size_error(est.size_m, scene.size)
                row.update(est_sides=" x ".join(f"{x:.4f}" for x in est.size_m),
                           room_error_m=float(err), success=bool(err < SUCCESS_M),
                           first_order_hits=first_order_hits(echo_taps, scene, config, advance, shift))
                row["time_s"] = time.perf_counter() - started
                rows.append(row)
                continue
            else:
                raise ValueError(method)
            row.update(lattice_row(est, scene, speed, true_latency_ms))
        except Exception as exc:  # InversionError and the like: a failure, kept in the table
            row.update(room_error_m=float("inf"), success=False, reliable=False,
                       error=f"{type(exc).__name__}: {exc}")
        row["time_s"] = time.perf_counter() - started
        if method == "M1":
            m1_time = row["time_s"]
        rows.append(row)
    return rows


def plan(experiment: str, n_override: int | None = None) -> list[dict]:
    """The jobs of one experiment; freeze.json keeps the hash of every list."""
    jobs = []

    def add(name, scenes, methods, condition="main", config=None, mismatch=None, per_scene=None):
        for i, sc in enumerate(scenes):
            m = methods if per_scene is None else per_scene(i)
            jobs.append({"experiment": name, "condition": condition, "index": i,
                         "scene": sc.__dict__, "methods": m, "config": config or {},
                         "mismatch": mismatch or {}})

    n = n_override
    if experiment == "E1":
        sc = scenes_for("E1", n or 500, 2)
        add("E1", sc, None, per_scene=lambda i: ["M1", "M2", "M3", "M4", "M5", "M7"] + (["M6"] if i < 200 else []))
    elif experiment == "E1t":
        # Timing only: run with --workers 1 so the wall times are single-process.
        add("E1t", scenes_for("E1", n or 20, 2), ["M1", "M2", "M3"], "timing")
    elif experiment == "E1b":
        # The first-order closed form in its own regime: only first-order reflections simulated,
        # synchronized recording; the first 100 scenes of E1.
        sc = [Scene(**{**s.__dict__, "latency_ms": 0.0}) for s in scenes_for("E1", n or 100, 2)]
        add("E1b", sc, ["M1", "M4", "M5"], "sim_order_1", {"acoustics.max_order": 1})
    elif experiment == "E2":
        sc = scenes_for("E2", n or 200, 3)
        add("E2", sc, None, per_scene=lambda i: ["M1", "M2", "M7"] + (["M6"] if i < 100 else []))
    elif experiment == "E3":
        sc = scenes_for("E3", n or 100, 2)
        for snr in (0, 10, 20, 30, 40):
            add("E3", sc, ["M1", "M5"], f"snr_{snr}dB", {"synthetic.snr_db": float(snr)})
    elif experiment == "E4":
        sc = scenes_for("E4", n or 100, 2)
        # The response keeps 1600 taps after the latency in every condition.
        for lat in (0, 10, 20, 40):
            taps = round(lat * 16)
            fixed = [Scene(**{**s.__dict__, "latency_ms": float(lat)}) for s in sc]
            add("E4", fixed, ["M1", "M2"], f"latency_{lat}ms",
                {"identification.n_taps": 1600 + taps, "acoustics.rir_taps": 1600 + taps})
        fixed = [Scene(**{**s.__dict__, "latency_ms": 150.0}) for s in sc]
        add("E4", fixed, ["M1", "M2"], "latency_150ms",
            {"identification.n_taps": 4000, "acoustics.rir_taps": 4000,
             "identification.method": "least_squares"})
    elif experiment == "E5":
        sc = scenes_for("E5", n or 100, 2)
        rng = np.random.default_rng([MASTER_SEED, 5])
        betas = [rng.uniform(0.6, 0.95, 4).round(3).tolist() for _ in sc]
        for i, s in enumerate(sc):
            jobs.append({"experiment": "E5", "condition": "R1_wall_reflection", "index": i,
                         "scene": s.__dict__, "methods": ["M1"], "config": {},
                         "mismatch": {"reflection": betas[i]}})
        add("E5", sc, ["M1"], "R2_sim_order_8", mismatch={"sim_order": 8})
        add("E5", sc, ["M1"], "R3_speed_15C", mismatch={"temperature_c": 15.0})
        add("E5", sc, ["M1"], "R3_speed_30C", mismatch={"temperature_c": 30.0})
        add("E5", sc, ["M1"], "R4_spurious_pulses", mismatch={"spikes": 5})
        add("E5", sc, ["M1"], "R5_skew_20ppm", mismatch={"skew_ppm": 20.0})
        add("E5", sc, ["M1"], "R5_skew_100ppm", mismatch={"skew_ppm": 100.0})
    elif experiment == "E6":
        sc = scenes_for("E1", n or 200, 2)  # the first 200 scenes of E1
        add("E6", sc, list(ABLATIONS))
    elif experiment == "E9":
        add("E9", scenes_for("E9_D1", n or 100, 2, low=(2.0, 2.0), high=(4.0, 4.0)), ["M1"], "D1_small")
        add("E9", scenes_for("E9_D2", n or 100, 2, low=(10.0, 8.0), high=(20.0, 15.0)), ["M1"], "D2_large",
            {"identification.n_taps": 3200, "acoustics.rir_taps": 3200, "synthetic.duration_s": 4.0})
        add("E9", scenes_for("E9_D3", n or 100, 2, distance=(2.5, 5.0)), ["M1"], "D3_far_source")
        add("E9", scenes_for("E9_D4", n or 100, 2, mic_margin=(0.05, 0.3)), ["M1"], "D4_mic_near_wall")
    else:
        raise ValueError(experiment)
    return jobs


# --------------------------------------------------------------------- provenance
def code_identity() -> dict:
    files = sorted((PROJECT / "src").rglob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(PROJECT).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    try:
        head = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PROJECT), "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except Exception:
        head, dirty = None, None
    return {"git_head": head, "src_dirty": dirty, "src_tree_sha256": h.hexdigest(), "n_files": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--n", type=int, default=None, help="Override the scene count (smoke tests).")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--tag", default="", help="Suffix of the output files.")
    args = parser.parse_args()
    import pandas as pd

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # one BLAS thread per worker process
    jobs = plan(args.experiment, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.experiment}{args.tag}"
    meta = {"experiment": args.experiment, "jobs": len(jobs), "master_seed": MASTER_SEED,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code": code_identity(), "python": sys.version, "platform": platform.platform(),
            "workers": args.workers, "n_override": args.n}
    rows, failures = [], 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, spec) for spec in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rows += fut.result()
            except Exception:
                failures += 1
                rows.append({"experiment": args.experiment, "error": traceback.format_exc()[-2000:]})
            if k % 20 == 0 or k == len(futures):
                print(f"{stem}: {k}/{len(futures)} jobs, {time.perf_counter() - started:.0f} s", flush=True)
    table = pd.DataFrame(rows).sort_values(["condition", "scene", "method"], na_position="last")
    table.to_csv(args.out / f"{stem}.csv", index=False)
    meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                wall_s=time.perf_counter() - started, job_failures=failures)
    (args.out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out / (stem + '.csv')} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
