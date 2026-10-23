"""Lattice and first-order inversion on random rooms with unknown latency and bearing.

    python examples/benchmark_lattice_inversion.py --out examples/output/benchmark_lattice
    python examples/benchmark_lattice_inversion.py --dims 3 --scenes 20 --out examples/output/3d

Every scene draws a rectangular room, a microphone and a source at random, and
delays the microphone channel by a random 0-20 ms, as a playback chain with its
own buffers would. The whole chain runs on it: a white-noise measurement at
30 dB, identification, echo picking and inversion.

Planar scenes are inverted twice, by the lattice inversion and by the
first-order closed form, which has to assume the source bearing (45 degrees)
and treats the recording as synchronized. 3-D scenes run at 48 kHz with
least-squares identification; the first-order form does not apply to them.

Writes scenes.csv (one row per scene and method) and summary.csv.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from echo_lattice import (
    Config,
    InversionError,
    configure_logging,
    latency_taps,
    simulate_scene,
)

LOGGER = logging.getLogger("benchmark_lattice_inversion")

#: A fit within this distance of the true sides (summed) counts as right.
SUCCESS_M = 0.05

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


def draw_scene(rng: np.random.Generator, n_dims: int) -> dict[str, list[float]]:
    """Room sides, microphone and source, 0.6-2.5 m apart and 0.3 m clear of every wall."""
    low = np.array([3.0, 3.0, 2.4])[:n_dims]
    high = np.array([10.0, 8.0, 4.0])[:n_dims]
    while True:
        size = np.round(rng.uniform(low, high), 3)
        mic = np.round(rng.uniform(0.3, size - 0.3), 3)
        src = np.round(rng.uniform(0.3, size - 0.3), 3)
        if 0.6 <= np.linalg.norm(src - mic) <= 2.5:
            return {"size": size.tolist(), "mic": mic.tolist(), "src": src.tolist()}


def canonical_direction(size, mic, src) -> np.ndarray:
    """Unit source direction on the sorted axes, every component non-negative."""
    order = np.argsort(-np.asarray(size), kind="stable")
    offset = np.abs(np.subtract(src, mic))[order]
    return offset / np.linalg.norm(offset)


def estimate_direction(bearing_deg) -> np.ndarray:
    if len(bearing_deg) == 1:
        angle = np.deg2rad(bearing_deg[0])
        return np.array([np.cos(angle), np.sin(angle)])
    azimuth, elevation = np.deg2rad(bearing_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in table.groupby("method", sort=False):
        right = part["room_error_m"] < SUCCESS_M
        row = {
            "method": method,
            "scenes": len(part),
            "right": int(right.sum()),
            "median_room_error_m": float(part["room_error_m"].median()),
            "median_error_when_right_m": float(part.loc[right, "room_error_m"].median()),
        }
        if method == "lattice":
            reliable = part["reliable"].astype(bool)
            row.update(
                median_latency_error_ms=float(part.loc[right, "latency_error_ms"].median()),
                median_bearing_error_deg=float(part.loc[right, "bearing_error_deg"].median()),
                reliable=int(reliable.sum()),
                reliable_and_right=int((reliable & right).sum()),
                wrong_flagged=int((~reliable & ~right).sum()),
                wrong=int((~right).sum()),
                mean_time_s=float(part["time_s"].mean()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/benchmark_lattice_inversion"),
        help="Output folder.",
    )
    parser.add_argument("--dims", type=int, choices=(2, 3), default=2, help="Room dimensions.")
    parser.add_argument("--scenes", type=int, default=40, help="Random scenes.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument("--quick", action="store_true", help="Three short scenes.")
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    LOGGER.setLevel(logging.INFO)

    base = Config().override(PLANAR if args.dims == 2 else SPATIAL)
    n_scenes = args.scenes
    if args.quick:
        n_scenes = 3
        base = base.override(
            {"synthetic.duration_s": 1.0, "identification.method": "least_squares"}
        )
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for number in range(n_scenes):
        scene = draw_scene(rng, args.dims)
        settings = {
            "room.size_m": scene["size"],
            "room.microphone_m": scene["mic"],
            "room.source_m": scene["src"],
            "synthetic.latency_ms": float(np.round(rng.uniform(0.0, 20.0), 3)),
        }
        config = base.override(settings)
        rate = config.acoustics.sample_rate
        true_latency_ms = 1000.0 * latency_taps(config) / rate
        true_distance = float(np.linalg.norm(np.subtract(scene["src"], scene["mic"])))
        methods = ("lattice", "first_order") if args.dims == 2 else ("lattice",)
        for method in methods:
            started = time.perf_counter()
            try:
                result = simulate_scene(config.override({"inversion.method": method}), rng=number)
            except InversionError as exc:
                LOGGER.info("scene %d, %s: %s", number, method, exc)
                rows.append(
                    {"scene": number, "method": method, "room_error_m": np.inf, "reliable": False}
                )
                continue
            elapsed = time.perf_counter() - started
            estimate = result.inversion.estimate
            row = {
                "scene": number,
                "method": method,
                "sides_m": " x ".join(f"{side:.3f}" for side in scene["size"]),
                "estimate_m": " x ".join(f"{side:.3f}" for side in estimate.size_m),
                "room_error_m": result.room_error_m,
                "distance_error_m": abs(estimate.source_distance_m - true_distance),
                "true_latency_ms": true_latency_ms,
                "time_s": elapsed,
            }
            if method == "lattice":
                truth = canonical_direction(scene["size"], scene["mic"], scene["src"])
                found = estimate_direction(estimate.bearing_deg)
                angle = np.degrees(np.arccos(np.clip(float(truth @ found), -1.0, 1.0)))
                row.update(
                    latency_error_ms=abs(
                        estimate.latency_ms(config.acoustics.speed) - true_latency_ms
                    ),
                    bearing_error_deg=float(angle),
                    reliable=estimate.reliable,
                    explained_fraction=estimate.explained_fraction,
                    missed_fraction=estimate.missed_fraction,
                    n_echoes=estimate.n_echoes,
                )
            rows.append(row)
        last = rows[-len(methods)]
        LOGGER.info(
            "scene %d: %s m, latency %.2f ms -> lattice error %.3f m (%s)",
            number,
            " x ".join(f"{side:.3f}" for side in scene["size"]),
            true_latency_ms,
            last["room_error_m"],
            "reliable" if last.get("reliable") else "flagged",
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "scenes.csv", index=False, lineterminator="\n")
    summary = summarize(table)
    summary.to_csv(args.out / "summary.csv", index=False, lineterminator="\n")
    for row in summary.to_dict("records"):
        LOGGER.info(
            "%s: %d of %d rooms within %.0f cm, median error %.4f m",
            row["method"],
            row["right"],
            row["scenes"],
            100 * SUCCESS_M,
            row["median_room_error_m"],
        )
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
