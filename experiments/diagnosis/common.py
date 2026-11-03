"""Shared pieces of the diagnosis: geometry, predicted image arrivals and the arrival detectors compared.

Run the diagnosis scripts with the environment where echo-lattice is installed; they import the package
as E10 did and read the calibrated geometry through ../e10_run.py, which parses the calibration script as
text and never executes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e10_run  # noqa: E402  (geometry parser of E10)

RATE = 48_000
SPEED = 345.844  # m/s, the frozen E10 value
WINDOW = 4800  # samples analysed after time zero, as in E10 (100 ms)
SURFACES = ("floor", "ceiling", "west", "south", "east", "north")  # order of the room-code digits
# (axis, sign of the image index) -> surface; west x = 0, east x = L_x, south y = 0, north y = L_y
AXIS_WALL = {(0, -1): "west", (0, 1): "east", (1, -1): "south", (1, 1): "north", (2, -1): "floor", (2, 1): "ceiling"}


def geometry() -> dict:
    return e10_run.geometry(e10_run.DATA / "main_geometry_from_echo_calibration.py")


def load_dev(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def reflective(code: str) -> dict:
    return {name: digit == "1" for name, digit in zip(SURFACES, code)}


def images(room, mic, src, max_order: int = 2, max_reflections: int | None = 2) -> list[dict]:
    """Image sources with their distance and the surfaces each path reflects off (with multiplicity)."""
    room, mic, src = (np.asarray(v, dtype=np.float64) for v in (room, mic, src))
    ks = np.arange(-max_order, max_order + 1)
    out = []
    for kx in ks:
        for ky in ks:
            for kz in ks:
                k = np.array([kx, ky, kz])
                n = int(np.abs(k).sum())
                if max_reflections is not None and n > max_reflections:
                    continue
                disp = np.where(k % 2 == 1, -1.0, 1.0) * src + (k + k % 2) * room - mic
                walls = []
                for axis, ki in enumerate(k):
                    if ki == 0:
                        continue
                    # |k| reflections alternating between the two walls, starting with the one on the sign side
                    first = AXIS_WALL[(axis, int(np.sign(ki)))]
                    other = AXIS_WALL[(axis, -int(np.sign(ki)))]
                    walls += [first if i % 2 == 0 else other for i in range(abs(int(ki)))]
                out.append({"k": (int(kx), int(ky), int(kz)), "order": n, "distance": float(np.linalg.norm(disp)),
                            "walls": walls})
    out.sort(key=lambda d: d["distance"])
    return out


def direct_peak(h: np.ndarray, fraction: float = 0.5) -> int:
    """First local maximum of |h| that reaches ``fraction`` of the largest, as pick_arrivals takes it."""
    mag = np.abs(h)
    peaks, _ = find_peaks(np.concatenate([[0.0], mag, [0.0]]))
    peaks = peaks - 1
    return int(peaks[mag[peaks] >= fraction * mag.max()][0])


def parabolic(y: np.ndarray, i: np.ndarray) -> np.ndarray:
    i = np.asarray(i, dtype=int)
    i = np.clip(i, 1, len(y) - 2)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-18, 0.5 * (a - c) / np.where(np.abs(den) > 1e-18, den, 1.0), 0.0)
    return i + np.clip(off, -0.5, 0.5)


def matched_envelope(h: np.ndarray, before: int = 12, after: int = 36) -> tuple[np.ndarray, int]:
    """Envelope of the response correlated with its own direct sound (Hann-tapered template).

    The template runs from ``before`` samples ahead of the direct peak to ``after`` samples behind it; the
    output is aligned so that an arrival shaped like the direct sound peaks at the sample where its own
    peak sits, the convention of the raw response.
    """
    nd = direct_peak(h)
    lo, hi = max(0, nd - before), nd + after
    template = h[lo:hi] * np.hanning(hi - lo)
    corr = np.correlate(h, template, mode="full")[len(template) - 1:]  # corr[l] = sum_j h[l + j] t[j]
    aligned = np.concatenate([np.zeros(nd - lo), corr])[: len(h)]
    return np.abs(hilbert(aligned)), nd


def raw_envelope(h: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(h))


def pick(signal: np.ndarray, n_max: int = 120, floor: float = 0.02, separation: int = 2,
         window: int = WINDOW) -> np.ndarray:
    """Peak positions (samples) from the direct peak on, above ``floor`` x the direct peak, earliest first."""
    s = np.asarray(signal[:window], dtype=np.float64)
    nd = direct_peak(s)
    peaks, _ = find_peaks(np.concatenate([[0.0], s, [0.0]]), distance=max(1, separation))
    peaks = peaks - 1
    peaks = peaks[peaks >= nd]
    peaks = peaks[s[peaks] >= floor * s[nd]][:n_max]
    return parabolic(s, peaks)


def frozen_pick(h: np.ndarray) -> np.ndarray:
    """F0: the frozen picker of E10 (pruned |h|, 2% floor, 2-sample spacing, 120 peaks)."""
    import echo_lattice as pkg

    config = e10_run.harness.base_config(3)
    s = config.lattice
    magnitude = pkg.prune_small_taps(h[:WINDOW], config.identification.prune_threshold)
    return pkg.pick_arrivals(magnitude, s.max_echoes, relative_floor=s.relative_floor,
                             min_separation_taps=s.min_separation_taps)


def to_m(samples) -> np.ndarray:
    return np.asarray(samples, dtype=np.float64) * SPEED / RATE
