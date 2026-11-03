"""Check the wall configuration in the SOFA file names against the responses (rule in e10_data_note.md).

    <project>/.venv/Scripts/python.exe e10_label_check.py --rirs <npz from e10_extract.py> [--out results]

For every primary room code, the first-order reflection of each surface that changes between codes is
measured relative to the direct sound, at the arrival times of the calibrated geometry shifted by the
median offset of the frozen onset check. m(code, surface) is the median over the 54 source-microphone
pairs; a surface passes when every code with it reflective has a larger m than every code with it
absorbent. Writes E10_label_check.json; e10_run.py refuses to run unless it passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import e10_run as e

# digit of the room code (floor, ceiling, west, south, east, north) -> axis and which wall of it
SURFACES = {"ceiling": (1, 2, "high"), "west": (2, 0, "low"), "south": (3, 1, "low"),
            "east": (4, 0, "high"), "north": (5, 1, "high")}
HALF = 12  # samples on each side of an arrival, 0.25 ms at 48 kHz


def image(src: np.ndarray, room: np.ndarray, axis: int, side: str) -> np.ndarray:
    p = src.copy()
    p[axis] = -p[axis] if side == "low" else 2.0 * room[axis] - p[axis]
    return p


def window_peak(h: np.ndarray, t: float) -> float:
    c = int(round(t))
    lo, hi = max(0, c - HALF), min(h.size, c + HALF + 1)
    return float(np.max(np.abs(h[lo:hi]))) if hi > lo else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=e.HERE / "results")
    args = parser.parse_args()
    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    room = geo["room"]
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        index = list(zip(z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()))
    offset_ms = float(np.median(e.onset_offsets_ms(rirs, index, geo, speed)))
    shift = e.RATE * offset_ms / 1000.0

    primary = freeze["e10"]["subset"]["room_codes_primary"]
    ratios: dict[str, dict[str, list[float]]] = {c: {s: [] for s in SURFACES} for c in primary}
    for k, (code, j, i) in enumerate(index):
        if code not in ratios:
            continue
        src, mic, h = geo["sources"][j], geo["mics"][i], rirs[k]
        direct = window_peak(h, np.linalg.norm(src - mic) / speed * e.RATE + shift)
        for name, (_, axis, side) in SURFACES.items():
            arrival = np.linalg.norm(image(src, room, axis, side) - mic) / speed * e.RATE + shift
            ratios[code][name].append(window_peak(h, arrival) / direct if direct > 0 else np.nan)
    m = {c: {s: float(np.nanmedian(v)) for s, v in per.items()} for c, per in ratios.items()}
    verdict = {}
    for name, (digit, _, _) in SURFACES.items():
        on = [m[c][name] for c in primary if c[digit] != "0"]
        off = [m[c][name] for c in primary if c[digit] == "0"]
        verdict[name] = {"reflective_min": min(on), "absorbent_max": max(off), "pass": bool(min(on) > max(off))}
    result = {"onset_offset_median_ms": offset_ms, "half_window_samples": HALF, "pairs_per_code": 54,
              "median_ratio": m, "surfaces": verdict, "pass": all(v["pass"] for v in verdict.values())}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "E10_label_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    print("label check", "PASSED" if result["pass"] else "FAILED")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
