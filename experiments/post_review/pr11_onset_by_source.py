"""Post-review PR11: the onset check of E10 for every source type. Not pre-registered.

    <echo-lattice>/.venv/Scripts/python.exe post_review/pr11_onset_by_source.py --rirs <e10_responses.npz>

The frozen onset check of e10_run.py used the four room-facing loudspeakers only. This repeats it
(time of max |h| minus the calibrated direct path over c, no artificial latency) for the 540 primary
responses, split by source type: room-facing, wall-facing, and the three uncalibrated
omnidirectional loudspeakers. Nothing frozen is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e10_run as e  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        codes, srcs, mics = z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()
    groups: dict[str, list[float]] = {}
    for k in range(len(codes)):
        if codes[k] == "020002":
            continue
        peak = int(np.argmax(np.abs(rirs[k])))
        direct = float(np.linalg.norm(geo["sources"][srcs[k]] - geo["mics"][mics[k]])) / speed
        groups.setdefault(e.SOURCE_TYPE[srcs[k]], []).append(1000.0 * (peak / e.RATE - direct))
    one_sample = 1000.0 / e.RATE
    out = {}
    for name, values in groups.items():
        v = np.abs(np.array(values))
        out[name] = {"n": int(v.size), "median_abs_ms": float(np.median(v)), "max_abs_ms": float(v.max()),
                     "within_one_sample": int((v <= one_sample).sum())}
    out["note"] = "post-review exploratory analysis, not pre-registered"
    target = HERE.parent / "results" / "post_review" / "PR11_onset_by_source.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def first_pick_offsets(rirs_path: Path) -> dict:
    """The picker's first peak (taken as the direct sound) minus the calibrated direct path, by source type."""
    import harness

    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    config = harness.base_config(3).override(freeze["e10"]["config_overrides"])
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    with np.load(rirs_path, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        codes, srcs, mics = z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()
    groups: dict[str, list[float]] = {}
    for k in range(len(codes)):
        if codes[k] == "020002":
            continue
        picks = np.sort(harness.picked_lengths(rirs[k][:4800], config))
        direct = float(np.linalg.norm(geo["sources"][srcs[k]] - geo["mics"][mics[k]]))
        groups.setdefault(e.SOURCE_TYPE[srcs[k]], []).append(1000.0 * (picks[0] - direct) / speed)
    one_sample = 1000.0 / e.RATE
    out = {}
    for name, values in groups.items():
        v = np.abs(np.array(values))
        out[name] = {"n": int(v.size), "median_abs_ms": float(np.median(v)), "max_abs_ms": float(v.max()),
                     "within_one_sample": int((v <= one_sample).sum())}
    return out
