"""Post-review PR12: first-order echoes, dataset annotations against the calibrated geometry. Not pre-registered.

    <echo-lattice>/.venv/Scripts/python.exe post_review/pr12_annotation_reference.py --rirs <e10_responses.npz> --annotations <dechorate_annotations.npz>

The annotations (step 1, pr12_export_annotations.py) give the arrival times of the direct sound and
the six first-order reflections for the 6 directional loudspeakers and all 30 microphones. For the
360 empty-room responses of E10 with a directional loudspeaker, this checks how many of the six
first-order arrivals have a picked peak within 2 cm, 0.1 ms and 0.5 ms, once with the annotated
times and once with the image arrivals of the calibrated geometry, split by whether the wall is
reflective in that configuration, against picks redrawn uniformly over the same span (1,000 draws).
It also reports how far the annotated times are from the geometric ones. Picks are those of the
frozen picker on the synchronized response (no artificial latency). Nothing frozen is changed.
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
WALL_AXIS = {"w": (0, 0), "e": (0, 1), "s": (1, 0), "n": (1, 1), "f": (2, 0), "c": (2, 1)}
CODE_INDEX = {"f": 0, "c": 1, "w": 2, "s": 3, "e": 4, "n": 5}  # room code: floor, ceiling, west, south, east, north


def image(src, wall, room):
    axis, side = WALL_AXIS[wall]
    out = src.copy()
    out[axis] = -src[axis] if side == 0 else 2 * room[axis] - src[axis]
    return out


def hits(picks, arrivals, tol):
    gap = np.abs(arrivals[:, None] - picks[None, :]).min(axis=1)
    return gap < tol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "post_review")
    args = parser.parse_args()
    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    config = harness.base_config(3).override(freeze["e10"]["config_overrides"])
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    room = geo["room"]
    with np.load(args.annotations, allow_pickle=False) as a:
        toa, walls = a["echo_toa"], a["echo_wall"]
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        codes, srcs, mics = z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()

    # annotated vs geometric arrival times, all 7 x 30 x 6
    gaps = []
    for j in range(6):
        for i in range(30):
            for k in range(7):
                w = str(walls[k, i, j])
                s = geo["sources"][j] if w == "d" else image(geo["sources"][j], w, room)
                gaps.append(abs(toa[k, i, j] - np.linalg.norm(s - geo["mics"][i]) / speed))
    gaps = np.array(gaps)

    rng = np.random.default_rng([harness.MASTER_SEED, 12])
    tols = {"2cm": 0.02, "0.1ms": 1e-4 * speed, "0.5ms": 5e-4 * speed}
    tally = {}
    n_resp = 0
    for k in range(len(codes)):
        if codes[k] == "020002" or srcs[k] > 5:
            continue
        n_resp += 1
        picks = np.sort(harness.picked_lengths(rirs[k][:4800], config))
        lo, hi = picks[0], picks[-1]
        rows = [r for r in range(7) if str(walls[r, mics[k], srcs[k]]) != "d"]
        wl = [str(walls[r, mics[k], srcs[k]]) for r in rows]
        ref = {"annotation": toa[rows, mics[k], srcs[k]] * speed,
               "geometry": np.array([np.linalg.norm(image(geo["sources"][srcs[k]], w, room) - geo["mics"][mics[k]]) for w in wl])}
        state = ["reflective" if codes[k][CODE_INDEX[w]] != "0" else "absorbent" for w in wl]
        draws = [np.sort(np.concatenate([[lo], rng.uniform(lo, hi, picks.size - 1)])) for _ in range(N_DRAWS)]
        for name, arr in ref.items():
            inside = (arr >= lo) & (arr <= hi)
            for tname, tol in tols.items():
                h = hits(picks, arr, tol)
                rand = np.mean([hits(d, arr, tol) for d in draws], axis=0)
                for idx, st in enumerate(state):
                    key = (name, tname, st)
                    t = tally.setdefault(key, {"arrivals": 0, "inside_span": 0, "hit": 0, "hit_inside": 0, "random_expected": 0.0})
                    t["arrivals"] += 1
                    t["inside_span"] += int(inside[idx])
                    t["hit"] += int(h[idx])
                    t["hit_inside"] += int(h[idx] and inside[idx])
                    t["random_expected"] += float(rand[idx])
    rows = [{"reference": n, "tolerance": t, "wall": s, **v} for (n, t, s), v in sorted(tally.items())]
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "PR12_annotation_reference.csv", index=False)
    meta = {"responses": n_resp, "annotation_vs_geometry_max_s": float(gaps.max()),
            "annotation_vs_geometry_median_s": float(np.median(gaps)), "speed_m_s": speed, "n_draws": N_DRAWS,
            "annotations_npz_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            "rirs_sha256": hashlib.sha256(args.rirs.read_bytes()).hexdigest(), "code": harness.code_identity(),
            "note": "post-review exploratory analysis, not pre-registered; no setting changed; no pickle loaded"}
    (args.out / "PR12_annotation_reference.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str))
    print(pd.DataFrame(rows).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
