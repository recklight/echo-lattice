"""Exploratory follow-up of the failed label check: which surface changes in each room code.

    <project>/.venv/Scripts/python.exe e10_label_diagnostics.py --rirs <npz> [--out results]

Not part of the pre-specified analysis. For every source-microphone pair, the first-order ratios of
e10_label_check.py are compared with the same pair in room 000000 (paired differences, median over the
54 pairs), and every pair of room codes is compared by the correlation of the same pair's first 100 ms.
Writes E10_label_diagnostics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import e10_label_check as lc
import e10_run as e


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=e.HERE / "results")
    args = parser.parse_args()
    freeze = json.loads(e.FREEZE.read_text(encoding="utf-8"))
    speed = freeze["e10"]["speed_of_sound"]["m_s"]
    geo = e.geometry(e.DATA / "main_geometry_from_echo_calibration.py")
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        keys = list(zip(z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()))
    where = {key: k for k, key in enumerate(keys)}
    check = json.loads((args.out / "E10_label_check.json").read_text(encoding="utf-8"))
    shift = e.RATE * check["onset_offset_median_ms"] / 1000.0
    codes = freeze["e10"]["subset"]["room_codes_primary"] + freeze["e10"]["subset"]["room_codes_secondary"]
    pairs = sorted({(j, i) for _, j, i in keys})
    names = list(lc.SURFACES)

    ratio = np.zeros((len(codes), len(pairs), len(names)))
    for a, code in enumerate(codes):
        for b, (j, i) in enumerate(pairs):
            src, mic, h = geo["sources"][j], geo["mics"][i], rirs[where[(code, j, i)]]
            direct = lc.window_peak(h, np.linalg.norm(src - mic) / speed * e.RATE + shift)
            for c, (_, axis, side) in enumerate(lc.SURFACES.values()):
                arrival = np.linalg.norm(lc.image(src, geo["room"], axis, side) - mic) / speed * e.RATE + shift
                ratio[a, b, c] = lc.window_peak(h, arrival) / direct
    paired = np.median(ratio - ratio[codes.index("000000")][None], axis=1)  # [code, surface]

    head = rirs[:, :4800]
    unit = head / np.linalg.norm(head, axis=1, keepdims=True)
    corr = np.zeros((len(codes), len(codes)))
    for a, ca in enumerate(codes):
        for b, cb in enumerate(codes):
            corr[a, b] = np.median([unit[where[(ca, j, i)]] @ unit[where[(cb, j, i)]] for j, i in pairs])

    result = {"note": "exploratory, not pre-specified", "surfaces": names, "codes": codes,
              "paired_median_ratio_change_vs_000000": {c: dict(zip(names, map(float, paired[a]))) for a, c in enumerate(codes)},
              "median_same_pair_correlation_first_100ms": {ca: dict(zip(codes, map(float, corr[a]))) for a, ca in enumerate(codes)}}
    (args.out / "E10_label_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("paired change of r against 000000 (median over 54 pairs)")
    print("code    " + " ".join(f"{s:>8s}" for s in names))
    for a, c in enumerate(codes):
        print(c, " ".join(f"{v:8.3f}" for v in paired[a]))
    print("\nmedian same-pair correlation, first 100 ms")
    print("        " + " ".join(f"{c:>6s}" for c in codes))
    for a, c in enumerate(codes):
        print(c, " ".join(f"{v:6.3f}" for v in corr[a]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
