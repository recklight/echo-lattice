"""Collect the development responses of the diagnosis (dev-A, dev-B) from the dEchorate SOFA files.

    python -I -X utf8 d0_extract.py <sofa folder> <out.npz>

dev-A is the first capsule of every array (microphones 0, 5, ..., 25), dev-B the fifth (4, 9, ..., 29),
for the ten primary room codes and the six directional loudspeakers (sources 0-5). Only these two rows of
Data.IR are read from each file; the middle capsule that E10 used is never decoded. Run with -I from
outside the data folder; the SOFA files are only read through h5py.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

CODES = ["000000", "000001", "000010", "000100", "001000", "010000", "011000", "011100", "011110", "011111"]
SOURCES = range(6)
CAPSULES = {"dev-A": 0, "dev-B": 4}


def main():
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    rirs, split, code_col, src_col, mic_col, rates, src_xyz, rcv_xyz = [], [], [], [], [], [], [], []
    for code in CODES:
        for j in SOURCES:
            for arr in range(1, 7):
                name = f"dEchorate_room{code}_src{j + 1}_arr{arr}_mics{5 * (arr - 1) + 1}-{5 * arr}.sofa"
                with h5py.File(folder / name, "r") as f:
                    shape = f["Data.IR"].shape
                    if len(shape) != 3 or shape[:2] != (1, 5):
                        raise ValueError(f"{name}: Data.IR shape {shape}")
                    rate = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])
                    source = np.asarray(f["SourcePosition"], dtype=np.float64).ravel()[:3]
                    for label, within in CAPSULES.items():
                        rirs.append(np.asarray(f["Data.IR"][0, within, :], dtype=np.float64))
                        rcv_xyz.append(np.asarray(f["ReceiverPosition"][within, :, 0], dtype=np.float64))
                        split.append(label)
                        code_col.append(code)
                        src_col.append(j)
                        mic_col.append(5 * (arr - 1) + within)
                        rates.append(rate)
                        src_xyz.append(source)
    if len(set(rates)) != 1:
        raise ValueError(f"sample rates differ: {sorted(set(rates))}")
    np.savez_compressed(out, rirs=np.vstack(rirs), split=np.array(split), room_code=np.array(code_col, dtype="U6"),
                        src_id=np.array(src_col), mic_id=np.array(mic_col), sample_rate=np.array(int(rates[0])),
                        source_xyz=np.vstack(src_xyz), receiver_xyz=np.vstack(rcv_xyz))
    print(f"{len(rirs)} responses ({split.count('dev-A')} dev-A, {split.count('dev-B')} dev-B) at {rates[0]:g} Hz -> {out}")


if __name__ == "__main__":
    main()
