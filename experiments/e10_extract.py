"""Collect the E10 responses from the dEchorate SOFA files into one npz, in the order freeze.json fixes.

    python -I -X utf8 e10_extract.py <sofa folder> <dEchorate_database.csv> <freeze.json> <out.npz> <manifest.csv>

For every room code, source and middle microphone of the frozen subset it reads one row of Data.IR
from dEchorate_room<code>_src<source+1>_arr<array>_mics<a>-<b>.sofa, with the file's source and
receiver positions, and takes rec_artifacts from the chirp recording of the same code, source and
microphone in the database table. The manifest lists every downloaded file with its size and SHA-256.
Run with -I from outside the data folder; the SOFA files are only read through h5py.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifacts_table(csv_path: Path) -> dict:
    out = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["src_signal"] != "chirp":
                continue
            key = (row["room_code"].zfill(6), int(float(row["src_id"])), int(float(row["mic_id"])))
            out[key] = float(row["rec_artifacts"] or "nan")
    return out


def main():
    folder, table, freeze_path, out, manifest = map(Path, sys.argv[1:6])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    s = freeze["e10"]["subset"]
    codes = s["room_codes_primary"] + s["room_codes_secondary"]
    sources = sorted(v for group in s["sources"].values() for v in group)
    mics = s["microphones"]
    artifacts = artifacts_table(table)

    rirs, code_col, src_col, mic_col, art_col, src_xyz, rcv_xyz, rates, files = [], [], [], [], [], [], [], [], []
    for code in codes:
        for j in sources:
            for i in mics:
                arr, within = i // 5 + 1, i % 5
                name = f"dEchorate_room{code}_src{j + 1}_arr{arr}_mics{5 * (arr - 1) + 1}-{5 * arr}.sofa"
                with h5py.File(folder / name, "r") as f:
                    ir = np.asarray(f["Data.IR"])
                    if ir.ndim != 3 or ir.shape[:2] != (1, 5):
                        raise ValueError(f"{name}: Data.IR shape {ir.shape}")
                    rirs.append(ir[0, within].astype(np.float64))
                    rates.append(float(np.asarray(f["Data.SamplingRate"]).ravel()[0]))
                    src_xyz.append(np.asarray(f["SourcePosition"], dtype=np.float64).ravel()[:3])
                    rcv_xyz.append(np.asarray(f["ReceiverPosition"], dtype=np.float64)[within, :, 0])
                code_col.append(code)
                src_col.append(j)
                mic_col.append(i)
                art_col.append(artifacts.get((code, j, i), float("nan")))
                files.append(name)
    if len(set(rates)) != 1:
        raise ValueError(f"sample rates differ: {sorted(set(rates))}")
    lengths = {r.size for r in rirs}
    if len(lengths) != 1:
        raise ValueError(f"IR lengths differ: {sorted(lengths)}")
    np.savez_compressed(
        out, rirs=np.vstack(rirs), room_code=np.array(code_col, dtype="U6"), src_id=np.array(src_col),
        mic_id=np.array(mic_col), rec_artifacts=np.array(art_col), sample_rate=np.array(int(rates[0])),
        source_xyz=np.vstack(src_xyz), receiver_xyz=np.vstack(rcv_xyz), sofa_file=np.array(files))
    print(f"{len(rirs)} responses of {lengths.pop()} samples at {rates[0]:g} Hz -> {out}")
    print(f"recordings with rec_artifacts != 0: {int(np.nansum(np.array(art_col) != 0))}, "
          f"without a table row: {int(np.isnan(art_col).sum())}")

    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "bytes", "sha256"])
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.endswith(".part"):
                writer.writerow([path.name, path.stat().st_size, sha256(path)])
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
