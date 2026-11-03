"""Metadata of every dEchorate SOFA file, without the impulse responses, compared with the file name.

    python -I -X utf8 e10_sofa_metadata.py <sofa folder> <out.csv>

One row per file: the room code, source and array the name gives, the same items as the file's
attributes describe them, the sample rate, the IR shape, Data.Delay, the room corners, the
temperature, and the source and receiver positions. Data.IR itself is not read.
"""

import csv
import re
import sys
from pathlib import Path

import h5py
import numpy as np

NAME = re.compile(r"^dEchorate_room(\d{6})_src([1-9])_arr([1-6])_mics(\d+)-(\d+)\.sofa$")


def text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def main():
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for path in sorted(folder.glob("*.sofa")):
        m = NAME.match(path.name)
        if not m:
            continue
        with h5py.File(path, "r") as f:
            a = f.attrs
            desc = text(a.get("RoomDescription", ""))
            code_attr = re.search(r"room_code:(\d+)", desc)
            src_attr = re.search(r"loudspeaker_id:(\d+)", text(a.get("SourceShortName", "")))
            arr_attr = re.search(r"array_id:(\d+)", text(a.get("ListenerShortName", "")))
            receivers = np.asarray(f["ReceiverPosition"])[:, :, 0]
            rows.append({
                "file": path.name, "name_code": m.group(1), "name_src": int(m.group(2)),
                "name_arr": int(m.group(3)), "name_mics": f"{m.group(4)}-{m.group(5)}",
                "attr_code": code_attr.group(1) if code_attr else "", "attr_src": src_attr.group(1) if src_attr else "",
                "attr_arr": arr_attr.group(1) if arr_attr else "", "room_description": desc,
                "source_description": text(a.get("SourceDescription", "")),
                "title": text(a.get("Title", "")), "date_modified": text(a.get("DateModified", "")),
                "sample_rate": float(np.asarray(f["Data.SamplingRate"]).ravel()[0]),
                "ir_shape": "x".join(map(str, f["Data.IR"].shape)),
                "delay": " ".join(f"{v:g}" for v in np.asarray(f["Data.Delay"]).ravel()),
                "corner_a": " ".join(f"{v:.4f}" for v in np.asarray(f["RoomCornerA"]).ravel()),
                "corner_b": " ".join(f"{v:.4f}" for v in np.asarray(f["RoomCornerB"]).ravel()),
                "temperature_k": float(np.asarray(f["RoomTemperature"]).ravel()[0]),
                "source_xyz": " ".join(f"{v:.5f}" for v in np.asarray(f["SourcePosition"]).ravel()),
                "listener_xyz": " ".join(f"{v:.5f}" for v in np.asarray(f["ListenerPosition"]).ravel()),
                "receivers_xyz": ";".join(" ".join(f"{v:.5f}" for v in r) for r in receivers),
            })
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} files -> {out}")


if __name__ == "__main__":
    main()
