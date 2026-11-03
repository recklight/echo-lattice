"""Post-review PR12, step 1: copy the dEchorate HDF5 echo annotations into an npz (no pickle).

    python -I -X utf8 post_review/pr12_export_annotations.py <dEchorate_annotations.h5> <out.npz>

The echo-lattice environment has no h5py and stays as freeze.json records it, so this step runs
in another Python with h5py. Only numeric arrays and the one-letter wall labels are copied.
"""

import hashlib
import sys

import h5py
import numpy as np


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    with h5py.File(src, "r") as f:
        arrays = {k: f[k][()] for k in ("echo_toa", "echo_amp", "microphones", "room_size",
                                        "sources_directional_position")}
        arrays["echo_wall"] = np.array([[[w.decode() for w in row] for row in plane] for plane in f["echo_wall"][()]])
    np.savez(out, **arrays, source_sha256=np.array(hashlib.sha256(open(src, "rb").read()).hexdigest()))
    print("wrote", out, {k: v.shape for k, v in arrays.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
