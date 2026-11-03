"""Write freeze.json: code, environment, configurations and plans, fixed before any evaluation run.

Run once with the project's Python 3.10 environment, before the first run of
E1-E9 and before any dEchorate RIR is read::

    <project>/.venv/Scripts/python.exe freeze.py <analysis plan>

The analysis plan is the text written before the runs; only its SHA-256 goes
into freeze.json (``blueprint_sha256``), the plan itself is not in this
repository.

Refuses to overwrite an existing freeze.json and refuses to run when the
package's src/ has uncommitted changes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import harness

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "dechorate"
OUT = HERE / "freeze.json"

EXPERIMENTS = ["E1", "E1t", "E1b", "E2", "E3", "E4", "E5", "E6", "E9"]

# E10, fixed here before any RIR is read.
E10 = {
    "dataset": "dEchorate v2.0 (Di Carlo et al., 2021), Zenodo 10.5281/zenodo.6576203, CC BY 4.0",
    "files": {
        "dEchorate_database.csv": {"bytes": 2931172},
        "dEchorate_echo_notes.pkl": {"bytes": 21267, "use": "never loaded"},
        "main_geometry_from_echo_calibration.py": {"bytes": 13150, "use": "read as text, never executed"},
        "dEchorate_rirs_gzip7.hdf5": {"bytes": 4192401641, "md5": "8b42a0e99766acce8a100130774bd5b3"},
    },
    "room_m": [5.705, 5.965, 2.355],
    "speed_of_sound": {"formula": "331.3 + 0.606 * T", "temperature_c": 24.0, "m_s": 345.844},
    "subset": {
        "src_signal": "chirp",
        "room_codes_primary": ["000000", "000001", "000010", "000100", "001000",
                               "010000", "011000", "011100", "011110", "011111"],
        "room_codes_secondary": ["020002"],
        "room_code_digits": "floor, ceiling, west, south, east, north (1 = reflective)",
        "sources": {"directional_to_room": [0, 1, 2, 3], "directional_to_wall": [4, 5],
                    "omnidirectional_uncalibrated": [6, 7, 8]},
        "microphones": [2, 7, 12, 17, 22, 27],
        "n_rirs": {"primary": 540, "secondary": 54, "total": 594},
        "order": "room code as listed (primary, then secondary), source 0-8, microphone as listed",
    },
    "truth": ("room, microphone and directional-source positions as calibrated in "
              "main_geometry_from_echo_calibration.py; sources 6-8 use the script's uncalibrated "
              "positions, so their position and bearing errors are reported apart from the main results"),
    "latency": {
        "artificial_ms": "uniform(0, 20), rounded to 0.001 ms, applied as round(ms * 48) whole samples",
        "seed": "np.random.SeedSequence([20261024, sum(map(ord, 'E10'))]).spawn(594), one child per RIR in subset order",
        "onset_check": ("before any inversion and without the artificial shift: offset = time of max |h| "
                        "minus calibrated direct distance / c, over sources 0-3, the 10 primary codes and the "
                        "6 microphones (240 RIRs); if |median| exceeds one sample it is reported and added to "
                        "the true latency"),
    },
    "response": ("the dataset RIR at 48 kHz with the artificial shift prepended, truncated to "
                 "4800 + max(0, round(48 * median onset offset in ms)) taps so that the window after the "
                 "emission matches E2; no other processing"),
    "methods": ["M1", "M2", "M7"],
    "m7_labels": "image arrivals of the calibrated geometry at c above, max order 5 per axis, same assignment rule as in simulation",
    "config_overrides": {"acoustics.speed_of_sound_m_s": 345.844, "acoustics.temperature_c": None},
    "metrics": ("sorted-side error, success < 5 cm (10 and 15 cm as secondary results if the truth "
                "uncertainty exceeds 5 cm), latency error, bearing error (sources 0-5 only), reliable-flag "
                "precision and failure detection; grouped by room code, number of reflective surfaces and "
                "source type"),
    "rule": "no parameter is changed after any dEchorate RIR is read; needed changes go to the discussion only",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: freeze.py <analysis plan>")
        return 2
    plan_file = Path(sys.argv[1])
    if OUT.exists():
        print(f"{OUT} exists; a freeze is written once")
        return 1
    code = harness.code_identity()
    if code["src_dirty"] is not False:
        print("src/ has uncommitted changes (or git failed); commit first")
        return 1
    freeze_cfg_2d = harness.base_config(2).to_dict()
    freeze_cfg_3d = harness.base_config(3).to_dict()
    e10_cfg = harness.base_config(3).override(E10["config_overrides"]).to_dict()
    plans = {}
    for name in EXPERIMENTS:
        jobs = harness.plan(name)
        plans[name] = {
            "jobs": len(jobs),
            "conditions": sorted({j["condition"] for j in jobs}),
            "methods": sorted({m for j in jobs for m in j["methods"]}),
            "method_runs": sum(len(j["methods"]) for j in jobs),
            "jobs_sha256": sha256_json(jobs),
        }
    small = {}
    for name in ("dEchorate_database.csv", "dEchorate_echo_notes.pkl", "main_geometry_from_echo_calibration.py"):
        path = DATA / name
        small[name] = {"bytes": path.stat().st_size, "md5": md5_file(path)}
    freeze = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": code,
        "harness_sha256": sha256_bytes((HERE / "harness.py").read_bytes()),
        "blueprint_sha256": sha256_bytes(plan_file.read_bytes()),
        "freeze_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                       text=True, check=True).stdout.split(),
        },
        "constants": {"master_seed": harness.MASTER_SEED, "success_m": harness.SUCCESS_M,
                      "planar_overrides": harness.PLANAR, "spatial_overrides": harness.SPATIAL,
                      "ablations": {k: {"patched": sorted(v.get("patch", {})),
                                        "overrides": v.get("overrides", {})}
                                    for k, v in harness.ABLATIONS.items()}},
        "configs": {
            "planar": {"sha256": sha256_json(freeze_cfg_2d), "values": freeze_cfg_2d},
            "spatial": {"sha256": sha256_json(freeze_cfg_3d), "values": freeze_cfg_3d},
            "e10": {"sha256": sha256_json(e10_cfg), "values": e10_cfg},
        },
        "plans": plans,
        "e10": {**E10, "files_checked_at_freeze": small},
    }
    OUT.write_text(json.dumps(freeze, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"sha256 {sha256_bytes(OUT.read_bytes())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
