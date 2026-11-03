"""E10: the lattice inversion on measured impulse responses (dEchorate), with the frozen parameters.

    <project>/.venv/Scripts/python.exe e10_run.py --rirs <npz from e10_extract.py> --workers 14

Reads the 594 responses e10_extract.py wrote, the calibrated geometry from
main_geometry_from_echo_calibration.py (parsed as text with ast.literal_eval,
never executed) and freeze.json. It first checks the response onset against
the calibrated direct path, then runs M1, M2 and M7 on every response with the
E10 configuration recorded in freeze.json. Nothing here changes a parameter of
the inversion.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import harness

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "dechorate"
FREEZE = HERE / "freeze.json"
RATE = 48_000
SOURCE_TYPE = {0: "directional_to_room", 1: "directional_to_room", 2: "directional_to_room",
               3: "directional_to_room", 4: "directional_to_wall", 5: "directional_to_wall",
               6: "omnidirectional", 7: "omnidirectional", 8: "omnidirectional"}


# --------------------------------------------------------------------- geometry
def literal_array(text: str, name: str) -> np.ndarray:
    """The numbers of ``name = np.array([...])`` or ``name = [...]`` in the calibration script."""
    found = re.search(rf"^\s*{name}\s*=\s*(?:np\.array\(\s*)?(\[.*?\])\s*\)?\s*(?:#.*)?$", text, re.S | re.M)
    if found is None:
        raise ValueError(f"{name} not found")
    return np.array(ast.literal_eval(found.group(1)), dtype=np.float64)


def geometry(script: Path) -> dict:
    text = script.read_text(encoding="utf-8")
    room = literal_array(text, "room_size")
    mics = literal_array(text, "mics").T  # [30, 3]
    sources = np.vstack([literal_array(text, "srcs_dir").T, literal_array(text, "srcs_omn").T])  # [9, 3]
    return {"room": room, "mics": mics, "sources": sources}


def n_reflective(code: str) -> int:
    return sum(int(d != "0") for d in code)


# --------------------------------------------------------------------- subset
def subset_order(freeze: dict) -> list[tuple[str, int, int]]:
    s = freeze["e10"]["subset"]
    codes = s["room_codes_primary"] + s["room_codes_secondary"]
    sources = sorted(v for group in s["sources"].values() for v in group)
    return [(c, j, i) for c in codes for j in sources for i in s["microphones"]]


def artificial_latencies(n: int) -> list[tuple[float, int]]:
    """Per response: (latency in ms, child seed) in subset order, as freeze.json states."""
    root = np.random.SeedSequence([harness.MASTER_SEED, sum(map(ord, "E10"))])
    out = []
    for child in root.spawn(n):
        rng = np.random.default_rng(child)
        out.append((float(np.round(rng.uniform(0.0, 20.0), 3)), int(rng.integers(0, 2**31 - 1))))
    return out


def onset_offsets_ms(rirs, index, geo, speed) -> np.ndarray:
    """Time of max |h| minus calibrated direct distance / c, sources 0-3, primary codes."""
    out = []
    for k, (code, j, i) in enumerate(index):
        if j > 3 or code == "020002":
            continue
        peak = int(np.argmax(np.abs(rirs[k])))
        direct = float(np.linalg.norm(geo["sources"][j] - geo["mics"][i])) / speed
        out.append(1000.0 * (peak / RATE - direct))
    return np.array(out)


# ------------------------------------------------------------------------- job
def job(spec: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    config = harness.base_config(3).override(spec["overrides"])
    scene = harness.Scene(size=spec["room"], mic=spec["mic"], src=spec["src"],
                          latency_ms=spec["true_latency_ms"], noise_seed=spec["seed"])
    response = np.asarray(spec["response"], dtype=np.float64)
    speed = config.acoustics.speed
    base = {"experiment": "E10", "condition": spec["code"], "scene": spec["k"], "room_code": spec["code"],
            "n_reflective": n_reflective(spec["code"]), "src_id": spec["src_id"],
            "source_type": SOURCE_TYPE[spec["src_id"]], "mic_id": spec["mic_id"],
            "true_sides": " x ".join(f"{x:.3f}" for x in spec["room"]),
            "artificial_latency_ms": spec["artificial_ms"], "true_latency_ms": spec["true_latency_ms"],
            "rec_artifacts": spec["rec_artifacts"]}
    rows = []
    for method in ("M1", "M2", "M7"):
        row = {**base, "method": method}
        started = time.perf_counter()
        try:
            if method == "M1":
                est = harness.run_lattice(response, config)
            elif method == "M2":
                est = harness.run_lattice(response, config, {"lattice.latency_ms": spec["true_latency_ms"]})
            else:
                est, used, total = harness.run_oracle_labels(response, config, scene,
                                                             1e-3 * spec["true_latency_ms"] * speed, spec["seed"])
                row["labelled_echoes"], row["picked_echoes"] = used, total
            row.update(harness.lattice_row(est, scene, speed, spec["true_latency_ms"]))
            for thr in (0.10, 0.15):
                row[f"success_{round(100 * thr)}cm"] = bool(row["room_error_m"] < thr)
            row["est_microphone_m"] = " ".join(f"{v:.4f}" for v in est.microphone_m)
            row["est_source_m"] = " ".join(f"{v:.4f}" for v in est.source_m)
        except Exception as exc:  # InversionError and the like: a failure, kept in the table
            row.update(room_error_m=float("inf"), success=False, reliable=False,
                       success_10cm=False, success_15cm=False, error=f"{type(exc).__name__}: {exc}")
        row["time_s"] = time.perf_counter() - started
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rirs", type=Path, required=True, help="npz written by e10_extract.py")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--limit", type=int, default=None, help="First N responses only (smoke test).")
    parser.add_argument("--speed", type=float, default=None,
                        help="Speed of sound of the secondary run, m/s (346.98, the dataset paper's value); "
                             "the frozen value otherwise.")
    parser.add_argument("--tag", default="", help="Suffix of the output files.")
    parser.add_argument("--label-check-decision", default="",
                        help="Where the decision to run despite a failed label check is recorded (goes into the meta).")
    args = parser.parse_args()
    stem = f"E10{args.tag}" if not args.limit else f"E10{args.tag}_limit{args.limit}"
    running = args.out / f"{stem}.running"
    if (args.out / f"{stem}.csv").exists() or running.exists():
        print(f"{stem} is already written or running; refusing to start it twice")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    running.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
    try:
        return run(args, stem)
    finally:
        running.unlink(missing_ok=True)


def run(args, stem: str) -> int:
    import pandas as pd

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    e10 = freeze["e10"]
    code = harness.code_identity()
    # The package source has to be the frozen one; the commit itself is only recorded.
    if code["src_tree_sha256"] != freeze["code"]["src_tree_sha256"]:
        print("the package source differs from the frozen one; refusing to run")
        return 1
    geo = geometry(DATA / "main_geometry_from_echo_calibration.py")
    room = geo["room"].tolist()
    assert np.allclose(room, e10["room_m"]), room
    frozen_speed = e10["speed_of_sound"]["m_s"]
    speed = args.speed if args.speed is not None else frozen_speed
    if speed not in (frozen_speed, 346.98):  # the two values fixed before any response was read
        print(f"speed {speed} is neither the frozen value nor the secondary 346.98 m/s; refusing to run")
        return 1
    overrides = {**e10["config_overrides"], "acoustics.speed_of_sound_m_s": speed}
    assert harness.base_config(3).override(overrides).acoustics.speed == speed

    label_check = args.out / "E10_label_check.json"
    if not label_check.exists():
        print("run e10_label_check.py first; refusing to run")
        return 1
    label_passed = bool(json.loads(label_check.read_text(encoding="utf-8"))["pass"])
    if not label_passed and not args.label_check_decision:
        print("the label check failed; running anyway needs --label-check-decision; refusing to run")
        return 1
    with np.load(args.rirs, allow_pickle=False) as z:
        rirs = z["rirs"].astype(np.float64)
        index = list(zip(z["room_code"].tolist(), z["src_id"].astype(int).tolist(), z["mic_id"].astype(int).tolist()))
        artifacts = z["rec_artifacts"].astype(float).tolist()
        rate = int(z["sample_rate"])
        file_src, file_mic = z["source_xyz"], z["receiver_xyz"]
    # the files' own positions against the frozen truth (calibration script)
    gap_src = max(float(np.linalg.norm(file_src[k] - geo["sources"][j])) for k, (_, j, _) in enumerate(index))
    gap_mic = max(float(np.linalg.norm(file_mic[k] - geo["mics"][i])) for k, (_, _, i) in enumerate(index))
    if max(gap_src, gap_mic) > 0.01:
        print(f"file positions differ from the calibrated geometry by up to {max(gap_src, gap_mic):.3f} m; refusing to run")
        return 1
    if rate != RATE:
        print(f"sample rate {rate}, expected {RATE}; refusing to run")
        return 1
    expected = subset_order(freeze)
    if index != expected:
        print("the npz does not hold the frozen subset in the frozen order")
        return 1

    offsets = onset_offsets_ms(rirs, index, geo, speed)
    median_ms = float(np.median(offsets))
    one_sample_ms = 1000.0 / RATE
    added_ms = median_ms if abs(median_ms) > one_sample_ms else 0.0
    window = 4800 + max(0, round(RATE * median_ms / 1000.0))
    onset = {"n": int(offsets.size), "median_ms": median_ms,
             "q25_ms": float(np.quantile(offsets, 0.25)), "q75_ms": float(np.quantile(offsets, 0.75)),
             "min_ms": float(offsets.min()), "max_ms": float(offsets.max()),
             "added_to_true_latency_ms": added_ms, "window_taps": window}
    print("onset check:", json.dumps(onset))

    latencies = artificial_latencies(len(index))
    specs = []
    for k, ((code_k, j, i), (art_ms, seed)) in enumerate(zip(index, latencies)):
        shift = round(art_ms * RATE / 1000.0)
        response = np.concatenate([np.zeros(shift), rirs[k]])[:window]
        specs.append({"k": k, "code": code_k, "src_id": j, "mic_id": i, "room": room,
                      "mic": geo["mics"][i].tolist(), "src": geo["sources"][j].tolist(),
                      "artificial_ms": art_ms, "true_latency_ms": 1000.0 * shift / RATE + added_ms,
                      "seed": seed, "overrides": overrides, "response": response,
                      "rec_artifacts": artifacts[k]})
    if args.limit:
        specs = specs[: args.limit]

    meta = {"experiment": "E10", "jobs": len(specs), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code": code, "freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rirs_sha256": hashlib.sha256(args.rirs.read_bytes()).hexdigest(), "onset_check": onset,
            "file_position_gap_m": {"sources": gap_src, "microphones": gap_mic},
            "speed_of_sound_m_s": speed, "speed_is_frozen_value": speed == frozen_speed,
            "label_check_sha256": hashlib.sha256(label_check.read_bytes()).hexdigest(),
            "label_check_passed": label_passed, "label_check_decision": args.label_check_decision,
            "python": sys.version, "workers": args.workers}
    rows, failures = [], 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, spec) for spec in specs]
        for n_done, fut in enumerate(as_completed(futures), 1):
            try:
                rows += fut.result()
            except Exception as exc:
                failures += 1
                rows.append({"experiment": "E10", "error": repr(exc)})
            if n_done % 20 == 0 or n_done == len(futures):
                print(f"{stem}: {n_done}/{len(futures)} responses, {time.perf_counter() - started:.0f} s", flush=True)
    table = pd.DataFrame(rows).sort_values(["scene", "method"], na_position="last")
    table.to_csv(args.out / f"{stem}.csv", index=False)
    meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                wall_s=time.perf_counter() - started, job_failures=failures)
    (args.out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out / (stem + '.csv')} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
