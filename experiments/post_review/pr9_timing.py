"""Post-review PR9: wall time of the 3-D M1 on the first 20 scenes of E2, in one process. Not pre-registered.

    <echo-lattice>/.venv/Scripts/python.exe post_review/pr9_timing.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["OMP_NUM_THREADS"] = "1"  # set before numpy is imported
import harness  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    out = HERE.parent / "results" / "post_review"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in harness.plan("E2")[:20]:
        scene = harness.Scene(**job["scene"])
        config = harness.scene_config(harness.base_config(3), scene, job.get("config"))
        response, _, _ = harness.measure(config, scene)
        started = time.perf_counter()
        try:
            harness.run_lattice(response, config)
            returned = True
        except Exception:
            returned = False
        rows.append({"scene": job["index"], "time_s": time.perf_counter() - started, "returned": returned})
    table = pd.DataFrame(rows)
    table.to_csv(out / "PR9_E2_timing.csv", index=False)
    meta = {"n": len(table), "median_s": float(table.time_s.median()), "q25": float(table.time_s.quantile(0.25)),
            "q75": float(table.time_s.quantile(0.75)), "code": harness.code_identity(), "workers": 1}
    (out / "PR9_E2_timing.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(table.time_s.describe())


if __name__ == "__main__":
    main()
