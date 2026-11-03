"""Post-review PR10: peak-to-noise ratio of the identified responses of E3. Not pre-registered.

    <echo-lattice>/.venv/Scripts/python.exe post_review/pr10_response_snr.py

E3 sets the noise level at the microphone during the 2 s white-noise excitation. The identified
response averages that noise over the whole excitation, so its own peak-to-noise ratio is higher.
For every E3 response this rebuilds the measurement with the original seeds and reports
20 log10(max|h_hat| / rms(h_hat - h)), with h the noise-free simulated response delayed by the same
latency and cut to the identified length. Nothing frozen is changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402


def main() -> int:
    import echo_lattice as pkg
    import pandas as pd

    out = HERE.parent / "results" / "post_review"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in harness.plan("E3"):
        scene = harness.Scene(**job["scene"])
        config = harness.scene_config(harness.base_config(2), scene, job.get("config"))
        weights, shift, truth = harness.measure(config, scene)
        reference = pkg.delay_response(truth.rir, shift)
        h = np.zeros_like(weights)
        n = min(h.size, reference.size)
        h[:n] = reference[:n]
        error = weights - h
        rows.append({"condition": job["condition"], "scene": job["index"],
                     "peak_to_noise_db": 20 * np.log10(np.abs(weights).max() / np.sqrt(np.mean(error**2)))})
    table = pd.DataFrame(rows)
    table.to_csv(out / "PR10_E3_response_snr.csv", index=False)
    summary = table.groupby("condition").peak_to_noise_db.describe(percentiles=[0.25, 0.5, 0.75])
    meta = {"code": harness.code_identity(),
            "note": "post-review exploratory analysis, not pre-registered; no setting changed",
            "summary_db": json.loads(summary.to_json(orient="index"))}
    (out / "PR10_E3_response_snr.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
