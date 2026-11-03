"""Post-review PR8 and PR2 (from the original result tables only). Not pre-registered.

    <echo-lattice>/.venv/Scripts/python.exe post_review/pr8_tables.py

PR8: per-condition 2 x 2 tables of the reliability flag for the fits pooled in E7 (E1, the five E3
SNR levels, the seven E5 conditions), the share of successful fits flagged reliable, and cluster
bootstrap intervals (10,000 resamples) for the pooled precision and failure detection (clusters:
experiment x scene) and for the E10 M7 success rate (clusters: source position). PR2: the indices of
the E1 failures and the source of the M1 row of Table 4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = RES / "post_review"
B = 10_000


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ("E1", "E3", "E5"):
        t = pd.read_csv(RES / f"{name}.csv")
        t = t[t.method == "M1"].copy()
        t["exp"] = name
        frames.append(t[["exp", "condition", "scene", "success", "reliable"]])
    pooled = pd.concat(frames, ignore_index=True)
    pooled["success"] = pooled.success.astype(bool)
    pooled["reliable"] = pooled.reliable.fillna(False).astype(bool)
    rows = []
    for (exp, cond), g in pooled.groupby(["exp", "condition"], sort=False):
        tp = int((g.success & g.reliable).sum()); fp = int((~g.success & g.reliable).sum())
        fn = int((g.success & ~g.reliable).sum()); tn = int((~g.success & ~g.reliable).sum())
        rows.append({"experiment": exp, "condition": cond, "n": len(g), "success_reliable": tp, "failure_reliable": fp,
                     "success_unreliable": fn, "failure_unreliable": tn,
                     "precision": tp / (tp + fp) if tp + fp else float("nan"),
                     "failure_detection": tn / (tn + fp) if tn + fp else float("nan"),
                     "success_flagged_reliable": tp / (tp + fn) if tp + fn else float("nan")})
    table = pd.DataFrame(rows)
    total = table[["success_reliable", "failure_reliable", "success_unreliable", "failure_unreliable"]].sum()
    table.to_csv(OUT / "PR8_E7_by_condition.csv", index=False)

    rng = np.random.default_rng(20261024)
    pooled["cluster"] = pooled.exp + ":" + pooled.scene.astype(str)
    groups = {c: (g.success.values, g.reliable.values) for c, g in pooled.groupby("cluster")}
    keys = list(groups)
    prec, fdet, srel = [], [], []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        s = np.concatenate([groups[keys[i]][0] for i in pick]); r = np.concatenate([groups[keys[i]][1] for i in pick])
        prec.append((s & r).sum() / max(1, r.sum())); fdet.append((~s & ~r).sum() / max(1, (~s).sum())); srel.append((s & r).sum() / max(1, s.sum()))
    tp, fp, fn, tn = (int(total[c]) for c in ("success_reliable", "failure_reliable", "success_unreliable", "failure_unreliable"))
    summary = {"E7_pooled": {"n": int(len(pooled)), "clusters": len(keys), "success_reliable": tp, "failure_reliable": fp,
                             "success_unreliable": fn, "failure_unreliable": tn,
                             "precision": tp / (tp + fp), "precision_wilson": wilson(tp, tp + fp), "precision_cluster_bootstrap": np.percentile(prec, [2.5, 97.5]).tolist(),
                             "failure_detection": tn / (tn + fp), "failure_detection_wilson": wilson(tn, tn + fp), "failure_detection_cluster_bootstrap": np.percentile(fdet, [2.5, 97.5]).tolist(),
                             "success_flagged_reliable": tp / (tp + fn), "success_flagged_reliable_wilson": wilson(tp, tp + fn),
                             "success_flagged_reliable_cluster_bootstrap": np.percentile(srel, [2.5, 97.5]).tolist()}}

    e10 = pd.read_csv(RES / "E10.csv")
    e10 = e10[(e10.method == "M7") & (e10.condition.astype(str).str.zfill(6) != "020002")]
    succ = {sid: g.success.astype(bool).values for sid, g in e10.groupby("src_id")}
    sids = list(succ)
    boot = []
    for _ in range(B):
        pick = rng.integers(0, len(sids), len(sids))
        v = np.concatenate([succ[sids[i]] for i in pick]); boot.append(v.mean())
    k = int(e10.success.astype(bool).sum()); n = len(e10)
    summary["E10_M7"] = {"n": n, "successes": k, "rate": k / n, "wilson": wilson(k, n), "clusters_source_positions": len(sids),
                         "cluster_bootstrap": np.percentile(boot, [2.5, 97.5]).tolist(),
                         "by_source": {int(s): float(v.mean()) for s, v in succ.items()}}
    e1 = pd.read_csv(RES / "E1.csv")
    m1 = e1[e1.method == "M1"]
    fails = sorted(int(s) for s in m1[~m1.success.astype(bool)].scene)
    summary["PR2_E1_failures"] = {"indices": fails, "n": len(fails), "in_first_200": sum(f < 200 for f in fails),
                                  "first_200_successes": int(m1[m1.scene < 200].success.astype(bool).sum())}
    (OUT / "PR8_PR2_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
