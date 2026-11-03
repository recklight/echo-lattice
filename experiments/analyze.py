"""Pre-specified analysis of the evaluation runs E1-E10.

    <project>/.venv/Scripts/python.exe analyze.py [--results results] [--out results/analysis]

Reads results/<experiment>.csv as harness.py writes them and writes one CSV per
table plus summary.md. Proportions get Wilson 95% intervals, medians a
percentile bootstrap (10,000 resamples), and the paired tests are exact
McNemar: M1 against M2 in E1 and E2, and M1 against each ablation of E6 with
Holm's correction over the six. The sweeps (E3, E4, E5, E9) are reported as
estimates with intervals only. Primary results: P1, the success rate of M1 in
E1, and P2, the precision and failure detection of the reliable flag in E1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from harness import MASTER_SEED, SUCCESS_M

Z95 = 1.959963984540054
N_BOOT = 10_000
THRESHOLDS_M = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
LATTICE_METHODS = {"M1", "M2", "M6", "M7"}


# ------------------------------------------------------------------- statistics
def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Proportion with its Wilson score 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / (1 + Z95**2 / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), b + c, 0.5)))


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def bootstrap_median(x, key: str) -> tuple[float, float, float]:
    """Median with a percentile bootstrap 95% interval; failures enter as +inf."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng([MASTER_SEED, sum(map(ord, key))])
    medians = np.median(x[rng.integers(0, x.size, (N_BOOT, x.size))], axis=1)
    # inverted_cdf picks resampled medians themselves, so +inf never meets interpolation
    lo, hi = np.quantile(medians, [0.025, 0.975], method="inverted_cdf")
    return float(np.median(x)), float(lo), float(hi)


# ---------------------------------------------------------------------- loading
def sides(text) -> np.ndarray | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return np.sort(np.array([float(v) for v in text.split(" x ")]))[::-1]


def load(results: Path, name: str) -> pd.DataFrame | None:
    path = results / f"{name}.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path)
    t["success"] = t["success"].map(lambda v: str(v) == "True")
    if "reliable" in t:
        t["reliable"] = t["reliable"].map(lambda v: None if pd.isna(v) else str(v) == "True")
    t["room_error_m"] = pd.to_numeric(t["room_error_m"], errors="coerce").fillna(np.inf)
    return t


def problems(t: pd.DataFrame) -> dict:
    """Rows that are not method results: a failed measurement or a crashed job."""
    return {"measurement_failures": int((t["method"] == "measurement").sum()) if "method" in t else 0,
            "job_failures": int(t["method"].isna().sum()) if "method" in t else len(t)}


# ----------------------------------------------------------------------- tables
def outcome_table(t: pd.DataFrame, by: list[str], key: str) -> pd.DataFrame:
    """Success rate, error distribution and per-side errors per group."""
    rows = []
    t = t[t["method"].notna() & (t["method"] != "measurement")]
    for group, g in t.groupby(by, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        k, n = int(g["success"].sum()), len(g)
        p, lo, hi = wilson(k, n)
        med, mlo, mhi = bootstrap_median(g["room_error_m"], f"{key}{group}")
        row = dict(zip(by, group))
        row.update(n=n, successes=k, success_rate=p, success_lo=lo, success_hi=hi,
                   median_error_m=med, median_error_lo=mlo, median_error_hi=mhi,
                   errors=int(g["error"].notna().sum()) if "error" in g else 0)
        for thr in THRESHOLDS_M:
            row[f"ecdf_{round(100 * thr)}cm"] = float((g["room_error_m"] < thr).mean())
        for wide in ("success_10cm", "success_15cm"):  # E10's secondary thresholds
            if wide in g:
                p, lo, hi = wilson(int(g[wide].map(lambda v: str(v) == "True").sum()), n)
                row.update({wide: p, f"{wide}_lo": lo, f"{wide}_hi": hi})
        diffs = [s_est - s_true for s_est, s_true in
                 ((sides(e), sides(tr)) for e, tr in zip(g["est_sides"], g["true_sides"]))
                 if s_est is not None and s_true is not None and s_est.size == s_true.size]
        if diffs:
            d = np.abs(np.vstack(diffs))
            for i in range(d.shape[1]):
                row[f"side{i + 1}_median_abs_m"] = float(np.median(d[:, i]))
                row[f"side{i + 1}_rmse_m"] = float(np.sqrt(np.mean(d[:, i] ** 2)))
        if g["method"].iloc[0] in LATTICE_METHODS or str(g["method"].iloc[0]).startswith("A"):
            for col, label in (("latency_error_ms", "latency_error_ms"), ("bearing_error_deg", "bearing_error_deg")):
                if col in g:
                    m, a, b = bootstrap_median(pd.to_numeric(g[col], errors="coerce").fillna(np.inf),
                                               f"{key}{group}{col}")
                    row.update({f"median_{label}": m, f"median_{label}_lo": a, f"median_{label}_hi": b})
        rows.append(row)
    return pd.DataFrame(rows)


def flag_table(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Precision, failure detection and false alarms of the reliable flag."""
    rows = []
    t = t[t["reliable"].notna()]
    for group, g in t.groupby(by, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        ok, rel = g["success"].astype(bool), g["reliable"].astype(bool)
        row = dict(zip(by, group))
        row["n"] = len(g)
        for name, k, n in (("precision", (ok & rel).sum(), rel.sum()),
                           ("failure_detection", (~ok & ~rel).sum(), (~ok).sum()),
                           ("false_alarm", (ok & ~rel).sum(), ok.sum())):
            p, lo, hi = wilson(int(k), int(n))
            row.update({name: p, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_k": int(k), f"{name}_n": int(n)})
        row["reliable_share"] = float(rel.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> dict:
    """Exact McNemar on success over the scenes both tables share."""
    m = a[["scene", "success"]].merge(b[["scene", "success"]], on="scene", suffixes=("_a", "_b"))
    only_a = int((m["success_a"] & ~m["success_b"]).sum())
    only_b = int((~m["success_a"] & m["success_b"]).sum())
    return {"a": label_a, "b": label_b, "n_pairs": len(m),
            "rate_a": float(m["success_a"].mean()), "rate_b": float(m["success_b"].mean()),
            "only_a": only_a, "only_b": only_b, "p_exact": mcnemar_exact(only_a, only_b)}


def calibration_table(t: pd.DataFrame) -> pd.DataFrame:
    """Laplace standard errors of the successful M1 fits: z per side and 95% coverage."""
    g = t[(t["method"] == "M1") & t["success"]]
    z = [np.array([float(v) for v in s.split()]) for s in g["side_z"] if isinstance(s, str)]
    if not z:
        return pd.DataFrame()
    z = np.vstack(z)
    rows = []
    for i in range(z.shape[1]):
        col = z[:, i][np.isfinite(z[:, i])]
        k = int((np.abs(col) <= Z95).sum())
        p, lo, hi = wilson(k, col.size)
        rows.append({"side": i + 1, "n": col.size, "mean_z": float(col.mean()), "sd_z": float(col.std(ddof=1)),
                     "q05": float(np.quantile(col, 0.05)), "q50": float(np.median(col)),
                     "q95": float(np.quantile(col, 0.95)), "coverage_95": p, "coverage_lo": lo, "coverage_hi": hi})
    return pd.DataFrame(rows)


def timing_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in t.groupby("method"):
        s = pd.to_numeric(g["time_s"], errors="coerce").dropna()
        rows.append({"method": method, "n": s.size, "median_s": float(s.median()), "q25_s": float(s.quantile(0.25)),
                     "q75_s": float(s.quantile(0.75)), "mean_s": float(s.mean()), "max_s": float(s.max())})
    return pd.DataFrame(rows)


def speed_table(t: pd.DataFrame) -> pd.DataFrame:
    """E5 R3: relative size error against the speed-of-sound ratio."""
    from echo_lattice.image_source import speed_of_sound

    rows = []
    for cond, temp in (("R3_speed_15C", 15.0), ("R3_speed_30C", 30.0)):
        g = t[(t["condition"] == cond) & (t["method"] == "M1")]
        if g.empty:
            continue
        total = np.array([sides(s).sum() for s in g["true_sides"]])
        rel = g["room_error_m"].to_numpy() / total
        med, lo, hi = bootstrap_median(rel, f"speed{cond}")
        rows.append({"condition": cond, "speed_ratio_minus_1": abs(speed_of_sound(temp) / 343.0 - 1.0),
                     "median_relative_error": med, "median_relative_error_lo": lo,
                     "median_relative_error_hi": hi, "n": len(g)})
    return pd.DataFrame(rows)


def details_table(t: pd.DataFrame) -> pd.DataFrame:
    """M6 starts within the time budget and the share of picked echoes M7 could label."""
    rows = []
    for method, col in (("M6", "tries"), ("M7", "labelled_echoes")):
        g = t[(t["method"] == method)]
        if g.empty or col not in g:
            continue
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        row = {"method": method, "n": v.size, f"median_{col}": float(v.median()),
               f"min_{col}": float(v.min()), f"max_{col}": float(v.max())}
        if method == "M7":
            share = v / pd.to_numeric(g.loc[v.index, "picked_echoes"], errors="coerce")
            row["median_labelled_share"] = float(share.median())
        rows.append(row)
    return pd.DataFrame(rows)


def hits_table(t: pd.DataFrame) -> pd.DataFrame:
    """M3-M5: how many of the nine true first-order arrivals were among the picks."""
    if "first_order_hits" not in t:
        return pd.DataFrame()
    g = t[t["method"].isin(["M3", "M4", "M5"]) & t["first_order_hits"].notna()]
    rows = []
    for method, h in g.groupby("method"):
        v = h["first_order_hits"].astype(int)
        rows.append({"method": method, "n": len(v), "mean_hits": float(v.mean()), "all_nine": float((v == 9).mean()),
                     "at_most_six": float((v <= 6).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- main
def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)\n"
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    head = "| " + " | ".join(map(str, out.columns)) + " |"
    rule = "|" + "---|" * len(out.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in out.itertuples(index=False)]
    return "\n".join([head, rule, *body]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    notes = {}
    data = {name: load(args.results, name) for name in ("E1", "E1t", "E1b", "E2", "E3", "E4", "E5", "E6", "E9")}
    for name, t in data.items():
        if t is not None:
            notes[name] = {**problems(t), "meta": json.loads((args.results / f"{name}.meta.json").read_text())["code"]}

    if (t := data["E1"]) is not None:
        tables["E1_outcomes"] = outcome_table(t, ["method"], "E1")
        tables["E1_flag_P2"] = flag_table(t[t["method"] == "M1"], ["method"])
        tables["E1_flag_all_lattice_methods"] = flag_table(t[t["method"].isin(LATTICE_METHODS)], ["method"])
        tables["E1_first_order_hits"] = hits_table(t)
        tables["E1_M6_M7_details"] = details_table(t)
        tables["E8_calibration"] = calibration_table(t)
        tests = [paired_test(t[t["method"] == "M1"], t[t["method"] == "M2"], "E1 M1", "E1 M2")]
        if (t2 := data["E2"]) is not None:
            tests.append(paired_test(t2[t2["method"] == "M1"], t2[t2["method"] == "M2"], "E2 M1", "E2 M2"))
        tables["paired_M1_vs_M2"] = pd.DataFrame(tests)
        if (t6 := data["E6"]) is not None:
            reference = t[(t["method"] == "M1") & (t["scene"] < 200)]
            family = [paired_test(reference, t6[t6["method"] == a], "E1 M1 (scenes 0-199)", a)
                      for a in sorted(t6["method"].dropna().unique())]
            adjusted = holm([f["p_exact"] for f in family])
            for f, p in zip(family, adjusted):
                f["p_holm"] = p
            tables["E6_ablations_paired"] = pd.DataFrame(family)
            tables["E6_outcomes"] = outcome_table(t6, ["method"], "E6")
    if (t := data["E1t"]) is not None:
        tables["E1t_timing"] = timing_table(t)
    if (t := data["E1b"]) is not None:
        tables["E1b_outcomes"] = outcome_table(t, ["method"], "E1b")
        tables["E1b_first_order_hits"] = hits_table(t)
    if (t := data["E2"]) is not None:
        tables["E2_outcomes"] = outcome_table(t, ["method"], "E2")
        tables["E2_flag"] = flag_table(t[t["method"].isin(LATTICE_METHODS)], ["method"])
        tables["E2_calibration"] = calibration_table(t)
        tables["E2_M6_M7_details"] = details_table(t)
    for name in ("E3", "E4", "E5", "E9"):
        if (t := data[name]) is not None:
            tables[f"{name}_outcomes"] = outcome_table(t, ["condition", "method"], name)
            tables[f"{name}_flag"] = flag_table(t[t["method"].isin(LATTICE_METHODS)], ["condition", "method"])
    if (t := data["E5"]) is not None:
        tables["E5_R3_speed"] = speed_table(t)
    # E10 at the frozen speed of sound, and the secondary run at the dataset paper's 346.98 m/s
    for stem, prefix in (("E10", "E10"), ("E10_c346.98", "E10s")):
        if (t := load(args.results, stem)) is None:
            continue
        notes[stem] = {**problems(t), "meta": json.loads((args.results / f"{stem}.meta.json").read_text())}
        t = t[t["method"].notna()].copy()
        t["room_code"] = t["room_code"].astype(str).str.zfill(6)
        t["subset"] = np.where(t["room_code"] == "020002", "secondary_furniture", "primary_empty")
        primary = t[t["subset"] == "primary_empty"]
        tables[f"{prefix}_outcomes"] = outcome_table(t, ["subset", "method"], prefix)
        tables[f"{prefix}_flag"] = flag_table(t, ["subset", "method"])
        tables[f"{prefix}_by_room_code"] = outcome_table(primary, ["room_code", "method"], f"{prefix}code")
        tables[f"{prefix}_by_reflective_surfaces"] = outcome_table(primary, ["n_reflective", "method"], f"{prefix}refl")
        tables[f"{prefix}_by_source_type"] = outcome_table(primary, ["source_type", "method"], f"{prefix}src")
        tables[f"{prefix}_flag_by_reflective_surfaces"] = flag_table(primary, ["n_reflective", "method"])
        # the label check could not confirm the south wall (e10_data_note.md); mark what relies on it
        by_code = tables[f"{prefix}_by_room_code"]
        by_code["label_note"] = by_code["room_code"].map(
            lambda c: "south label unverified" if c == "000100" else ("south digit unverified" if c[3] == "1" else ""))
        for name in (f"{prefix}_by_reflective_surfaces", f"{prefix}_flag_by_reflective_surfaces"):
            tables[name]["label_note"] = tables[name]["n_reflective"].map(
                lambda n: "count relies on the south label" if int(n) in (1, 3, 4, 5) else "")
        calibrated = primary[primary["src_id"] <= 5]  # sources 6-8 have uncalibrated positions
        tables[f"{prefix}_calibrated_sources_outcomes"] = outcome_table(calibrated, ["method"], f"{prefix}cal")
        tables[f"{prefix}_M7_details"] = details_table(t)
    pooled = [data[n][data[n]["method"] == "M1"].assign(source=n) for n in ("E1", "E3", "E5") if data[n] is not None]
    if pooled:
        tables["E7_flag_pooled"] = flag_table(pd.concat(pooled), ["source"])
        tables["E7_flag_pooled_all"] = flag_table(pd.concat(pooled).assign(pool="E1+E3+E5"), ["pool"])

    lines = ["# Evaluation results (pre-specified analysis)", "",
             f"Success: sorted-side error below {100 * SUCCESS_M:.0f} cm. Intervals: Wilson 95% for proportions, "
             f"percentile bootstrap ({N_BOOT} resamples) for medians.", "",
             "## Runs", "", "```", json.dumps(notes, indent=1), "```", ""]
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        lines += [f"## {name}", "", to_markdown(df), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(tables)} tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
