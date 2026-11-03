# Experiments of the paper

English | [繁體中文](README.zh-TW.md)

These are the scripts behind the paper in [`../paper/`](../paper/): the simulation experiments E1–E9, the measured responses of E10, the analyses PR1–PR12 added after review, and the diagnosis of the measured failure (PR13, section 5.7 and appendix E of the paper). The per-scenario result tables are not in the repository; ask me if you need them.

## Contents

| Experiment | Script |
| --- | --- |
| Freeze of code, configurations, seeds and job lists | `freeze.py`, which wrote `freeze.json` |
| E1–E9, simulated rooms | `run_all.sh`, which calls `harness.py` |
| Pre-specified analysis of E1–E10 | `analyze.py` |
| E10 file metadata, responses and label check | `e10_sofa_metadata.py`, `e10_extract.py`, `e10_label_check.py`, `e10_label_diagnostics.py` |
| E10 run | `e10_run.py` |
| Peak-picking diagnosis of E10 | `e10_pick_diagnostics.py` |
| PR1, the method of Moore et al. (2013) | `post_review/moore2013.py` |
| PR2, PR3, PR6, PR7 | `post_review/rerun.py` |
| PR2, PR8 | `post_review/pr8_tables.py` |
| PR4, PR5, the E10 part of PR6 | `post_review/e10_post_review.py` |
| PR9 | `post_review/pr9_timing.py` |
| PR10 | `post_review/pr10_response_snr.py` |
| PR11 | `post_review/pr11_onset_by_source.py` |
| PR12 | `post_review/pr12_export_annotations.py`, `post_review/pr12_annotation_reference.py` |
| PR13, diagnosis of the measured failure | `diagnosis/d0_extract.py`, `d1_d2_timing.py`, `d3_frontends.py`, `d4_inversion.py`, with `common.py` |

`e10_inspect.py` lists the groups and datasets of an HDF5 file without reading any values; I wrote it to look at the v2.0 response file, which turned out to be unreadable (see below).

## Environment

The scripts that use the package run in the environment where echo-lattice is installed (`pip install -e .` at the repository root) and also need pandas. The ones that open HDF5 or SOFA files (`e10_inspect.py`, `e10_sofa_metadata.py`, `e10_extract.py`, `post_review/pr12_export_annotations.py`, `diagnosis/d0_extract.py`) need h5py instead, which echo-lattice does not depend on; any Python with h5py and NumPy will do. Results go to `results/` next to each script, and git ignores them.

## The freeze

`freeze.py` wrote `freeze.json` on 2026-10-23, before the first run, and refuses to write it again. It holds the master seed, the configurations, the number of jobs of every experiment with the hash of its job list, the E10 subset and its rules, the installed packages, and the commit and source hash of the package. The commit it names, `1948ea0`, is not in this repository's history; commit `ffe8311` holds the same code (git tree `b12d5c4`), so the hash of `src/` is the frozen one. `e10_run.py` refuses to run unless the hash of `src/` equals the frozen value.

`freeze.json` also keeps the SHA-256 of `harness.py`, of `freeze.py` and of the analysis plan as they were that day. The copies here find the repository from their own location instead of through a fixed path on my machine, and `freeze.py` takes the plan as an argument, so their hashes differ; nothing that computes a result changed. The plan itself is not in the repository. The `freeze.json` here is a copy of the freeze file: the paper explains the one line in which it differs and gives the SHA-256 of both.

## Simulations

```bash
cd experiments
bash run_all.sh                 # E1t on one worker, then E1–E9; PYTHON and WORKERS change the defaults
python analyze.py               # the pre-specified tables, from results/
python harness.py --experiment E1 --n 5 --out results/smoke   # a few scenes, as a smoke test
```

Scenes, responses and seeds are fixed, so a rerun in the same environment gives the same estimates. The exception is M6, the random multi-start baseline: it gets the wall-clock time that M1 took on the same response, so the number of starts it tries, and its result, depend on the machine and its load.

## Data for E10, PR4–PR6, PR11, PR12 and the diagnosis

The measured responses come from dEchorate (Di Carlo et al., 2021), which is not redistributed here.

- From Zenodo record [10.5281/zenodo.6576203](https://doi.org/10.5281/zenodo.6576203): `dEchorate_database.csv` and `main_geometry_from_echo_calibration.py`. Put both in `data/dechorate/` at the repository root, where `e10_run.py` looks for them (git ignores `data/`). The calibration script is read as text to get the calibrated room, microphones and loudspeakers; nothing runs it.
- From the SOFA release, <https://sofacoustics.org/data/database/dechorate/>: the 594 files `dEchorate_room<code>_src<1-9>_arr<1-6>_mics<a>-<b>.sofa`, about 1,000 MB, and for PR12 `dEchorate_annotations.h5`. Any folder will do; the scripts take it as an argument.

`freeze.json` names the record's response file, `dEchorate_rirs_gzip7.hdf5`. That file cannot be read: its HDF5 group metadata was never completely written, the same failure that dEchorate's GitHub issues #35 and #37 report. E10 therefore uses the SOFA release, in which the dataset's authors regenerated the responses, so they are not byte for byte the v2.0 responses. The Zenodo record also holds `dEchorate_echo_notes.pkl`; no script here loads it.

The SONICOM Ecosystem record of the SOFA release gives its licence as CC BY 4.0, while the licence attribute inside the files is the MIT licence; cite the dataset as Di Carlo et al. (2021) below.

## E10

```bash
cd experiments
python -I -X utf8 e10_sofa_metadata.py <sofa folder> results/E10_sofa_metadata.csv
python -I -X utf8 e10_extract.py <sofa folder> ../data/dechorate/dEchorate_database.csv freeze.json <e10.npz> results/E10_sofa_manifest.csv
python e10_label_check.py --rirs <e10.npz>
python e10_run.py --rirs <e10.npz> --label-check-decision "<where the decision is recorded>"
python e10_run.py --rirs <e10.npz> --speed 346.98 --tag _c346.98 --label-check-decision "<where the decision is recorded>"
```

The first two lines run with the h5py Python, the others in the echo-lattice environment. In my run the label check failed for the south, east and north walls, so `e10_run.py` needs `--label-check-decision`. I ran E10 as frozen and marked the results that depend on the south label as unverified; the paper reports the check and its diagnostics. The PR scripts that read E10 take the same npz with `--rirs`, and `pr12_annotation_reference.py` also takes the npz that `pr12_export_annotations.py` writes from `dEchorate_annotations.h5`.

## Diagnosis (PR13)

```bash
cd experiments/diagnosis
mkdir -p data
python -I -X utf8 d0_extract.py <sofa folder> data/dev_responses.npz
python d1_d2_timing.py dev-A
python d1_d2_timing.py dev-B
python d3_frontends.py
python d4_inversion.py --workers 14
```

`d0_extract.py` reads only the first and the fifth capsule of every array, for development and confirmation; the third capsule, which E10 used, is never decoded. The criteria were written down before any of these responses were read. Table E2 of the paper lists them, and appendix E says where the procedure departed from the plan.

## References

- D. Di Carlo, P. Tandeitnik, C. Foy, N. Bertin, A. Deleforge, and S. Gannot, "dEchorate: A calibrated room impulse response dataset for echo-aware signal processing," *EURASIP Journal on Audio, Speech, and Music Processing*, vol. 2021, Art. no. 39, 2021, doi: 10.1186/s13636-021-00229-0.
- A. H. Moore, M. Brookes, and P. A. Naylor, "Room geometry estimation from a single channel acoustic impulse response," in *Proc. 21st European Signal Processing Conference (EUSIPCO)*, 2013, pp. 1–5.
