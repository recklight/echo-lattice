# 論文的實驗

[English](README.md) | 繁體中文

這裡是 [`../paper/`](../paper/) 那篇論文背後的腳本：模擬實驗 E1–E9、量測響應的 E10、審查後新增的分析 PR1–PR12，以及量測失敗的診斷（PR13，論文第 5.7 節與附錄 E）。逐場景的結果表沒有放進 repository，需要的話可以向我索取。

## 內容

| 實驗 | 腳本 |
| --- | --- |
| 凍結程式碼、設定、種子與工作清單 | `freeze.py`，寫出 `freeze.json` |
| E1–E9，模擬房間 | `run_all.sh`，呼叫 `harness.py` |
| E1–E10 事先寫定的分析 | `analyze.py` |
| E10 的檔案中繼資料、響應與標記檢查 | `e10_sofa_metadata.py`、`e10_extract.py`、`e10_label_check.py`、`e10_label_diagnostics.py` |
| E10 執行 | `e10_run.py` |
| E10 的撿峰診斷 | `e10_pick_diagnostics.py` |
| PR1，Moore 等人（2013）的方法 | `post_review/moore2013.py` |
| PR2、PR3、PR6、PR7 | `post_review/rerun.py` |
| PR2、PR8 | `post_review/pr8_tables.py` |
| PR4、PR5、PR6 的 E10 部分 | `post_review/e10_post_review.py` |
| PR9 | `post_review/pr9_timing.py` |
| PR10 | `post_review/pr10_response_snr.py` |
| PR11 | `post_review/pr11_onset_by_source.py` |
| PR12 | `post_review/pr12_export_annotations.py`、`post_review/pr12_annotation_reference.py` |
| PR13，量測失敗的診斷 | `diagnosis/d0_extract.py`、`d1_d2_timing.py`、`d3_frontends.py`、`d4_inversion.py`，共用 `common.py` |

`e10_inspect.py` 列出 HDF5 檔的群組與資料集，不讀任何數值；我寫它是為了檢查 v2.0 的響應檔，結果那個檔讀不出來（見下文）。

## 執行環境

用到套件的腳本在裝了 echo-lattice 的環境執行（在 repository 根目錄 `pip install -e .`），另外需要 pandas。開啟 HDF5 或 SOFA 檔的腳本（`e10_inspect.py`、`e10_sofa_metadata.py`、`e10_extract.py`、`post_review/pr12_export_annotations.py`、`diagnosis/d0_extract.py`）需要的是 h5py，echo-lattice 不依賴它，任何裝了 h5py 與 NumPy 的 Python 都可以。結果寫到各腳本旁邊的 `results/`，git 不追蹤。

## 凍結檔

`freeze.py` 在 2026-10-23 第一次執行之前寫出 `freeze.json`，之後不會再寫。裡面有主種子、各組設定、每個實驗的工作數與工作清單的雜湊、E10 的子集與規則、安裝的套件，以及套件的 commit 與原始碼雜湊。它記錄的 commit `1948ea0` 不在這個 repository 的歷史裡；commit `ffe8311` 是同一份程式碼（git tree `b12d5c4`），所以 `src/` 的雜湊就是凍結的值。`e10_run.py` 只在 `src/` 的雜湊等於凍結值時才執行。

`freeze.json` 也記錄了當天 `harness.py`、`freeze.py` 與分析計畫的 SHA-256。這裡的副本從自己的位置找到 repository，不用我電腦上的固定路徑，`freeze.py` 也以參數接收計畫檔，所以雜湊不同；計算結果的程式都沒有變。計畫本身不在 repository 裡。這裡的 `freeze.json` 是凍結檔的副本，論文說明了它不同的那一行，並列出兩者的 SHA-256。

## 模擬

```bash
cd experiments
bash run_all.sh                 # 先以一個 worker 跑 E1t，再跑 E1–E9；PYTHON 與 WORKERS 可改預設值
python analyze.py               # 事先寫定的表格，讀 results/
python harness.py --experiment E1 --n 5 --out results/smoke   # 只跑幾個場景，快速檢查
```

場景、響應與種子都是固定的，在同一個環境重跑會得到相同的估計。例外是隨機起點的 M6：它用的是 M1 在同一筆響應上花的牆鐘時間，所以它試的起點數與結果，會隨機器與負載而不同。

## E10、PR4–PR6、PR11、PR12 與診斷所需的資料

量測響應取自 dEchorate（Di Carlo 等人，2021），這裡不重新散布。

- 從 Zenodo 紀錄 [10.5281/zenodo.6576203](https://doi.org/10.5281/zenodo.6576203) 下載 `dEchorate_database.csv` 與 `main_geometry_from_echo_calibration.py`，兩個都放到 repository 根目錄的 `data/dechorate/`，`e10_run.py` 會在那裡找（git 不追蹤 `data/`）。校準腳本只當作文字讀，從中取出校準後的房間、麥克風與喇叭位置，不會執行它。
- 從 SOFA 版本 <https://sofacoustics.org/data/database/dechorate/> 下載 594 個 `dEchorate_room<code>_src<1-9>_arr<1-6>_mics<a>-<b>.sofa`，約 1,000 MB；PR12 另外需要 `dEchorate_annotations.h5`。放在哪個資料夾都可以，腳本以參數接收。

`freeze.json` 記錄的是這筆 Zenodo 紀錄的響應檔 `dEchorate_rirs_gzip7.hdf5`。這個檔讀不出來：它的 HDF5 群組中繼資料沒有完整寫入，dEchorate 在 GitHub 的 issue #35 與 #37 回報的也是同一個問題。所以 E10 用的是 SOFA 版本，也就是資料集作者重新產生的響應，和 v2.0 的響應並非逐位元組相同。Zenodo 紀錄裡還有 `dEchorate_echo_notes.pkl`，這裡沒有任何腳本會載入它。

SONICOM Ecosystem 上 SOFA 版本的紀錄標示授權為 CC BY 4.0，檔案內的授權屬性則是 MIT 授權；引用時請用下方的 Di Carlo 等人（2021）。

## E10

```bash
cd experiments
python -I -X utf8 e10_sofa_metadata.py <sofa folder> results/E10_sofa_metadata.csv
python -I -X utf8 e10_extract.py <sofa folder> ../data/dechorate/dEchorate_database.csv freeze.json <e10.npz> results/E10_sofa_manifest.csv
python e10_label_check.py --rirs <e10.npz>
python e10_run.py --rirs <e10.npz> --label-check-decision "<where the decision is recorded>"
python e10_run.py --rirs <e10.npz> --speed 346.98 --tag _c346.98 --label-check-decision "<where the decision is recorded>"
```

前兩行用裝了 h5py 的 Python，其餘在 echo-lattice 的環境執行。我執行時，標記檢查在南、東、北三面牆沒有通過，所以 `e10_run.py` 需要 `--label-check-decision`。我照凍結的設定跑了 E10，並把依賴南牆標記的結果標為未查證；論文報告了這項檢查與它的診斷。讀 E10 響應的 PR 腳本以 `--rirs` 接收同一個 npz，`pr12_annotation_reference.py` 另外接收 `pr12_export_annotations.py` 從 `dEchorate_annotations.h5` 寫出的 npz。

## 診斷（PR13）

```bash
cd experiments/diagnosis
mkdir -p data
python -I -X utf8 d0_extract.py <sofa folder> data/dev_responses.npz
python d1_d2_timing.py dev-A
python d1_d2_timing.py dev-B
python d3_frontends.py
python d4_inversion.py --workers 14
```

`d0_extract.py` 只讀每個陣列的第 1 支與第 5 支麥克風，分別作開發與確認；E10 用的第 3 支從未解碼。判斷標準在讀取任何一筆這些響應之前寫定，論文的表 E2 列出了它們，附錄 E 說明實際做法和計畫不同的地方。

## References

- D. Di Carlo, P. Tandeitnik, C. Foy, N. Bertin, A. Deleforge, and S. Gannot, "dEchorate: A calibrated room impulse response dataset for echo-aware signal processing," *EURASIP Journal on Audio, Speech, and Music Processing*, vol. 2021, Art. no. 39, 2021, doi: 10.1186/s13636-021-00229-0.
- A. H. Moore, M. Brookes, and P. A. Naylor, "Room geometry estimation from a single channel acoustic impulse response," in *Proc. 21st European Signal Processing Conference (EUSIPCO)*, 2013, pp. 1–5.
