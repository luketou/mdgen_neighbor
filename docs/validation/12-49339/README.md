# L40S 量測：#12 Ticket 5 Halo 近似誤差（2026-10-23）

以 `python -m mdgen.full_length.halo_error`（方法見 `docs/validation/halo-error.md`）在官方 ATLAS checkpoint 上，比較 Training region 加 `--halo_rings` 1／2 圈 halo 與完整蛋白 forward 在 loss core 上的輸出差（同一雜訊、t 與 loss group）。

結論（只陳述量到的範圍）：

- **1 圈 halo**：第 0 圈（loss group 的 ghost 直接由 region 外的 group 擁有）的相對誤差中位數 2.2–2.8%，最大 9.9–15.8%；較內側的第 1 圈中位數 0.5–0.8%、最大 ≤ 2.14%。
- **2 圈 halo**（只有 3ilw_A 量得到）：第 0 圈中位數降到 0.39–0.41%、最大 2.1–2.2%，約為 1 圈的 1/6 與 1/5。
- 代價：3ilw_A 同一組 54 個 loss 殘基，1 圈 region 共 115 殘基，2 圈為 250 殘基（2.2 倍）。在固定 `--region_residues` 下，2 圈代表每步的 loss 殘基約少一半。
- 判讀：1 圈的典型誤差約 2%，但邊界 loss group 有個別殘基達 10–16%；若要把邊界最大誤差壓在約 2%，需要 2 圈。是否加深取決於能否接受每步 loss 殘基減半，本紀錄不替這個取捨下結論。

## 執行環境

| 項目 | 值 |
|---|---|
| Slurm job | 49339、49341、49343（3ilw_A，`--halo_rings 1 2`）；49340、49342、49344（6irx_A，`--halo_rings 1`）。分區 `l40s`，node-201 |
| 程式版本 | `fix/halo-error-skip-unfit-seeds` @ `a9e4fe5`（main `5b492c1` + 跳過放不下預算的 seed） |
| 快照 | `/share/home/r15543056/MDgen/hpc_runs/l40s-20261023T073430Z-a9e4fe5/src`（`git archive`、SHA-256 驗證、唯讀） |
| 軟體 | Python 3.11.12、torch 2.11.0+cu128、CUDA 12.8 |
| Checkpoint | `/share/home/r15543056/MDgen/mdgen/ckpt/atlas.ckpt`（HF `bjing-mit/mdgen` @ `a92efe5`，SHA-256 `bc90c66e…`），`--local_spatial`、5 層、`cell_summaries` 關閉、`num_frames` 250、`frame_interval` 無 |
| 分區參數 | `spatial_cell_width` 16、`spatial_contact_radius` 16、`core_capacity` 128、`ghost_capacity` 128（checkpoint 預設） |
| 資料 | `/share/home/r15543056/MDgen/mdgen/data/atlas_data/<name>_R<r>_i40.npy`（251 幀）、split `splits/atlas.csv` |
| 參數 | `--region_residues 320 --regions 8 --times 0.25 0.5 0.75 --seed 0` |

提交指令（在 `<run>/src`，`MDGEN_VENV=/share/home/r15543056/.venvs/mdgen`，另設 `MDGEN_ATLAS_CKPT`、`MDGEN_HALO_DATA`、`MDGEN_HALO_SPLIT` 與 `MODEL_DIR=<run>/workdir_<tag>`）：

```bash
sbatch --chdir=<run>/src job/submit_halo_error.sh --pdb_id 3ilw_A --replicate R --suffix _i40 \
  --region_residues 320 --regions 8 --times 0.25 0.5 0.75 --seed 0
sbatch --chdir=<run>/src job/submit_halo_error.sh --pdb_id 6irx_A --replicate R --suffix _i40 \
  --region_residues 320 --regions 8 --times 0.25 0.5 0.75 --seed 0 --halo_rings 1
```

## 蛋白與 region 的選擇

- 用 `atlas_data` 中最長的兩個蛋白：6irx_A（496 殘基、46 group）與 3ilw_A（470 殘基、48 group）。其餘 18 個 ≤ 301 殘基，2 圈 halo 幾乎涵蓋整個蛋白，沒有 region 外的殘基可比。
- 2 圈 halo 太大：3ilw_A 的 8 個 seed 中 7 個放不進 320（種子 group 加 2 圈需 402–470 殘基），每個 replicate 只剩 seed 0 的一個 region；6irx_A 沒有任何 seed 放得下 2 圈（job 49335–49337），因此只量 1 圈。
- 6irx_A 用 1 圈時 8 個 seed 中 2 個（seed 1、3）放不下，其餘 6 個得到 5 個不同的 region（兩個 seed 長出相同的 loss group）。
- 三個 replicate 的第 0 幀完全相同（R1–R3 的 Cα 最大差 0.0 Å，第 125 幀差 10.8–11.9 Å），所以分區與 region 相同，差別只在軌跡內容。三個 replicate 不是三個獨立的 region 樣本。
- 跳過的 seed 多是需要較大 halo 的 group，結果只代表放得下的 region（見 `skipped_seeds`）。

## 結果（相對誤差 = 逐殘基 L2 ÷ 完整蛋白輸出 L2；3 個 t 合併）

### 3ilw_A（1 個 region：7 個 loss group、54 個 loss 殘基；1 圈 region 115 殘基，2 圈 250 殘基）

| Job | Replicate | k | 第 0 圈 中位數／最大（n=135） | 第 1 圈 中位數／最大（n=27） | 全部 中位數／最大 |
|---|---|---|---|---|---|
| 49339 | R1 | 1 | 2.33%／10.8% | 0.73%／2.14% | 2.17%／10.8% |
| 49339 | R1 | 2 | 0.40%／2.07% | 0.11%／0.63% | 0.38%／2.07% |
| 49341 | R2 | 1 | 2.17%／11.0% | 0.65%／2.03% | 2.03%／11.0% |
| 49341 | R2 | 2 | 0.39%／2.08% | 0.10%／0.50% | 0.37%／2.08% |
| 49343 | R3 | 1 | 2.35%／9.88% | 0.76%／2.04% | 2.21%／9.88% |
| 49343 | R3 | 2 | 0.41%／2.22% | 0.11%／0.58% | 0.38%／2.22% |

### 6irx_A（5 個 region、共 111 個 loss 殘基；只有 `--halo_rings 1`）

| Job | Replicate | 第 0 圈 中位數／最大（n=315） | 第 1 圈 中位數／最大（n=18） | 全部 中位數／最大（n=333） |
|---|---|---|---|---|
| 49340 | R1 | 2.83%／15.8% | 0.67%／1.27% | 2.60%／15.8% |
| 49342 | R2 | 2.48%／13.7% | 0.53%／1.07% | 2.14%／13.7% |
| 49344 | R3 | 2.39%／13.5% | 0.47%／0.92% | 2.17%／13.5% |

所有 job 的相對誤差 `nonfinite` 皆為 0。逐 t、逐 region 的數值見各 JSON 的 `proteins[0].regions[].by_halo.<k>.times`。

## 原始資料

| 檔案 | 內容 |
|---|---|
| `halo_error_<job>.json` | 完整報告（遠端 `<run>/results/halo_error_<job>/halo_error.json`） |
| `halo_error_<job>.out`／`.err` | log；`.out` 末尾為各 k 與各圈的摘要 |
| `smoke_halo_error_49317.*` | 程式修正前的 smoke（main `5b492c1`，3ilw_A R1、`--region_residues 256 --regions 1 --times 0.5`），與正式結果同量級 |
| `pytest_halo_49330.out` | 快照上的 `tests/test_halo_error.py tests/test_group_region.py`：23 passed、1 failed（見下） |
| `job_history`、`meta.json` | 本快照的提交紀錄與快照來源 |

## 沒有產生結果的 job

| Job | 原因 |
|---|---|
| 49323 | main `5b492c1`，4 蛋白、預算 256：seed group 37 加 2 圈需 435 殘基，整個 job 中止 |
| 49324–49329 | main `5b492c1`，預算 320：同上（3ilw_A 435、6irx_A 483 殘基）。促成 `a9e4fe5` 的跳過修正 |
| 49330 | pytest：`test_with_halo_keeps_the_loss_groups` 失敗，因 HPC Python 的 `/opt/python-3.11.12-gpu/.../site-packages/tests` 遮蔽 repo 的 `tests` namespace package；與本次修正無關。其餘 23 項通過 |
| 49331–49333 | 依賴 49330 `afterok`，取消 |
| 49335–49337 | 6irx_A 沒有 seed 放得下 2 圈 halo，整個 job 報錯、未寫報告 |

## 未涵蓋

- 只有兩個約 500 殘基的蛋白；2 圈的比較只有 3ilw_A 的一個 region。長蛋白（如 Piezo1）上 region 外的殘基多得多，誤差分布可能不同，需另外量。
- `_i40` 資料的時間間隔是否與 checkpoint 訓練資料相同，本次沒有查證。
- 只比較單次 forward 的輸出，沒有量對 loss 或訓練結果的影響。
