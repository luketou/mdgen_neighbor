# Halo 近似誤差實驗（#12 Ticket 5）

`python -m mdgen.full_length.halo_error` 量測 Training region 的 halo 近似：同一個蛋白先以完整蛋白第 0 幀分區跑一次 forward，再對同一組 loss group 分別加 `--halo_rings` 圈 halo 跑 region forward，兩者共用雜訊與 t，比較 loss core 在未條件化幀上的模型輸出。

```bash
python -m mdgen.full_length.halo_error --sim_ckpt atlas.ckpt --data_dir D --split S \
  --pdb_id NAME --replicate 1 --output report.json \
  [--partition_dir P] [--region_residues 1024] [--halo_rings 1 2] [--regions 4] \
  [--times 0.5] [--seed 0] [--spatial_cell_width ... 其他 local 覆寫]
```

L40S 上用 `job/submit_halo_error.sh`（見 `job/README.md`）。實測結果與 JSON 放在 `docs/validation/12-<job>/`，結論附 Slurm job ID。

## 方法

- loss group 由 `sample_region_groups` 以最深的 `--halo_rings` 選出，因此每種設定都放得進 `--region_residues`；較淺的設定只是 halo 較少，loss group 相同。種子 group 加上最深 halo 就超過預算的 seed 會跳過並記在 `skipped_seeds`；同一蛋白沒有任何 seed 放得下時報錯。跳過的 seed 偏向 halo 較大的 group，結果只代表放得下的 region。
- 每個 region 以 `--seed + region 編號` 選種子 group，選出相同 loss group 的 region 只算一次；模型輸入沿用 checkpoint 的 `frame_interval`、取第 0 幀起的視窗；雜訊以 `--seed` 產生，每個 `--times` 值一份，完整蛋白與所有 region 共用。結果與蛋白在 split 中的順序無關。
- 只比較 loss core 殘基、`x_cond_mask` 為 0 的幀，且只計 loss mask 監督的通道（未定義的 torsion 不計）。逐殘基 L2 = 差值在幀與通道上的 L2；相對誤差 = 該 L2 ÷ 完整蛋白輸出的 L2。
- 邊界圈數：loss group 的 ghost 若由 loss group 以外的 group 擁有，為第 0 圈；ghost 由第 r − 1 圈擁有者為第 r 圈。沒有 ghost 擁有者路徑通到 loss group 以外時（例如整個蛋白都是 loss group）為 `None`。
- 蛋白須讓完整蛋白 forward 裝得下 GPU。

## 報告欄位

- `summary.<k>`：`--halo_rings k` 下所有蛋白、region、t 的 loss 殘基彙整，`l2`／`relative` 各有 `count`、`median`、`max`（只計有限值；完整蛋白輸出為 0 而無法算相對誤差的殘基計入 `nonfinite`）；`by_ring.<r>` 依邊界圈數分組。
- `proteins[]`：`name`、`residues`、`groups`、`summary`、`skipped_seeds`（`seed`、`reason`），以及 `regions[]`：`loss_groups`、`by_halo.<k>`（`halo_groups`、`region_residues`、`loss_residues`、每個 t 的 `l2`／`relative`）。
- `settings`（CLI 參數與模型的 local 參數）、`environment`。

## 判讀

原實驗只作描述性報告；新工作若需要硬門檻，須先確認自己的比較契約，不追溯更改舊結論。ATLAS checkpoint 的數值只能在 GPU 上量測。

## 容量先導與完整報告

`--num_frames` 明確覆寫 checkpoint 的視窗長度；`--warmup`／`--repeats` 控制同輸入 forward 的同步時間與 GPU peak allocated／reserved 量測。在一般成對誤差模式中，region 的記憶體包含仍存活的完整蛋白 reference，不能當成獨立 region forward 的記憶體或直接宣稱省量。

`--inventory_only` 不實例化模型，只在 CPU Compute Job 讀 checkpoint 組態、分區與第零幀座標，列出每個 halo 設定全部種子 group 的最小 region 大小、分位數及不可行原因。

`--capacity_only` 分別量測並釋放完整蛋白與 region 輸出；完整蛋白 OOM 時仍繼續 region 路徑。此模式只報容量、時間與有限性，不產生誤差分布，也不保留 reference；一般比較模式則需要完整蛋白 reference 成功。OOM 路徑的 peak 值及失敗原因留在該路徑紀錄。

`--seed_groups` 指定候選 group ID；或使用 `--selection_seed` 產生 CPU `torch.randperm`，取前 `--regions` 個。不可行與重複候選都不補抽；固定候選全部不可行時仍寫報告。未使用這兩個選項則保留原本的隨機 seed 行為。

`proteins[].coverage` 保存候選可行／跳過／重複／去重數及 loss 殘基聯集覆蓋率；`seed_regions` 保存全部 group 的可行率。`observations` 保存逐 region／t／halo／殘基的原始 L2、參考範數、有效分量數與 RMS 差異；`strata` 按 t、halo、邊界圈數及實際截斷狀態分層，保留 median、p95、max。蛋白及 replica 由每個 protein entry 的名稱區別。

零／近零參考（參考 RMS 不超過 `1e-6`）與無 supervision 的相對誤差為 `null`；`strata` 分別計數，不當成非有限模型輸出。`summary.relative.undefined` 計 undefined 分母，`nonfinite` 計模型 NaN／Inf。正式判定應使用分層結果及已確認的門檻。

本次先導的核准範圍見 [比較提案](43-contract/proposal.md) 與 [門檻設定](43-contract/approved-forward.json)。設定 `MDGEN_ISSUE43_FRAMES`、`MDGEN_ISSUE43_DATA` 後，既有 launcher 透過 `scripts/issue43_pilot.py` 依固定次序呼叫本 CLI，保存每個 case 的參數、原始 JSON、log 與失敗狀態。`summary.json` 只省略逐殘基 observations；原始輸出 SHA-256 及路徑保留在 `manifest.json`。
