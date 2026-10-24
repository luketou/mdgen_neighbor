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

載入 checkpoint 時一律關閉 gradient checkpointing、dropout 設為 0；非 `--inventory_only` 的模型量測以 `fp32-reference` 精度（matmul `highest`、TF32 關閉）執行。`--num_frames` 明確覆寫 checkpoint 的視窗長度；`--warmup`／`--repeats` 控制同輸入 forward 的同步時間與 GPU peak allocated／reserved 量測。在一般成對誤差模式中，region 的記憶體包含仍存活的完整蛋白 reference，不能當成獨立 region forward 的記憶體或直接宣稱省量。

`--inventory_only` 不實例化模型，只在 CPU Compute Job 讀 checkpoint 組態、分區與第零幀座標，列出每個 halo 設定全部種子 group 的最小 region 大小、分位數及不可行原因。

`--capacity_only` 分別量測並釋放完整蛋白與 region 輸出；完整蛋白 OOM 時仍繼續 region 路徑。此模式只報容量、時間與有限性，不產生誤差分布，也不保留 reference；一般比較模式則需要完整蛋白 reference 成功。OOM 路徑的 peak 值及失敗原因留在該路徑紀錄。

`--seed_groups` 指定候選 group ID；或使用 `--selection_seed` 產生 CPU `torch.randperm`，取前 `--regions` 個。不可行與重複候選都不補抽；固定候選全部不可行時仍寫報告。未使用這兩個選項則保留原本的隨機 seed 行為。

`proteins[].coverage` 保存候選可行／跳過／重複／去重數及 loss 殘基聯集覆蓋率；`seed_regions` 保存全部 group 的可行率。`observations` 保存逐 region／t／halo／殘基的原始 L2、參考範數、有效分量數與 RMS 差異；`strata` 按 t、halo、邊界圈數及實際截斷狀態分層，保留 median、p95、max。蛋白及 replica 由每個 protein entry 的名稱區別。

零／近零參考（參考 RMS 不超過 `1e-6`）與無 supervision 的相對誤差為 `null`；`strata` 分別計數，不當成非有限模型輸出。`summary.relative.undefined` 計 undefined 分母，`nonfinite` 計模型 NaN／Inf。正式判定應使用分層結果及已確認的門檻。

本次先導的核准範圍見 [比較提案](43-contract/proposal.md) 與 [門檻設定](43-contract/approved-forward.json)。設定 `MDGEN_ISSUE43_FRAMES`、`MDGEN_ISSUE43_DATA` 後，既有 launcher 透過 `scripts/issue43_pilot.py` 依固定次序呼叫本 CLI，保存每個 case 的參數、原始 JSON、log 與失敗狀態。`summary.json` 只省略逐殘基 observations；原始輸出 SHA-256 及路徑保留在 `manifest.json`。

## Issue #43 Ticket 3 正式 forward

`baseline_forward` 模式執行[凍結契約 v1.1](43-contract/frozen-contract-v1.1.json) 的三個基準蛋白矩陣，呼叫上述正式 Halo CLI，保留完整蛋白 reference 作成對比較。原有 `forward` 模式仍是容量先導。CPU 規劃指令（須能讀到 Ticket 1 資料與 checkpoint）：

```bash
python scripts/issue43_pilot.py --mode baseline_forward --plan_only \
  --data_root /share/home/r15543056/MDgen/mdgen/data/atlas_validation_ticket1_20261023b \
  --checkpoint /share/home/r15543056/MDgen/mdgen/ckpt/atlas.ckpt \
  --output_dir results/issue43-t3-plan
```

共有 162 個 CLI case：三蛋白 × R1–R3 × 4／250 幀 × 三個 t × 三個 noise seed。每個 case 同時比較 halo 1／2，以 selection seed 43002 的 randperm 前八候選、預算 896、暖機一次加三次量測執行。基準蛋白應全部完整涵蓋；這只能提供工具一致性證據。`--require_gpu` 防止正式執行時靜默退回 CPU。

省略 `--plan_only` 即執行；`--cases ID ...` 選擇明列 case，或加 `--shard_count N --shard_index I`（I 從 0 起）作互斥分片。分片不等於增加 job 配額；orchestrator 仍須遵守已核准資源上限。driver 預設 27 分鐘後停止啟動 case，未執行者留在 manifest；單 case 預設 timeout 為短幀 120 秒／250 幀 360 秒，會受剩餘 job 預算限制。可從 manifest 的未執行 ID 規劃下一批，使用新的輸出目錄以保留證據。實際成對模式總時間與容量尚待 GPU 驗證。

launcher 設定 `MDGEN_ISSUE43_MODE=baseline_forward`、`MDGEN_ISSUE43_DATA`、`MDGEN_ATLAS_CKPT`，並透過 `scripts/hpc_run.py` 的固定快照流程使用 `job/submit_halo_error.sh`；orchestrator 應指定 30 分鐘 scheduler limit。此模式不需要舊的 `MDGEN_ISSUE43_FRAMES`、`MDGEN_HALO_DATA`／`MDGEN_HALO_SPLIT`。launcher 會傳遞 case／shard 選項。manifest 保存來源檔 SHA-256、輸入 SHA-256、原始輸出 SHA-256、命令、log、Job ID，並沿用 `MDGEN_RUN_DIR` 的快照證據。

Halo CLI 可用 `--noise_seeds 0 1 2` 在同一呼叫重現雜訊矩陣；候選選擇與 noise seed 無關。同一 noise seed 在每個 t 重建同一份 CPU 常態雜訊，完整蛋白與各 region 共用該輸入。逐殘基 observation／stratum 明列 protein、replica、frames、noise seed 與 `paired_common_subset`。固定候選的深圈不可行時，會用同一候選選出可行淺圈的 loss groups，標為非共同子集；不補抽。只有全部 halo 都不可行才從模型比較中跳過該候選。舊隨機 `--seed` 選樣介面保留深圈不可行即跳過的行為。

`candidates[].by_halo` 保存每個候選在各 halo 的狀態與實際 region ID；重複 loss-group 集合只計算一次。`coverage.by_halo` 報 attempted、feasible、skipped、duplicate、deduplicated_regions、loss_residue_union_fraction，另報去重後的 full_coverage_regions／full_coverage_fraction；分母為零時 fraction 為 null。`seed_regions` 仍保留全部 seed group 的可行率。region geometry 保存 loss 殘基索引與邊界圈數，讓判定器能偵測缺少的 observation。非有限值在暖機及每次量測中分別保留，不會因最後一次輸出有限而消失。

以 CPU 判定原始 JSON（不要使用省略 observations 的 `summary.json`）：

```bash
python -m mdgen.full_length.forward_evaluation \
  --contract docs/validation/43-contract/frozen-contract-v1.1.json \
  --ticket 3 --manifest results/halo_error_JOBID/manifest.json \
  --output results/issue43-t3-evaluation.json
```

也可直接在選項前提供多個原始 JSON 路徑；多份 manifest 可一起傳入 `--manifest`。manifest 的原始輸出 hash 須相符；檔案路徑沿用執行時路徑，取回本機後需讓原始路徑可讀，或直接提供原始 JSON。判定器保存契約／manifest／原始 JSON 的 SHA-256，另保留 OOM、missing_gpu、timeout、program_error 等執行失敗。它不執行模型。

逐契約分層輸出 `pass`、`fail`（threshold／nonfinite）、`not_evaluable`（沒有 regular 樣本）、`incomplete`（缺少矩陣 cell、region 或 observation）。完整涵蓋套逐殘基包絡；截斷套各自的 linear p95／max 門檻。零／近零參考仍要求絕對誤差，relative 為 null；沒有 regular 樣本不能宣告通過。coverage 的 loss 殘基聯集按該邊界分層計算；無法指定邊界的 skipped 候選以該 halo 總數附列，不能跨邊界相加。全矩陣 `passed` 只在每個必要分層都為 `pass` 時為 true；即使其他分層通過，`not_evaluable` 也會阻止整體通過，不能以 pooled 數值替代。

判定前會核對凍結的 checkpoint hash、候選選法／數量、預算、halo、暖機／重複、FP32／TF32、dropout、local spatial 與模型幀數；不符或缺少必要設定時，報 `fail`／`settings_mismatch` 並列出欄位。候選完整但某 halo 全部不可行時，該 cell 為 `no_feasible_region`（保留 attempted／skipped 計數），與缺漏結果的 `incomplete` 分開，兩者都不算通過。同一 loss-group set 在同一 halo 只量一次；淺圈候選若重複成對候選，保留候選對應與各自的共同子集標記，唯一量測仍屬成對共同子集。

## Issue #43 Ticket 4 長蛋白 forward

`long_forward` 模式執行凍結契約 v1.1 的 `6sup_A`（2128 殘基）矩陣，命令與 `baseline_forward` 相同，只把蛋白換成 `forward_matrix.tickets.4`：R1–R3 × 4／250 幀 × 三個 t × 三個 noise seed，共 54 個 CLI case。launcher 設定 `MDGEN_ISSUE43_MODE=long_forward`。halo1 用 selection seed 43002 的前八候選；halo2 只有 group 95 可行，其餘七個候選以各自的 1 圈結果報告，標為非共同子集，不補抽。`6sup_A` 約為上游已評估最長蛋白的 2.94 倍，結果屬外推。

成對模式下任一路徑 OOM 時不中止程式：該路徑記為 `status: oom`，保存 `failure_phase`（`full_forward` 或 `region_forward`）、錯誤訊息與當時的 peak allocated／reserved，再繼續後續 t／seed／region。完整蛋白 OOM 時，同一 t／seed 的 region 全部不量（沒有 reference）。缺少的 observation 讓該分層為 `incomplete`，原因為 `oom`，並附上失敗路徑；不外推容量或誤差。其他階段（載入、prep）的 OOM 仍由 driver 記為 `failure_class: oom`。

`--ticket 4` 判定時另外核對：每個 replica 在深圈可行的候選須等於契約的 `long_protein_halo2_feasible_candidates`（`[95]`），不符則 `report_failures` 記 `deep_halo_candidates_mismatch`。每個分層的 `coverage.regions` 列出各 region 的殘基數，以及完整蛋白在 region 以外的上下文殘基數（`outside_context_residues`）。`common_subset_bias` 對每個淺圈 cell 並列全部 region 與成對共同子集的 absolute／relative median、p95、max 及其差（共同子集減全部），只作描述，不參與判定。

```bash
python -m mdgen.full_length.forward_evaluation \
  --contract docs/validation/43-contract/frozen-contract-v1.1.json \
  --ticket 4 --manifest results/halo_error_JOBID/manifest.json \
  --output results/issue43-t4-evaluation.json
```
