# Issue #43 Ticket 4：`6sup_A` 長蛋白成對 forward

**結論：依[凍結契約 v1.1](../43-contract/frozen-contract-v1.1.json)，54 個 case 全部完成，162 個分層全部 `fail`（原因 `threshold`），整體 `passed: false`。** 沒有 OOM、非有限值、零或近零參考值，也沒有 `report_failures`。halo1 的誤差約為門檻的 10 倍；halo2 只有一個殘基可比較，誤差也仍高於 p95 門檻。這是 Issue #43 唯一的截斷 forward 證據。`6sup_A`（2128 殘基）約為上游已評估最長蛋白的 2.94 倍，屬外推。這些數字是工程容差判定，不代表物理或生成軌跡品質。

## 執行

- 程式：`feat/issue43-t4-long-forward` 的 `34bc2cb`；`scripts/issue43_pilot.py --mode long_forward`，透過 `job/submit_halo_error.sh`（[halo-error.md](../halo-error.md#issue-43-ticket-4-長蛋白-forward)）。
- run `issue43-t4-forward-20261024T062450Z-34bc2cb`：`l40s` 分區，單張 L40S，30 分鐘上限，`--shard_count 6`。
- Job：51235 為 smoke（shard 0），51238–51242 為 shard 1–5。原 smoke 51234 在啟動前取消：它的 nice 10000 會排在約 1,520 個 Ticket 5 Job 之後，改用 nice 0 重送，取消原因已記在 server 端的 `job_history`。
- 矩陣：R1–R3 × 250 幀（目標層）／4 幀（短幀層）× t 0.25／0.5／0.75 × noise seed 0／1／2，共 54 個 case，全部 `completed`，每個 driver 都是 `driver_finished: true`。
- 固定設定：FP32（matmul `highest`、TF32 關閉）、dropout 0、batch 1、local spatial 16 Å／128／128、cell summaries 關閉、`group_batch_size` 4、`ffn_chunk_size` 65536、預算 896、暖機 1 次加量測 3 次。判定器核對後沒有設定不符。

## 證據檔

- [evaluation.json](evaluation.json)：`forward_evaluation --ticket 4` 的輸出，含 162 個分層、coverage（含各 region 的上下文）與 `common_subset_bias`，並保存每份原始 JSON 的 SHA-256。
- [manifests/](manifests)：各 Job 的 driver manifest，含命令、來源與輸入的 SHA-256，以及原始輸出 hash。
- [logs/](logs)：各 Job 的 stdout／stderr。
- 54 份原始 JSON（含逐殘基 observations，合計約 17 MB）只留在 server 的 `hpc_runs/<run>/results/halo_error_<job>/`。判定前已逐一核對 hash 與 manifest 相符。

## 候選、可行率與上下文

| halo | 候選 | 可行 | 去重 region | loss 殘基聯集比例 |
|---|---|---|---|---|
| 1 | 8（seed 43002 的 randperm 前 8 個：89, 65, 63, 115, 28, 141, 92, 95） | 8 | 8 | 7 個非共同子集 region 合計 16.7%，加上 group 95 再多 1 個殘基 |
| 2 | 8 | 1（group 95） | 1 | 0.05%（1 個殘基） |

- 三個 replica 結果相同：深圈可行候選等於契約的 `[95]`。全部 group 的可行率見契約：halo1 189/190、halo2 51/190。
- halo1 的 8 個 region 有 726–881 個殘基、34–66 個 loss 殘基，各缺少 1,247–1,402 個殘基的上下文。group 95 在 halo1 只有 270 個殘基，缺少 1,858 個；在 halo2 有 883 個殘基，缺少 1,245 個。
- 所有 loss 殘基的邊界圈數都是 0：每個 loss group 都有 ghost 落在 loss groups 以外。

## 誤差（`a/sqrt(N)` 與 relative；截斷門檻 p95 ≤ 0.01、max ≤ 0.05）

每列先統計 27 個分層（3 replica × 3 t × 3 noise seed），再列出跨分層的最小／中位／最大值。

| 幀 | 分層 | 樣本／分層 | relative p95 | relative max | abs p95 | abs max |
|---|---|---|---|---|---|---|
| 250 | halo1，非共同子集 | 356 | 0.078／0.095／0.102 | 0.151／0.181／0.248 | 0.100／0.105／0.141 | 0.188／0.215／0.352 |
| 250 | halo1，共同子集（group 95） | 1 | 0.092／0.113／0.150 | 同左 | 0.122／0.153／0.163 | 同左 |
| 250 | halo2，共同子集（group 95） | 1 | 0.011／0.014／0.029 | 同左 | 0.016／0.019／0.031 | 同左 |
| 4 | halo1，非共同子集 | 356 | 0.119／0.134／0.205 | 0.177／0.250／0.437 | 0.138／0.151／0.205 | 0.192／0.291／0.380 |
| 4 | halo1，共同子集（group 95） | 1 | 0.094／0.139／0.268 | 同左 | 0.109／0.166／0.260 | 同左 |
| 4 | halo2，共同子集（group 95） | 1 | 0.010／0.021／0.085 | 同左 | 0.010／0.023／0.074 | 同左 |

- 分層都沒有合併判定：162 個分層各自超過門檻。只有一個 halo2 4 幀分層的 relative 0.0096 低於 0.01，但它的 `a/sqrt(N)` 0.0104 仍超過門檻。
- 在 group 95 上，halo2 的誤差約為 halo1 的 1/8（250 幀 relative 中位 0.014 對 0.113）。這只是 **單一殘基** 的成對比較，不能推廣到其他 region 或一般 training startup。
- 共同子集偏差（`common_subset_bias`，halo1 的共同子集減全部 region，relative）：250 幀時 median +0.05～+0.10、p95 +0.01～+0.05、max −0.13～−0.04；4 幀時 median +0.03～+0.18、max −0.25～−0.01。group 95 的單一殘基高於全部 region 的中位數，但低於最大值。這只作描述，不參與判定。

## 資源（同步後計時；peak 每次量測前重設）

| 幀 | 路徑 | 量測次數 | 時間中位／最大 | peak allocated 最大 | peak reserved 最大 |
|---|---|---|---|---|---|
| 250 | 完整蛋白（2128 殘基） | 81 | 4.79／4.84 s | 8.82 GiB | 11.54 GiB |
| 250 | halo1 region（270–881 殘基） | 648 | 1.09／1.23 s | 4.03 GiB | 15.73 GiB |
| 250 | halo2 region（883 殘基） | 81 | 1.17／1.19 s | 4.02 GiB | 15.73 GiB |
| 4 | 完整蛋白 | 81 | 0.77／0.81 s | 0.78 GiB | 2.37 GiB |
| 4 | halo1 region | 648 | 0.14／0.17 s | 0.41 GiB | 2.37 GiB |
| 4 | halo2 region | 81 | 0.14／0.15 s | 0.42 GiB | 2.37 GiB |

- region 的 reserved 包含先前完整蛋白 forward 留下的 allocator cache，所以不據此宣稱記憶體收益；allocated 才是各路徑本身的 peak。
- 每個 case 的牆鐘時間（含載入）：250 幀中位 77 s、最大 87 s；4 幀中位 29 s、最大 37 s。54 個 case 合計約 48 分鐘 GPU 時間。
- 成對模式下所有路徑都沒有 OOM，所以沒有「未完成」的比較。容量只在 L40S（44.39 GiB）上實測，不外推到其他 GPU 或更長的蛋白。

## 限制

- halo2 與成對共同子集只有 group 95 的 1 個 loss 殘基，halo1 與 halo2 的比較建立在單一殘基上。要擴大這個子集，需要另立契約版本並經使用者確認，不能依本結果調整。
- 只描述固定的 8 個候選，不代表全部 seed group 或一般 training startup。三個 replica 與兩個幀數層不是獨立的空間樣本。
- checkpoint 以 `crop=256`、250 幀訓練；完整的 2128 殘基蛋白與 4 幀都屬外推。400 ps 間隔的限制照舊。
