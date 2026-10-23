# 長蛋白訓練幀數與記憶體設定掃描（GPU A）

依 [追加申請](../43-contract/gpu-request-frames.md) 對 `6sup_A` R1 完整蛋白（2128 殘基）跑 FP32 訓練步，gradient checkpointing 開啟，batch 1，seed 0，t 0.5。結果：

| 幀數 | 記憶體設定 | 狀態 | 記憶體 |
|---:|---|---|---|
| 64 | 預設（`group_batch_size` 4、`ffn_chunk_size` 65536） | 完成 | peak allocated 34668308992 bytes（32.29 GiB） |
| 96 | 預設 | backward OOM | PyTorch 已配置 40.24 GiB |
| 128 | 預設 | 依規則跳過（96 幀已 OOM） | — |
| 250 | `group_batch_size` 1 | backward OOM | 已配置 42.68 GiB |
| 250 | 再加 `ffn_chunk_size` 8192 | backward OOM | 已配置 42.68 GiB |
| 250 | 再加 `expandable_segments` | backward OOM | 已配置 43.45 GiB |

64 幀的四次訓練步 loss 均有限，梯度與更新後參數的非有限值計數為零，實際 batch 為 `[1, 64, 2128, 3]`。在已確認的固定設定與單張 L40S（44.39 GiB）下，長蛋白完整蛋白訓練步的最大可行幀數為 64；96 幀以上不可行。這三種分批與配置設定都沒有讓 250 幀放得下。介於 64 與 96 之間的幀數沒有量測。

逐案例結果與 OOM 數值見 [彙整](outcome.json)；[原始 manifest](results/training_region_probe_49513/manifest.json) 保存命令、快照、輸入 hash、每個 case 的實際記憶體設定與跳過原因，[原始摘要](results/training_region_probe_49513/summary.json) 保存 64 幀的完整報告。失敗 log：[96 幀](results/training_region_probe_49513/6sup_A-R1-full-f96.log)、[250 幀 gbs1](results/training_region_probe_49513/6sup_A-R1-full-f250-gbs1-ffn65536.log)、[250 幀 gbs1 ffn8192](results/training_region_probe_49513/6sup_A-R1-full-f250-gbs1-ffn8192.log)、[250 幀加 expandable](results/training_region_probe_49513/6sup_A-R1-full-f250-gbs1-ffn8192-expandable.log)。

這是容量結果，不是近似誤差。本 Job 的快照為 `02edb1f`（run `issue43-t2-sweepA-20261023T170924Z-02edb1f`），[preflight](results/preflight_49513.json) 通過；佇列已無本 Job，stdout 以 [manifest 完成標記](logs/49513.out) 結束，另保存 [stderr](logs/49513.err)。叢集沒有可用的 `sacct` 終態證據。
