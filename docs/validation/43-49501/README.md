# 250 幀訓練容量與控制先導

先導已寫入完成的 manifest：30 個測例完成、2 個失敗、21 個不可行種子、126 個重複候選、28 個因時間預算未執行。完成的測例 loss 均有限，梯度與更新後參數的非有限值計數均為零；三個基準蛋白 R1 控制均已完成，未套用控制驗收門檻。

兩個失敗都是 `6sup_A` 完整蛋白（R1、R2）在 backward 的 gradient checkpoint 重算中 CUDA OOM：L40S 總容量 44.39 GiB，PyTorch 已配置 43.00 GiB，再要求 230 MiB 失敗。R3 的完整蛋白未執行。同一長蛋白的 896 殘基 region（R1、R2 各 8 個 halo1 候選與 group 95 的 halo2）全部完成，實際 batch 為 `[1, 250, 896, 3]`。基準蛋白的 region 實際涵蓋全部殘基，屬完整涵蓋，不是截斷 case。

逐案例結果、失敗分類與 OOM 數值見 [彙整](outcome.json)；[原始 manifest](results/training_region_probe_49501/manifest.json) 保存命令、來源快照、輸入 hash、失敗階段與未執行紀錄，[原始摘要](results/training_region_probe_49501/summary.json) 保存模型設定、實際 batch、梯度、clipping、optimizer、記憶體及時間。失敗 log：[R1](results/training_region_probe_49501/6sup_A-R1-full.log)、[R2](results/training_region_probe_49501/6sup_A-R2-full.log)。

本次是 FP32、gradient checkpointing 開啟的單步容量測試。一般 full／region case 使用各自的 supervision，不能以其 loss 判斷近似誤差或宣稱效能收益。長蛋白沒有完整蛋白的 250 幀訓練 reference，因此無法在 250 幀做長蛋白的成對 loss／梯度比較；這是容量結果，不是近似誤差結果。

本 Job 沿用 `9c64271` 的唯讀快照，run ID 保留首次短幀訓練命名，但命令、設定與實際 batch 均為本次的 250 幀。GPU preflight 見 [JSON](results/preflight_49501.json)；stdout 以 [manifest 完成標記](logs/49501.out) 結束，另保存 [stderr](logs/49501.err)。終態依據是這個完成標記與 manifest 的 `driver_finished`；未保存佇列查詢或 `sacct` 輸出。Ticket 2 是否完成以 [契約狀態](../43-contract/README.md) 為準。
