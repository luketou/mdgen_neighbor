# 64 幀訓練容量確認與控制（GPU B）

依 [追加申請](../43-contract/gpu-request-frames.md)，以 [GPU A](../43-49513/README.md) 找到的最大可行幀數 64、預設記憶體設定（`group_batch_size` 4、`ffn_chunk_size` 65536）執行 7 個指定 case，全部完成：

- `6sup_A` R2、R3 完整蛋白訓練步：peak allocated 均約 32.29 GiB，實際 batch `[1, 64, 2128, 3]`。R1 已在 GPU A 量測，此處不重跑。
- `6sup_A` R1 group 95 的 halo1 與 halo2 region：實際 883 殘基、截斷，batch `[1, 64, 896, 3]`。
- 三個基準蛋白 R1 的完整涵蓋控制：noise／mask hash 與有效分母兩側一致；同路徑重跑差異為零，完整涵蓋差異列入 [控制校準](../43-contract/control-calibration.md)。

所有 case 的 loss 均有限，梯度與更新後參數的非有限值計數均為零。逐案例數值見 [彙整](outcome.json)；[原始 manifest](results/training_region_probe_49515/manifest.json) 保存命令、快照、輸入 hash、選取的 case 與實際記憶體設定，[原始摘要](results/training_region_probe_49515/summary.json) 保存模型設定、梯度、clipping、optimizer、記憶體及時間。

一般 full／region case 使用各自的 supervision，不能以其 loss 判斷近似誤差或宣稱效能收益；未套用控制驗收容差。64 幀是獨立的一層，不能代替 250 幀，checkpoint 以 250 幀訓練，64 幀也屬外推。

本 Job 沿用 GPU A 的快照 `02edb1f`（run `issue43-t2-sweepA-20261023T170924Z-02edb1f`）。[preflight](results/preflight_49515.json) 通過；佇列已無本 Job，stdout 以 [manifest 完成標記](logs/49515.out) 結束，另保存 [stderr](logs/49515.err)。叢集沒有可用的 `sacct` 終態證據。
