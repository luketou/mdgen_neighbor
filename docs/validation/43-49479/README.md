# 短幀訓練容量與控制先導

先導已寫入完成的 manifest：49 個測例完成、21 個不可行種子、126 個重複候選、11 個因時間預算未執行；完成的測例 loss 均有限，梯度與更新後參數的非有限值計數均為零。三個基準蛋白 R1 控制均已完成；未套用控制驗收門檻。

逐案例結果見 [彙整](outcome.json)；[原始 manifest](results/training_region_probe_49479/manifest.json) 保存命令、來源快照、輸入 hash、失敗與未執行紀錄，[原始摘要](results/training_region_probe_49479/summary.json) 保存模型設定、實際 batch、梯度、clipping、optimizer、記憶體及時間。

本次是 FP32 單步容量測試。一般 full／region case 使用各自的 supervision，不能以其 loss 判斷近似誤差或宣稱效能收益。控制測例則對齊 noise、t、supervision 與分母，報同路徑重跑及完整涵蓋對照的原始梯度／實際更新差異；尚未套用控制驗收容差。

GPU preflight 見 [JSON](results/preflight_49479.json)，執行終止證據以 [stdout](logs/49479.out)、[stderr](logs/49479.err) 及 manifest 的 `driver_finished` 一併核對；叢集沒有可用的 `sacct` 終態證據。Ticket 2 是否完成以 [契約狀態](../43-contract/README.md) 為準。
