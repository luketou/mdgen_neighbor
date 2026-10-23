# 250 幀 forward 容量先導

24 個 CLI case 均完成；內部 81 條模型路徑完成、輸出非有限值計數為零，包括 24 次完整蛋白 forward、27 條完整涵蓋 region、30 條真正截斷 region。結果與逐路徑量測見 [彙整](outcome.json)；這是容量結果，未計算成對輸出誤差或套用正式 forward 門檻。

[原始 manifest](results/halo_error_49489/manifest.json) 保存命令、來源快照與輸入 hash；[原始摘要](results/halo_error_49489/summary.json) 保存 batch shape、候選到實際 loss-group／region 的對應、不可行與重複候選、實際 FP32 設定及每次量測。完整蛋白與 region 分開執行並釋放 reference 輸出；peak reserved 含 allocator cache，不據此宣稱記憶體或速度收益。

本 Job 沿用 `9c64271` 的唯讀快照，run ID 保留首次短幀訓練命名，但命令、設定與實際 batch 均為本次的 250 幀。[preflight](results/preflight_49489.json) 通過；stdout 以 [manifest 完成標記](logs/49489.out) 結束，另保存 [stderr](logs/49489.err)。終態依據是這個完成標記與 manifest 的 `driver_finished`；未保存佇列查詢或 `sacct` 輸出。

長蛋白 halo2 仍只有固定前八候選中的 group 95 可行，replica 不視為獨立空間樣本。全部 group 的可行率與先前範圍決策見 [契約](../43-contract/README.md)。
