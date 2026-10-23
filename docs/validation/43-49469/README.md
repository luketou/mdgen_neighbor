# 短幀 forward 容量先導

GPU Job `49469` 在 L40S 上完成短幀容量流程：batch 1、4 幀、FP32 reference；完整蛋白及 region 獨立量測，暖機 1 次、量測 3 次。來源 run／commit、執行參數、全部輸入與 checkpoint hash 見 [manifest](results/halo_error_49469/manifest.json)；完整統計與路徑計數見 [outcome](outcome.json)，原始量測見 [summary](results/halo_error_49469/summary.json)。

- 四蛋白、全部 replica 的 24 個 CLI case 均退出成功；其中長蛋白的 3 個 paired case 沒有可行 region，不能算模型路徑成功。
- 真正截斷：長蛋白的一圈 halo 共 12 個 region 路徑完成，實際 region 為 726、819、841、866 殘基；每個 replica 各自保存，不能當成獨立蛋白數。
- 完整涵蓋：三個基準蛋白共 27 個 region 路徑完成，不將其平凡涵蓋結果用來代表截斷近似。
- 所有實際量測的輸出均有限。兩圈 halo 的不可行 seed、重複 region 與 loss 殘基覆蓋率保留在每個 protein entry 中。

此模式沒有量測成對輸出誤差；沒有判定 forward 門檻、loss／gradient 或生成品質，也沒有驗證 250 幀與 backward。Region 的 reserved 記憶體可能保留前一路徑的 allocator cache，不能由此宣稱獨立 region 的 reserved 省量；allocated 與同步時間保留原始值，沒有宣稱加速比。

排程器已無此 Job，所有 case 的原始 return code 均為成功；GPU [preflight](results/preflight_49469.json) 通過，執行完成訊息見 [stdout](logs/49469.out)，[stderr](logs/49469.err) 僅含 module 載入訊息。未使用不可用的 sacct 推定 Slurm 最終退出碼。

下一步依 [容量決策草稿](../43-contract/capacity-decision.md) 確認候選範圍，再使用剩餘核准額度繼續先導；最終矩陣尚未凍結。
