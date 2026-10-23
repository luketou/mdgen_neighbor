# 比較契約與先導進度

**比較契約 v1 已於 2026-10-24 凍結，見 [frozen-contract-v1.md](frozen-contract-v1.md) 與 [JSON](frozen-contract-v1.json)。** Ticket 3–5 照此版本執行；修改需另立新版本並重新確認。

Forward 門檻、候選矩陣、測試介面與先導額度先前已獲確認，見 [核准設定](approved-forward.json) 與 [完整提案](proposal.md)。工具已實作並經雙軸 [審查](review.md)；完整 CPU 檢查為 218 passed，證據見 [檢查設定](tool-checks.json) 與 [log](tool-checks.log)。

已完成 [CPU group 盤點](../43-49466/README.md) 與 [短幀 forward 容量先導](../43-49469/README.md)。剩餘資源與尚未執行項目見 [進度紀錄](pilot-progress.json)；[短幀訓練先導](../43-49479/README.md) 已結束並保留未執行項目；[250 幀 forward](../43-49489/README.md) 已完成；[250 幀訓練先導](../43-49501/README.md) 已結束，長蛋白完整蛋白的訓練步在 backward OOM。四個 GPU Job 額度已用盡，先導結束。

依核准的 [追加申請](gpu-request-frames.md)，[幀數掃描](../43-49513/README.md) 找到長蛋白完整蛋白訓練步的最大可行幀數為 64（96 幀與 250 幀的三種記憶體設定均 OOM），[64 幀確認](../43-49515/README.md) 的 7 個 case 全部完成。追加額度用了 2 個 GPU Job，CPU 檢查在本機 WSL 執行。

凍結內容依據 [最終矩陣草案](final-matrix-proposal.md) 與 [控制校準](control-calibration.md)（選項 B）。[候選範圍](capacity-decision.md) 與 [訓練數值政策](training-policy-proposal.md) 的決策過程見 [Ticket 2 訪談](discussion.md)，方法依據見 [原始來源查核](training-threshold-research.md)。既有本機紀錄 commit 已另拆為 PR #46。
