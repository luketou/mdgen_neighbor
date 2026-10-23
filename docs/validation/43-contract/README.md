# 比較契約與先導進度

Forward 門檻、候選矩陣、測試介面與先導額度已獲確認，見 [核准設定](approved-forward.json) 與 [完整提案](proposal.md)。最終可行矩陣尚未凍結，Ticket 2 維持未完成。

工具已實作並經雙軸 [審查](review.md)；完整 CPU 檢查為 218 passed，證據見 [檢查設定](tool-checks.json) 與 [log](tool-checks.log)。

已完成 [CPU group 盤點](../43-49466/README.md) 與 [短幀 forward 容量先導](../43-49469/README.md)。剩餘資源與尚未執行項目見 [進度紀錄](pilot-progress.json)；[短幀訓練先導](../43-49479/README.md) 已結束並保留未執行項目；[250 幀 forward](../43-49489/README.md) 已完成；[250 幀訓練先導](../43-49501/README.md) 已結束，長蛋白完整蛋白的訓練步在 backward OOM。四個 GPU Job 額度已用盡，先導結束。

依先導結果整理的 [最終矩陣草案](final-matrix-proposal.md) 與納入 250 幀控制的 [控制校準提案](control-calibration.md) 待使用者確認；確認前不凍結矩陣，Ticket 2 維持未完成。

已確認 [候選範圍](capacity-decision.md) 與 [訓練數值政策](training-policy-proposal.md)：截斷的訓練比較採描述性報告；控制容差及近零分類先校準，再確認並凍結。方法依據見 [原始來源查核](training-threshold-research.md)。既有本機紀錄 commit 已另拆為 PR #46；本分支的 Ticket 2 工作沒有推送或發佈。

[Ticket 2 訪談](discussion.md) 的逐項決策已收斂，控制測例與先導已完成；最終矩陣尚未凍結。
