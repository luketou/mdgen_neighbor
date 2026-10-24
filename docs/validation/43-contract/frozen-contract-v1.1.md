# 比較契約 v1.1（已凍結）

使用者於 2026-10-24 確認：「補比較契約 v1.1：loss 分母為零（region 沒有受監督分量）的 case 標為 no_supervision，不計入 loss／梯度比較、單獨計數，不當成 NaN 失敗。寫進契約並推送到 PR #50。」機器可讀版本見 [frozen-contract-v1.1.json](frozen-contract-v1.1.json)。

v1.1 等於 [v1](frozen-contract-v1.md) 加上下面這一條規則；v1 的其他決定全部不變。Ticket 3–5 照 v1.1 執行。

## Loss 分母為零

成對訓練比較中，若某個 case 的有效 loss-mask 分母為零（region 沒有任何受監督分量）：

- 標為 `no_supervision`，不計入 loss、梯度與更新的比較；
- 在每個分層中與不可行、重複候選一樣單獨計數並列出；
- 不算 NaN／Inf 失敗，也不算通過。

分母不為零時出現的 NaN／Inf 仍依 v1 的失敗政策讓該 case 不通過。forward 的逐殘基 `N_i == 0` 原本就記為無 supervision，規則不變。
