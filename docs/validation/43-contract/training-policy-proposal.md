# 成對 loss／gradient／optimizer 數值政策

狀態：使用者以 `Q5 是 Q6 是` 確認下列政策；控制容差與近零分類的數值尚待校準並再次確認，最終契約尚未凍結。原始來源依據見 [研究紀錄](training-threshold-research.md)，整體決策見 [訪談紀錄](discussion.md)。

## 真正截斷的比較

Loss、clipping 前 gradient 及實際 optimizer 更新量 delta 採描述性報告。舊草稿的 loss 差 1%、cosine 0.99、norm 差 5% 沒有被核准為硬門檻，亦不輸出「訓練近似通過」的判定。已核准的 forward 門檻保持獨立。

兩側固定同一 checkpoint、FP32 reference、實際 noise、t、loss 殘基、有效通道與非條件幀；先抽完整 noise，再依完整蛋白索引選取 region。Loss 使用相同 masked squared error 分子定義及有效 mask 分母，逐 case 保存兩側分母並檢查相等。僅重設相同 seed 或比較各自 capacity loss 不足以滿足此條件。

報告保存 loss 的 signed／absolute difference、reference 值與可評估的相對差；各參數群保存 reference／candidate／difference norm 與 RMS、cosine、norm ratio、差向量相對 norm，以及實際 delta。保持逐群結果與有限性、inactive／missing、零／近零類別；不可靜默刪除不穩定分母樣本。

## 控制校準與有效性

先取得兩類零截斷控制：同路徑 FP32 重跑，以及完整涵蓋且 supervision 相同的 full／region 對照。保存兩側原始尺度、有效分量與差異，依控制證據提出工具一致性容差與近零分類數值，經使用者確認後在正式比較前凍結。真正截斷的誤差不參與校準，不據其結果放寬標準。

數值尚未確認前，只報控制差異，不預設控制通過。已確認的控制若失敗，先查比較工具與設定；不可將它混入截斷近似分布。輸入、輸出、loss、梯度、更新後參數的非有限值及兩側 supervision 不一致均須明列；比較有效性與近似差異分開呈現。

## 執行約定

每側從相同初始權重與全新相同 optimizer 狀態開始，沿用既有 probe 的 Adam 與 clipping 設定並明列實際參數。保存 clipping 前梯度、兩側 clipping norm／係數及是否觸發，再執行 optimizer 單步；實測 delta，不用理想更新公式替代。空狀態 Adam 單步不能代表恢復訓練的動差歷史。

關閉 dropout，明列 checkpointing、chunking、空間參數及 batch shape。停用 cell summaries 造成的預期無梯度參數按名稱列出，與非預期缺梯度及單側缺失分開。沿用已確認的兩個 CLI／JSON 測試介面，擴充正式工具，不建立分岔模型。

這些結果只描述固定可行子集、checkpoint 與單步設定；不支持全部種子可用、長期收斂、訓練穩定性或生成品質宣稱。
