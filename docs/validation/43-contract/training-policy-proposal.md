# 成對 loss／gradient／optimizer 判定提案（待確認）

此提案供正式矩陣凍結前確認；目前 training-step 先導只量容量，不以其誤差推定以下門檻。工程容差未校準到生成品質。

兩側使用同一 checkpoint、FP32 reference、相同完整輸入的雜訊／t，再選出相同 loss 殘基；非條件幀與 checkpoint 有效通道的 mask 完全相同。Loss 使用 sum(masked squared velocity error) / sum(effective mask)，完整蛋白側也只監督對應 region 的相同 loss 殘基；分母數值必須保存且逐 case 相等。現有各自容量結果因 supervision 不同，不參與這項判定。

建議正式設定：Adam、lr 1e-4、betas (0.9, 0.999)、eps 1e-8、weight decay 0，global norm clipping 1.0；每側從同一初始權重及全新空 optimizer 狀態開始，只執行一個 step。關閉 dropout，開啟 gradient checkpointing、固定 FFN chunk 與空間參數。保存 clipping 前梯度與 clipping norm、更新後參數有限性，以及參數更新量 delta。

真正截斷 case 的建議門檻：

- Loss：逐 case 相對差 <= 0.01；reference loss <= 1e-12 時，另列近零，改用絕對差 <= 1e-12，不當成一般相對樣本。
- 各 active 參數群 clipping 前 gradient：cosine >= 0.99，norm 相對差 <= 0.05；每層報所有參數群與整體的 median／p95／max 及最差群組，不以 pooled 平均蓋過群組失敗。
- 各 active 參數群 optimizer 更新量 delta：cosine >= 0.99，norm 相對差 <= 0.05；另保存兩側 delta 的絕對 L2 與差值 L2。
- 梯度或更新 reference RMS <= 1e-12 的群組另計，不算一般 cosine／相對差；以差值 RMS <= 1e-12 判定。任一側非有限值均不通過。若沒有任何一般參考群組，方向一致性為無法評估。
- 完整涵蓋對照：相同 loss，兩側梯度／delta 的差值 RMS <= 1e-5 + 1e-5 * reference RMS，有限且相同 supervision；未達者視為工具一致性問題，不能混入截斷近似通過率。
- cell summaries 關閉造成兩側共同缺梯度的參數須逐名列出，與非預期 missing gradient 分開；其他單側缺梯度視為不通過。

理由：loss 允許百分之一的局部目標擾動；gradient／update 同時限制方向與大小，避免 norm 相近但方向偏移。cosine 0.99 對應小角度一致性，norm 5% 是獨立的大小容差。近零門檻保留絕對條件，避免除以微小值；這些都是待確認的工程選擇，並非由新先導量測調整而得。

如使用者選擇僅作描述性梯度報告，也必須在凍結契約明確紀錄該決定，不將描述性結果稱為訓練近似通過。
