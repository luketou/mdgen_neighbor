# 八候選先導計畫驗證

CPU Job `49478` 完成真實輸入雜湊、分區與候選清單驗證，沒有執行模型。來源 run／commit、全部輸入與命令見 [manifest](results/issue43-t2-plan/manifest.json)，計數見 [outcome](outcome.json)，執行完成訊息見 [stdout](logs/49478.out)。

共 207 個候選紀錄：60 個待執行、126 個重複、21 個種子超出預算；重複與不可行候選均保留而不補抽。各設定的實際 loss-group 集合及空間大小由正式 sampler 產生。長蛋白 R1 排在三個基準蛋白 R1 前，其餘 replica 隨後。

三個基準蛋白 R1 的完整涵蓋控制均列入計畫。這個 CPU Job 只驗證計畫與資料可讀，不代表 GPU 容量、控制一致性或數值門檻已驗收。原始 stderr 的不可寫 NumPy view 警告保留於 [log](logs/49478.err)。
