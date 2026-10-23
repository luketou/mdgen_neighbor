# mdgen_neighbor

以 local spatial attention 讓 MDGen 處理數千殘基的完整長度蛋白；本詞彙表收錄空間分區與 Training region 的用語。

## 空間分區

**Group**：
完整蛋白第 0 幀依空間格點切出的一組殘基，是 local attention 的計算單位。
_Avoid_：cell、block

**Core**：
一個 group 擁有並負責更新的殘基；每個殘基恰屬於一個 group 的 core。
_Avoid_：owned residues

**Ghost**：
在某 group 的接觸半徑內、但由其他 group 擁有的殘基；該 group 只讀取它們，不更新它們。
_Avoid_：neighbor residues、鄰居殘基

## Training region

**Training region**：
從完整蛋白第 0 幀分區選出的 loss group、halo 與上下文殘基，構成一個訓練樣本；其大小以這三者的殘基總數計。
_Avoid_：crop、空間切塊、單獨使用的「region」

**Loss group**：
Training region 中 core 計入 loss 的 group；這些 core 殘基稱為 **loss 殘基**。
_Avoid_：loss core、以 core 指稱 loss 範圍

**種子 group**：
Training region 開始長出 loss group 的那個 group；其餘 loss group 依離它的遠近加入。
_Avoid_：center、中心 group

**種子 region**：
只以種子 group 為 loss group 時的 Training region，是以該 group 為起點所需的最小 region。
_Avoid_：minimum region

**Halo**：
由 loss group 向外逐圈擁有其 ghost 的 group，深度為 halo 圈數；它們正常更新、不計入 loss，其中每個 group 稱為 halo group。
_Avoid_：ghost group、buffer

**上下文殘基**（context residue）：
halo group 的 ghost 中不屬於任何被選 group 的殘基；只作為 key，空間狀態不更新。
_Avoid_：在 Training region 層級稱為 ghost

**Halo 圈數**：
Halo 從 loss group 向外延伸的層數。
_Avoid_：單獨使用的「圈」

**邊界圈數**：
loss 殘基到 Training region 邊界的層數；第 0 圈是 ghost 直接由 region 外的 group 擁有的 loss group。
_Avoid_：單獨使用的「圈」、與 halo 圈數混用
