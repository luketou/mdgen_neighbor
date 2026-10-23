---
status: accepted
---

# Training region 預設 896 殘基、1 圈 halo

Training region 的 `--region_residues` 預設為 896，`--halo_rings` 預設為 1。兩者都受同一個限制：FP32 在 L40S 上訓練一個 Training region × 250 幀，必須加 `--grad_checkpointing`，且 region 不能比約 1,000 殘基大多少；而 `check_region_budget` 要求預算 ≥ 訓練資料中最大的種子 region，否則啟動即報錯。

## 預算：896

- **下限來自資料。** Piezo1（6B3R，4,554 殘基）加 1 圈 halo 時，最大的種子 region 是 859 殘基；768 無法啟動訓練（[12-region-budget](../validation/12-region-budget/README.md)）。
- **上限來自記憶體。** 1024 的 peak reserved 為 43.73／44.39 GiB（job 49312），沒有餘裕；896 為 32.62 GiB、allocated 30.23 GiB、單步 6.3 s（job 49360，合成 7,500 殘基鏈 seed 0，FP32 + `--grad_checkpointing`），約有 12 GiB 餘裕留給不同 region 的變動。batch 一律 pad 到預算寬度，所以記憶體主要由預算決定，而非實際 region 大小。
- 換訓練資料時須重新確認最大種子 region；超過 896 時錯誤訊息會給出需要的殘基數。

## Halo：1 圈

- 1 圈的近似誤差：邊界圈數第 0 圈的 loss 殘基相對誤差中位數 2.2–2.8%、個別最大 10–16%；第 1 圈中位數 0.5–0.8%（[12-49339](../validation/12-49339/README.md)，兩個約 500 殘基的 ATLAS 蛋白）。2 圈可把最大值壓到約 2%。
- 2 圈不採用，主因不是 loss 殘基變少，而是放不下：Piezo1 加 2 圈的最大種子 region 為 1,873 殘基，需要約 1,900 的預算，遠超過 FP32／L40S 的上限。
- 邊界誤差是接受的近似。仍可用 `--halo_rings 2` 搭配足夠的預算明確開啟。

## 何時重新檢討

- BF16 訓練（#9 Ticket 4）或更大的 GPU 讓預算可達 2,048 以上時，重新比較 2 圈。
- 在長蛋白上量到的 halo 誤差明顯高於上述 ATLAS 數值時。

## Considered Options

- **1024**：每步 loss 殘基較多（Piezo1 1 圈、16 個 seed 的 loss 殘基 43–610，768 為 22–367），但記憶體只剩 0.66 GiB，其他 region 有 OOM 風險。
- **768**：記憶體寬裕（28.06 GiB），但 Piezo1 上無法啟動。
- **讓 `check_region_budget` 跳過放不下的種子 group**：那些 group 永遠不會成為 loss group，其殘基永遠不受監督，且不會有任何提示；維持啟動時報錯。
