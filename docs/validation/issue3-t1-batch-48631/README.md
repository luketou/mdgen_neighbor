# #3 Ticket 1：local 推論 batch 向量化的量測（2026-10-20）

結論：**無收益，不合併**。依 #3「先量測 B > 1 的時間與顯存，確認收益後再實作」，向量化實作保留在分支 `feat/issue-3-t1-batch-vectorize` 備查，不進 `main`。

## 量測條件

| 項目 | 值 |
|---|---|
| Slurm job | 48631（量測）、48630（正確性），分區 `l40s`，node-201 |
| GPU UUID | `a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb` |
| 向量化前 | `refactor/issue-2-local-global` @ `42f43d4`（`mdgen/` 與 `25eef76` 相同） |
| 向量化後 | `feat/issue-3-t1-batch-vectorize` @ `2c85a7b` |
| 輸入 | `fixture(104)`（5e3e_A），250 frames，原始 batch 複製 B 份後經 `prep_batch` |
| 設定 | ATLAS checkpoint，local、`cell_summaries=False`、`group_batch_size=4`，float32，TF32 關閉 |
| 計時 | `forward_inference`，暖機 1 次後 5 次取中位數 |

同一個 job、同一張卡、同一支腳本（`scripts/diagnostics/benchmark_local_batch.py`，以 `--repo` 選擇程式版本）。

## 結果

| B | 向量化前 | 向量化後 | 峰值顯存（前後相同） |
|---:|---:|---:|---:|
| 1 | 0.234 s | 0.237 s | 1.49 GB |
| 2 | 0.486 s | 0.491 s | 1.72 GB |
| 4 | 0.978 s | 0.992 s | 2.32 GB |
| 8 | 1.956 s | 1.974 s | 4.12 GB |

- 兩版時間皆隨 B 線性增加，向量化後約慢 1%，在量測誤差範圍內；顯存相同。
- 推測原因：250 frames 下單一樣本已佔滿 L40S，合併呼叫只省下很小的 Python 迴圈開銷。只測了 104 殘基；更長的蛋白或更少的 frames 未測。

## 正確性

- B > 1 與逐筆計算的最大差異：向量化前 1.6e-5～2.6e-5，向量化後 2.2e-5～2.7e-5，來源為 group 批次組成改變造成的浮點誤差，兩版同量級。
- job 48630：CUDA pytest 52 passed；對凍結基準，global 逐位元相同，local 前向逐位元相同、輸入梯度 1.46e-10、權重梯度 1.30e-8（`checkpoint_parity_48630.json`）。
- job 48630 的第一版量測腳本只複製 tensor 參數，`frames`（Rigid）未複製，結果無效，已由 48631 取代。
