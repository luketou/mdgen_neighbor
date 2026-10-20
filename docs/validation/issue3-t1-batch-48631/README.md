# #3 Ticket 1：local 推論 batch 向量化的量測（2026-10-20）

結論：**無收益，不合併**。依 #3「先量測 B > 1 的時間與顯存，確認收益後再實作」，向量化實作保留在分支 `feat/issue-3-t1-batch-vectorize` 備查，不進 `main`。

## 量測條件

| 項目 | 值 |
|---|---|
| Slurm job | 48631（量測）、48630（正確性），分區 `l40s`，node-201 |
| GPU UUID | `a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb` |
| 向量化前 | `../mdgen2026` checkout，即 `refactor/issue-2-local-global` @ `42f43d4`（`mdgen/` 與 `25eef76` 相同；與 `2c85a7b` 的差異僅 `mdgen/full_length/{attention,model,partition}.py`） |
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

- 兩版時間皆隨 B 線性增加。向量化後在各 B 一致慢約 1%～1.5%（B=1 時也是），而同一次執行內 5 次的波動不到 0.1%，所以這是小而一致的額外開銷（推測來自合併與 reshape），不是量測誤差。兩版各只跑一次、依序執行，未量測跨次執行的變異。
- B ≥ 2 的峰值顯存兩版逐位元相同，峰值應位於非 local 層（temporal attention／FFN），本量測無法看出合併對顯存的影響。
- 推測原因：250 frames 下單一樣本已佔滿 L40S，合併呼叫只省下很小的 Python 迴圈開銷。

### 適用範圍

「無收益」只對以下情境成立：104 殘基單鏈、250 frames、B 個相同樣本、`cell_summaries=False`、`group_batch_size=4`。未測：更長的蛋白、較少的 frames、內容不同的樣本（group 數與寬度不同時的 padding）、`cell_summaries=True`（其 `_exchange_spatial` 在 `2c85a7b` 仍逐筆執行）。JSON 未記錄每個樣本的 group 數。

## 正確性

- 量測腳本比較「一次 B 筆」與「B 次各 1 筆」的整個模型輸出：向量化前 1.6e-5～2.6e-5，向量化後 2.2e-5～2.7e-5。向量化前沒有合併 group 也有此差異，所以它主要來自非 local 層隨 batch 大小改變的 kernel，不能用來單獨驗證向量化；兩版在 B > 1 也未直接互相比對。
- 向量化本身的等價性由 `tests/test_batch_vectorized_local.py` 在層級驗證：B=3、各樣本 partition 不同、mask 含 padding、`group_batch_size` 為 1／2／32，輸出、輸入梯度與權重梯度皆與逐筆 loop 一致（CPU，以及 job 48630 的 CUDA）。
- job 48630：CUDA pytest 52 passed；對凍結基準，global 逐位元相同，local 前向逐位元相同、輸入梯度 1.46e-10、權重梯度 1.30e-8（`checkpoint_parity_48630.json`）。
- job 48630 的第一版量測腳本只複製 tensor 參數，`frames`（Rigid）未複製，結果無效，已由 48631 取代。
