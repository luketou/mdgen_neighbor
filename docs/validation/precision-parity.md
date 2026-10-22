# Precision parity 生成比較器（#9 Ticket 5）

執行 `python -m mdgen.precision_parity manifest.json --output results/parity.json`，
產出 JSON 與同名 Markdown；全部已評估項目通過時 exit 0，失敗 exit 1。
這是與參考輸出的 Precision parity，不代表模型品質或 Scale validation 通過。

Manifest 範例：

```json
{
  "thresholds": {"first_rmsd_max": 1.0},
  "targets": [
    {"protein": "targetA", "seed": 42,
     "reference": "fp32.npy", "candidate": "candidate.npy", "md": "md.npy",
     "chain_id": [0, 0, 0, 1], "residue_index": [0, 1, 2, 0]}
  ]
}
```

檔案路徑相對於 manifest。兩組 NPY 為 `[rollout, frame, residue, atom, 3]`，
座標單位 Å；atom 的前三格必須為 N、CA、C（可直接使用 atom14／atom37）。
每段至少兩幀，兩組形狀及殘基／原子順序必須一致。MD 為
`[frame, residue, atom, 3]`，幀數可不同。使用者負責以相同起始構型及雜訊生成，
manifest 的蛋白名稱與 seed 是配對及重現資訊，不會觸發生成。
如需由 XTC／PDB 轉入，須先轉成上述順序並從 nm 轉為 Å。

選填 `residue_mask`、`chain_id`、`residue_index` 是每殘基一值，預設所有殘基有效、
單鏈、連續編號。至少三個有效 CA，且需有有效的連續配對。NaN／Inf 或錯誤輸入會拒絕。
幾何計算排除 padding、跨鏈及編號不連續配對。

幾何定義對齊 `luketou/mdgen` 的 `mdgen/full_length/reconstruction.py::_geometry_metrics`
及 `scripts/compare_local_spatial.py`（核對版本
`219bbac347482818a539900cd11a8ea3b3256fad`）：C–N 在閉區間 [1.1, 1.6] Å 外不合理，
Cα–Cα 在 [3.4, 4.2] Å 外不合理；分布包含 count/min/p05/median/p95/max/mean，
另增加 population std（ddof=0）。本 repo 未有該模組，因此以 NumPy 實作相同公式，
不引入 sibling repo 的私有函式及訓練相依。

每個 protein/seed 各自判定，任何一組失敗即整體失敗。幾何統計涵蓋所有 rollout／幀。
每幀 RMSD 以 Kabsch proper rotation 疊合相對應的第 1 段參考幀。
RMSF 則把各幀 CA 疊合到同一個參考組首幀，再計算逐殘基波動。
後續每段分別比較 RMSF Pearson；對 MD 則使用所有段合併後的每殘基 RMSF。
相同的常數 profile 相關定義為 1；其他無法定義的常數 profile 回報 null 並失敗。
未提供 MD 或沒有後續段時列入 `not_evaluated`，不計入已評估項目的通過判定。

可於 `thresholds` 覆寫以下預設值，每項報告包含 value/threshold/operator/passed：

| 名稱 | 門檻 |
| --- | --- |
| bad_cn_ratio_difference | ≤ 0.01（1 個百分點） |
| ca_mean_difference | ≤ 0.02 Å |
| ca_std_difference | ≤ 0.02 Å |
| md_rmsf_pearson_difference | ≤ 0.02 |
| first_rmsd_median | ≤ 0.3 Å |
| first_rmsd_max | ≤ 1.0 Å |
| later_rmsf_correlation | ≥ 0.95（各後續段的最小值） |

CPU 驗證：`python -m pytest tests/test_precision_parity.py -q`。
