# PR #19 後續修正：三項 P2 的回歸驗證

在已合併 PR #19 的 `31cf5b4` 基準上，三項修正通過 CPU 驗證。
後續若整合更新主線，需另留該快照的驗證紀錄，不將本報告視為新版本的結果。

- 最終 job：`49060`，run `pr19-fix-final-green-20261023`；`l40s` partition/account，`node-201`，`COMPLETED`，`ExitCode=0:0`。
- 2 CPU、12G、10 分鐘上限，不申請 GPU；Python `3.11.12`、PyTorch `2.11.0+cu128`。
- 載入 `python/3.11.12-gpu` module 與 `/share/home/r15543056/.venvs/mdgen`，明確停用 CUDA。
- 固定快照：`/share/home/r15543056/MDgen/pr19-fix-final-green-20261023/source`，輸出獨立於 `results/`。
- `manifest.sha256` 的 148 個程式／設定檔於執行前核對通過；`snapshot.json` 記錄基準及 dirty 狀態。
- 遠端 repo 先 pull 至 `31cf5b4`；實驗只使用固定快照。

| Job | 階段 | 結果 | 紀錄 |
| --- | --- | --- | --- |
| 49055 | 跨鏈／編號中斷，修正前 | 2 failed（預期） | `chain-red.xml`、`chain-red.out` |
| 49056 | 跨鏈／編號中斷，修正後 | 2 passed | `chain-green.xml`、`chain-green.out` |
| 49057 | CSV 部分列留白，修正前 | 3 failed（預期） | `blank-red.xml`、`blank-red.out` |
| 49058 | CSV 部分列留白，修正後 | 3 passed | `blank-green.xml`、`blank-green.out` |
| 49059 | 兩個訓練入口的 overfit 驗證，修正前 | region 2 failed（預期）、legacy 2 passed | `overfit-red.xml`、`overfit-red.out` |
| 49060 | 全部資料集／訓練入口測試 | 21 passed | `dataset.xml` |
| 49060 | 完整 CPU suite | 117 passed、8 skipped | `full-suite.xml`、`pytest-49060.out` |

訓練入口測試只替換外部 Lightning Trainer，在 `fit` 接收的 DataLoader 上觀察資料取樣；
不模擬自己的資料集，不宣稱跑過正式訓練。完整 suite 包含既有真實模型的 CPU 訓練探針。
8 個 skip 為需要 CUDA 的既有測試。本次未驗證 GPU 容量／效能。
Python syntax 與 `git diff --check` 通過；此基準未配置靜態型別檢查器。
機器可讀摘要見 `summary.json`；launcher 見 `submit.sh`。文字 logs／JUnit 僅移除行末空白，遠端保留原始產物。
