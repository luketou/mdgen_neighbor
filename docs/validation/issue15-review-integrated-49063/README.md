# PR #19 後續修正：整合最新主線驗證

結論：三項 P2 修正接上已合併 Ticket 4 的主線後，完整 CPU suite 通過。
本報告為當時的整合快照，最新主線結果見 `../issue15-review-topology-49232/`；修正前與中間版本的 red → green 證據見 `../issue15-review-fixes-49060/`。

- Slurm job `49063`，run `pr19-fix-integrated-green-20261023`，`l40s` partition/account，`node-201`。
- 排程狀態 `COMPLETED`，`ExitCode=0:0`，詳見 `scheduler.txt`、`completion.txt`。
- 原始結果：資料集／訓練入口 **21 passed**（`dataset.xml`）；完整 suite **119 passed、8 skipped**（`full-suite.xml`、`pytest-49063.out`）。
- 主線基準 `e62d658`，實作 commit `ae7c28a`；`snapshot.json` 與 `manifest.sha256` 記錄 149 個程式／設定檔，執行前核對通過。
- 遠端主 checkout 先 pull 至 `e62d658`；實驗快照固定為 `/share/home/r15543056/MDgen/pr19-fix-integrated-green-20261023/source`，輸出另存同一 run 的 `results/`。
- 2 CPU、12G、10 分鐘上限，不申請 GPU；明確停用 CUDA。環境為 Python `3.11.12`、PyTorch `2.11.0+cu128`，`python/3.11.12-gpu` module、`/share/home/r15543056/.venvs/mdgen`。

保留主線的共用 `gather_training_region`、探針 diagnostics 與模式相容性檢查。
僅在資料集讀取 metadata 後修正角度 mask，clone 後處理，避免改動傳入的 item。
兩個訓練入口的 region overfit 驗證使用獨立固定 seed 資料集；非 region 模式不變。

以 `git diff e62d658...ae7c28a` 做兩個獨立 reviewer 的 Standards／Spec 審查，兩軸均 0 項發現。
Python syntax 與 `git show --check` 通過；本分支未配置靜態型別檢查器。
8 個 skip 是既有 CUDA 測試。本次只驗證 CPU 行為，不作 GPU 容量或效能宣稱。
文字 logs／JUnit 僅移除行末空白；遠端保留原始產物。摘要見 `summary.json`，launcher 見 `submit.sh`。
