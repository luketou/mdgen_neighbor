# PR #19 後續修正：最新主線整合驗證

三項 P2 修正接上主線 `8469b2a` 後，完整 CPU suite 通過。

- Slurm job `49232`：`COMPLETED`、`ExitCode=0:0`，`l40s` partition/account、`node-201`。
- Training region 測試 **23 passed**；完整 suite **152 passed、8 skipped**。結果見 JUnit、原始輸出與 `summary.json`。
- 修正前對照 job `49231` 使用主線 `8469b2a` 的 `mdgen/dataset.py` 搭配相同新測試：CSV／鏈資訊檔各自的跨鏈及編號斷點共 **4 failed**。另外兩項修正的 red → green 證據見 `../issue15-review-fixes-49060/`。
- 執行快照固定於 `/share/home/r15543056/MDgen/pr19-fix-topology-final-20261023/source`，輸出在同一 run 的 `results/`。遠端主 checkout 先 pull 至 `8469b2a`。
- `snapshot.json` 記錄打包當時 HEAD `8c9664f` 與 dirty 狀態；新增測試隨後提交為 `b0c9972`。156 個來源／設定檔的 SHA 在遠端執行前核對，亦逐檔確認與本地 `b0c9972` 內容一致。
- 使用 2 CPU、12G、10 分鐘上限，未申請 GPU且停用 CUDA；Python `3.11.12`、PyTorch `2.11.0+cu128`，`python/3.11.12-gpu` module 與 `/share/home/r15543056/.venvs/mdgen`。

保留主線鏈資訊 NPZ 優先權、共用 region gather、順序／coremask、診斷與模式檢查。角度遮罩修正在 metadata 選定後套用，涵蓋 NPZ 與 CSV，並保留原 item。兩個訓練入口在 region overfit 時建立固定 seed 的獨立驗證資料集。

兩個獨立 reviewer 對 `8469b2a...b0c9972` 執行 Standards／Spec 審查，兩軸皆 0 項發現。Python syntax 與 Git whitespace 檢查通過；未配置靜態型別檢查器。8 個 skip 為既有 CUDA 測試，本次不作 GPU 效能或容量宣稱。

本目錄是最新整合結果；`../issue15-review-integrated-49063/` 為較早 Ticket 4 整合快照。文字輸出與 JUnit 僅移除行末空白，遠端保留原始產物。
