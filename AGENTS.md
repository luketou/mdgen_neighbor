# Agent 導航

- 開始工作時先確認實際 repo／worktree、分支與 dirty diff；讀取 [README](README.md) 的專案入口。
- GitHub 目的 repo 為 `luketou/mdgen_neighbor`；先以 `git remote -v` 確認本機 remote 名稱。
- 安裝 CPU 開發環境或驗證修改時，依 [開發流程](docs/agents/development.md) 使用本機／CI 共用指令。
- 開發與審查，以及撰寫 commit、PR 或 issue 時，遵循[開發與寫作規範](docs/agents/conventions.md)。
- 審查時一併套用 [Coding Standards](CODING_STANDARDS.md)；parity 基準位於 [global2026](tests/reference/global2026) 與 [localbaseline](tests/reference/localbaseline)。
- GPU 驗證紀錄位於 [docs/validation](docs/validation)；L40S 測試入口為 [job/submit_neighbor_tests.sh](job/submit_neighbor_tests.sh)。
- 處理 issue／ticket 時，讀取 [issue tracker](docs/agents/issue-tracker.md) 與 [triage labels](docs/agents/triage-labels.md)。
- 涉及領域術語或架構決策時，先讀[領域文件入口](docs/agents/domain.md)。
- 發現或選擇外部工具、`gh` 不可用時，依[工具查找與備援](docs/agents/tools.md) 操作。

- HPC 診斷、module 環境與最小資源設定見 [HPC 操作入口](docs/validation/hpc/README.md)。
