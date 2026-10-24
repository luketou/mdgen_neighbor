# HPC 診斷與執行

Ticket 2 的 [L40S 實測報告](../l40s-ticket2/2026-10-23/README.md) 記錄 GPU preflight、完整小尺寸訓練步與兩個阻塞的修正證據。

固定設定見 [hpc/project.yaml](../../../hpc/project.yaml)。`deployment_ready: false` 保留：GPU preflight 與小尺寸訓練探針已通過，但正式模型 launcher、容量與資料路徑尚未驗證。

2026-10-22 的 [CUDA 診斷](2026-10-22-cuda-diagnosis.md) 記錄 job 49048/49051 的實際證據。當時 node-201 分配 GPU 0 的裝置存取回傳 EIO；不能據此判定另一張 GPU 或其他節點的狀態，也不能推論目前健康狀態。

## GPU readiness

`python scripts/diagnostics/gpu_preflight.py --output report.json` 使用目前解譯器做真正的 2×2 CUDA matmul、同步並驗證結果。CPU runner 無 GPU 時輸出 `status: skipped`、`passed: false`，退出碼為 0。`--require-gpu` 或 `SLURM_JOB_GPUS` 有值時，無法使用 CUDA 會輸出失敗報告與非零退出碼。報告亦記錄解譯器、PyTorch 版本與分配裝置變數。

`runtime.env` 中除 `MDGEN_MODULES` 之外的設定會以 shell quoting 匯出到 job。此 server 的 Python GPU module 含 DeepSpeed，模型匯入時會探測 CUDA 工具鏈；`CUDA_HOME` 指向已核對的 CUDA 12.8 module 路徑，避免 `MissingCUDAException`。設定檔只放非機密環境值。

`job/env.sh` 在啟用指定 venv 後，呼叫相同 preflight 並強制 GPU；報告預設 `results/preflight_JOBID.json`，可用 `MDGEN_PREFLIGHT_OUTPUT` 指向 run 的獨立輸出目錄。既有模型 launcher 在 preflight 失敗後停止。CPU 共用檢查入口也會執行可略過模式。

## 快照與 Slurm 工具

本機提交 launcher、查詢狀態與取回結果一律用 `scripts/hpc_run.py`（經 WSL 的 ssh／scp），用法見 [job/README.md](../../../job/README.md) 與 `python scripts/hpc_run.py -h`。

### 已踩過的坑

- Windows 端 `ssh` 會無輸出逾時（exit 255）；改走 WSL：`wsl -d Ubuntu -e bash <腳本>`（預設 distro 是 `docker-desktop`，沒有 `bash`）。`iam` 只是 `ssh r15543056@zinfandel.ncree.org` 的 alias，非互動 shell 不會展開。遠端指令含引號時寫成腳本檔再執行，不內嵌在 `wsl ... -c "..."` 裡。
- `sacct` 的資料庫連不上；查 job 用 `squeue -j`（結束後會回 `Invalid job id`），再讀 `logs/*.out`／`.err`。
- 結束的 job 先查 `.out` 開頭：`nvidia-smi` 印出 `No devices were found` 代表沒拿到 GPU，不是 OOM，不可當作量測結果；同一快照重送時在 run 目錄的 `job_history` 記下失敗的 job。
- `hpc_run.py` 只上傳 `job/`、`scripts/`、`mdgen/`、`hpc/` 底下的未追蹤檔；新的 `tests/` 檔案要先 commit，否則 pytest 找不到。
- 在 WSL 建立的 worktree，Windows 的 git 讀不到（`git ls-files` exit 128），`hpc_run.py submit` 在打包快照前就失敗。從 Windows 提交前先設 `$env:GIT_DIR='<主 repo>\.git\worktrees\<目錄名>'` 與 `$env:GIT_WORK_TREE='<worktree 的 Windows 路徑>'`。
- 契約用的單張 L40S 只有 `l40s` 分區（node-201，2 張，常有一張被其他使用者佔用）。`llm` 分區（node-181）只收 `llm` account，本帳號的 Job 會掛在 `users` 而永遠 pending（`Job's account not permitted`）；`sbatch --test-only` 不會發現這個限制。
- 排程器是 `priority/basic`：先送先跑。一次送大量 Job 時，送出時加 `--sbatch=--nice=10000`，讓之後送的其他工作先跑；需要優先的少量 Job 用 nice 0。一般使用者只能調高 nice；`scontrol update Priority=` 不會持久保存，且無法再調回。
- `/share/home` 約 360 GB、2026-10-24 剩約 105 GB，與其他使用者共用。會寫大檔的矩陣（例如每 case 2.3 GB 的原始梯度）送出前先估算總量。

### 較早的 preflight 診斷工具 `scripts/hpc.py`

只用於下列固定的 GPU preflight 診斷，不用於提交 launcher。在 repo 根目錄執行，設定檔僅含路徑與非機密選項；SSH 使用既有金鑰／agent。

```bash
.venv-cpu/bin/python scripts/hpc.py prepare --files scripts/diagnostics/gpu_preflight.py
# 從回傳 JSON 取得 state_file，代入下列 STATE。
.venv-cpu/bin/python scripts/hpc.py submit --state STATE
.venv-cpu/bin/python scripts/hpc.py status --state STATE
.venv-cpu/bin/python scripts/hpc.py collect --state STATE
```

- `prepare` 明列 tracked code；必要未追蹤檔另用 `--extra-files`。依設定排除資料、模型與憑證，限制每檔大小，保留指向已選檔案的相對 symlink。傳輸至唯一新目錄，SHA-256 核對並將 code 檔案與目錄設為唯讀後才標記 `ready`。保留 commit、dirty、檔案清單、生成 launcher 雜湊與設定。
- `submit` 僅執行已設定的最小 GPU preflight，重新核對快照後以設定資源投遞；Slurm 保存的 launcher 在 job 啟動時再次驗證 payload，排隊期間內容遭改動會停止。launcher 明確進入 snapshot/code，使用 module 與絕對 Python 路徑，輸出位於 snapshot 旁的 results/logs。此工具不會啟動正式訓練；`deployment_ready: false` 不阻擋已明確設定的診斷。
- 提交前先持久保存 `submission_unknown` 與時間；回覆有效 Job ID 後轉為 `submitted`。連線失敗或逾時保留未知狀態並阻止重送。`status` 依唯一 job name 查詢可找回仍在 queue 的 Job ID；已消失且缺乏產物時保留未知，須人工核對管理端紀錄，不自動重送。
- 每次查詢使用 SSH 總逾時；工具單次查詢即返回。輪詢請至少間隔設定的 30 秒。queue 消失本身不代表成功；只有 run 的退出碼為 0 且 JSON `passed: true` 才標示檢查 `passed`。這是探針結果，不是 `sacct` 的 Slurm 最終狀態或模型驗證結論。
- `collect` 只讀取固定的 preflight JSON、exit-status 與 log tail（100 行），總下載上限 20 MiB。驗證失敗回傳非零；結果不明也回傳非零。`active` 表示仍在排程器中，不代表驗證通過。
- 後續設定或程式變動建立新快照；現有 state 要搭配原設定。保留失敗快照供查核，工具不清除舊實驗或 checkpoint。
