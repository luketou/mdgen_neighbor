---
name: hpc-workflow
description: HPC（Slurm）job 的提交與查詢：部署程式快照、smoke test、大量分片提交、查佇列與失敗原因、取回結果，以及把 server 上的數字存成驗證紀錄。
---

# HPC 工作流程

本機到 HPC 一律經 `scripts/hpc_run.py`（`submit`／`status`／`tail`／`fetch`／`history`），用法見 [job/README.md](../../../job/README.md)。叢集限制與已踩過的坑在 [HPC 操作入口](../../../docs/validation/hpc/README.md)：分區、priority、空間、WSL worktree 的提交方式，每一步動手前先對照。

## 步驟

1. **接手**：`python scripts/hpc_run.py history` 找出先前的 run ID 與 Job ID，確認 worktree、分支與 dirty diff。
   完成：知道哪些 job 還在跑、哪些已結束、各自對應哪個快照。

2. **本機檢查**：在 WSL 跑 `bash scripts/check.sh`。快照只帶 tracked 檔，以及 `job/`、`scripts/`、`mdgen/`、`hpc/` 底下的 untracked 檔；新的測試先 commit。
   完成：exit 0。

3. **Smoke**：用正式 launcher 與正式入口、最小輸入送一個 job（`submit --tag <名稱> job/<launcher>.sh ...`）。
   完成：job 離開佇列；`.out` 開頭有拿到 GPU（不是 `No devices were found`）；產物 JSON 的狀態欄位是成功；量到單 case 時間與輸出大小。只拿到 Job ID 不算完成。

4. **估算**：以 smoke 的單 case 時間與輸出大小乘上矩陣規模，對照 `df -h /share/home/r15543056`、launcher 的時間上限與可用 GPU 數。
   完成：總輸出放得下、單 case 不會逼近時間上限；放不下就先和使用者確認存法或縮減範圍。

5. **正式提交**：沿用通過 smoke 的同一快照（`submit --run <run_id>`）；程式有變更就建立新快照並重做 smoke。只用 `l40s` 分區；大量 job 加 `--sbatch=--nice=10000`，讓其他工作先跑；少量需要優先的用 nice 0。逐筆記錄分片 → Job ID；任何一筆 submit 失敗或結果不明就停，先查 `history` 與 `squeue`，從中斷處續送。
   完成：預期數量全部有 Job ID，`squeue` 的 pending 原因是 `Priority` 或 `Resources`。

6. **監看**：遠端查詢先寫成腳本檔，再以 `wsl -d Ubuntu -e bash -c "ssh <target> 'bash -s' < <腳本>"` 執行。結束的 job 先讀 `logs/<job>.out`／`.err`；逾時會出現 `DUE TO TIME LIMIT`。最長的設定（最大幀數、最長蛋白）第一批跑完就核對時間。
   完成：知道完成數、失敗數與各自原因；有逾時風險時趁還在排隊取消重送。

7. **取回與紀錄**：用 `fetch` 只取小型 JSON 與 log（單次上限 20 MiB），大型張量留在 server。README 引用的每個 server 數字（空間、檔案大小、雜湊、佇列狀態）都存成同目錄的 log，指令與原始輸出一起寫入。
   完成：每個數字都能在同目錄找到出處（`CODING_STANDARDS.md` 的規定），`check.sh` 通過。

## 失敗與授權

OOM、沒拿到 GPU、逾時、程式錯誤與 `submission_unknown` 分開記錄並保留 log；沒拿到 GPU 不是 OOM，不當成量測結果。同一快照重送時，在 run 目錄的 `job_history` 補記先前失敗或取消的 job 與原因。

恢復訓練建立新 run，指定確切的來源 checkpoint 並記錄父 run；完成標準是成功繼續一步，不是只載入模型。

`hpc/project.yaml` 的資源設定是候選值，不是授權；清理或覆寫既有 run、checkpoint、結果，以及超出使用者已授權範圍的資源，都先取得使用者明確同意。
