---
name: hpc-workflow
description: 部署本地程式至 HPC、執行遠端 smoke test、提交訓練或模擬、恢復 checkpoint，以及查詢實驗結果時使用。
---

# HPC 工作流程

## 確認目標

從目標 repo 根目錄讀取 `AGENTS.md`、`hpc/project.yaml` 及本次 launcher；相對路徑以 repo 根目錄解析。`scripts/hpc.py` 讀取已支援的診斷設定；正式訓練與 checkpoint 欄位尚未自動執行。

確認實際 repo/worktree、分支與 dirty diff。讀取 `existing` 所列腳本與文件，選擇本次實驗的環境；同一專案可能有多個環境。`deployment_ready: false` 表示尚未完成遠端路徑與 launcher 相容性驗證，不能直接套用舊 launcher 送出新快照。

SSH 使用 `ssh.target` 與設定的選項；`iam` 只是互動式 shell alias。先以不修改檔案的短連線確認路徑、排程器與 Compute Node 可見的共享儲存。SSH 失敗時回報實際原因，繼續可做的本地工作；需要互動登入時使用既有終端流程。

只處理會阻止本次工作的缺漏資訊；先查文件與腳本，再向使用者詢問。檔案中的候選路徑與舊資源設定需核對，不視為已驗證或已授權。

## 建立程式快照

1. 先做本次修改適用的本地檢查，使用專案指定入口與環境。
2. 從 Git tracked 檔案中選取執行需要的程式與設定；另外明列必要的未追蹤程式。依 `sync.exclude` 排除產物與大型檔案，檢查單檔大小並預覽清單；超過上限的必要資產改用明確的遠端路徑。
3. 使用唯一 run ID，在 `remote.runs_root` 下的新暫存目錄傳輸；只操作本次新建的目的地。保留 symlink，不使用跟隨連結的傳輸方式；檢查其目的地在遠端有效。同步不使用 `--delete`。
4. 在本地為實際部署內容產生 SHA-256 manifest，記錄 commit、dirty 狀態與必要的未追蹤程式；Git diff 若要保存，先排除資料與憑證。傳輸後檢查遠端雜湊，完整才標記為可提交。
5. 程式快照固定；可寫入的 logs、cache、results 與 checkpoint 使用獨立路徑。對會寫入 `results/` 等相對目錄的 launcher，明確設好輸出路徑或專用 symlink。

完成條件：快照內容一致，執行所需資源在 Compute Node 可讀，寫入位置不改動快照程式。

投遞前檢查 launcher 的 `#SBATCH --chdir`、腳本內的 `cd`、repo 環境變數、相對資料路徑及環境位置。命令列的 `--chdir` 無法抵消腳本內另一次 `cd`；以小型 Job 確認實際執行的程式路徑。新快照不能暗中使用舊 repo 的程式。既有 launcher 若已有可靠快照機制，沿用它並記錄其實際快照及產物位置。

## Smoke test

選用相同訓練／模擬入口與環境，透過已支援參數限制資料、step、時間與資源；資源來自已確認的 smoke 設定。先建立 Slurm stdout/stderr 所需目錄。GPU 測試與真實模型載入都在 Compute Node 執行。

依變更驗證資料讀取、模型載入、forward/backward/optimizer，以及最小輸出。涉及 checkpoint 或恢復時，再驗證寫入、重新載入與繼續一步；純數值模擬以該 case 的數值驗證與產物檢查替代。既有 GPU 驗證腳本可能較昂貴，不直接當作最小 smoke。

完成條件：排程器顯示成功結束，且所需產物及專案的驗證門檻通過。只取得 Job ID 代表提交成功；小資料測試不證明正式資料量的記憶體與效能需求。

## 正式投遞與斷線處理

使用通過 smoke 的同一份程式快照，另存正式參數。記錄資料／模型識別、環境或 lockfile 識別、資源與來源 checkpoint。若更動程式，建立新快照並重做相關驗證。

提交前持久保存 run ID、唯一 Job 名稱或 comment、提交時間及狀態。取得 `sbatch --parsable` 回覆後保存 Job ID；沿用 wrapper 時確認它如何回傳及保存 ID。設定中的作業資源是需求或候選值，不是增加配額的授權。

提交中斷且結果不明時，先依 run ID、Job 名稱/comment、使用者與提交時間查 `squeue` 及可用的 `sacct` 紀錄。仍無法辨識時標示 `submission_unknown`，保留資訊並停止自動重送。SSH 斷線不等於 Job 失敗。

## Checkpoint 與大檔

資料、模型及 checkpoint 以明確遠端路徑引用；本地只留合成或允許使用的小樣本。資料掃描、統計及前處理送 CPU Job，再取回摘要。使用 node-local SSD 時由 Job 暫存必要資料，重要狀態寫回共享儲存。

區分推論權重與完整訓練恢復狀態。依框架檢查模型、optimizer、scheduler、step、隨機狀態及所有分片。寫入採框架支援的完整性機制：同一檔案系統暫存後 rename，或所有分片完成後寫完成標記；只恢復已完成 checkpoint。

恢復建立新 run，指定確切來源 checkpoint 並記錄父 run；檢查框架的 max steps 是總步數還是額外步數。完成條件是恢復進度正確且成功繼續，而非只成功載入模型。

保存失敗診斷與重要里程碑；清理、覆寫既有實驗或 checkpoint 依明確使用者授權執行。共享儲存不代表有備份，依儲存政策另存需長期保留的結果。

## 查詢與交付

按需查詢或以至少設定間隔輪詢。SSH／傳輸使用有限逾時；`ServerAliveInterval` 用於偵測失聯，不是遠端命令的總時限。必要時另設本地命令總逾時。logs 使用指定檔案的有限尾端，下載依明確清單與大小上限。

回報 run ID、Job ID、快照與結果路徑、實際狀態、已通過及未執行的驗證。失敗時保留錯誤摘要與診斷檔，依授權處理修正；只有終止狀態與產物檢查都符合條件，才宣告實驗通過。
