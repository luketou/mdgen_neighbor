# Ticket 5：成對訓練比較工具

本次只完成程式與 CPU 工具驗證，沒有提交 GPU Job、部署、push 或開 PR。真實蛋白的 loss／梯度／更新與資源結果仍未完成，不能宣稱訓練近似通過。凍結設定見[契約 v1.1](../43-contract/frozen-contract-v1.1.md)及其引用的 v1；本文件不修改契約。

數值來源：下列計畫 CLI／模型設定對應同目錄的 [離線設定紀錄](planned-configuration.json)，候選規模對應[矩陣摘要](planned-matrix.json)及[代表設定](raw-tensor-cases.json)；容量量測另逐項引用伺服器 log。

## 成對路徑

正式入口 `python -m mdgen.train_step_probe --paired_region --output NEW.json --seed_group GROUP --seed NOISE --time T --warmup 1 --repeats 3 -- <訓練參數>` 從完整蛋白輸入建立 Training region。僅支援 batch 1、local spatial、dropout 0、固定 t、velocity 的 masked squared error；不用不同 supervision 的 capacity probe 或 forward 代替梯度。

完整輸入先經官方 Dataset 轉換，再用 `assemble_region` 選取並 pad；因此 torsion 已包含前一殘基資訊。兩側使用完整蛋白第零幀的同一分區、全蛋白位置索引，以及同一 loss-group core 的有效通道與非條件幀。先產生完整 noise，再依全蛋白索引選 Training region。逐 case 檢查 latent、有效 mask、noise hash 與分母一致；零分母回報 `no_supervision`，不執行 backward／更新。

共同可行種子以兩圈預算選 loss groups，再各加一／兩圈 halo；兩圈放不下時，一圈保留自己的選取結果並標記 `paired_common_subset: false`。改變 halo 不會改變共同子集的 loss groups。這個受控探針不呼叫一般 training startup 的全種子預算檢查，也不宣稱一般 startup 可用。

沿用 `run_step` 的官方 Lightning training step、Adam、全域 norm clipping 與實測 optimizer 更新。每次重建相同初始權重與空的 Adam 狀態，保存初始權重 SHA-256、optimizer defaults、clipping 前梯度、clipping norm／係數及更新後參數有限性。相同路徑控制用 `--control_type same_path_repeat`；`paired` 在完整涵蓋時是 full／Training region 控制，在截斷時只有描述性比較。

控制套用契約選項 B，逐群判定，近零群組的相對差、norm ratio、cosine 為 null。截斷 case 的 `status: descriptive` 與 `control_passed: null` 不代表訓練近似通過。非有限值仍使 case 失敗。

每個分片內各 case 都是獨立子程序；同一成對 case 的兩側、暖機與重複量測在同一程序、同一 GPU 執行。時間在 training batch 起訖同步 CUDA；包含梯度複製至 CPU 的探針成本，不是純 kernel 時間。每側記錄 peak allocated／reserved，CPU 為 null。暖機與量測均由相同初始狀態起步，不連續訓練。指定保存的原始梯度與更新留在 server，勿整包取回大型 `.pt`。

成對 probe 與 matrix `run`（含 launcher 的 `--paired-shard`）都接受 `--raw-tensors {all,none}`，預設是 `none`；`all` 保存暖機與所有重複量測的原始 `.pt`，必須明確指定。`none` 只省略原始 `.pt`；逐參數群梯度／更新摘要、比較、控制判定與每側 `.json` 證據仍完整保存，`artifact_path` 為 null。此選項不改變計算或凍結矩陣。

[儲存預估](storage-estimate.json)是審查提供的下限估算，並非實測：`all` 每 case 至少 2.04 GiB，全部 1,620 個可執行 case 至少 3.22734375 TiB，未含序列化、JSON／log 與重跑。

[伺服器證據](server-evidence-20261024.log)於 `2026-10-24T10:10:47Z` 擷取：`df -h` 顯示 `/share/home` 可用 `104G`；smoke Job 49597 的張量目錄 `du -sh --apparent-size` 為 `2.3G`，`du -sb --apparent-size` 為 `2423638209` bytes（含目錄，不將 `G` 誤寫為十進位 GB）。全部保存放不下，因此正式執行採預設 `--raw-tensors none`，只有[代表 case 清單](raw-tensor-cases.json)明確選取的 case 使用 `all`。不同設定的張量大小尚未逐案量測，不提供代表集合的總容量估算；該清單保存實際代表設定。

## 正式執行的資料與 checkpoint manifest

Run `issue43-t5-formal-20261024T045815Z-f2a135d` 的完整 data-root 與 checkpoint SHA-256 清單見[伺服器證據](server-evidence-20261024.log)。以下離線逐檔對照 [Ticket 1 紀錄](../43-49427/summary.json)的 `outputs[].files`／`checkpoint`；資料根目錄的 `summary.json` 則直接對照該 Ticket 1 紀錄檔的 SHA-256。未在 Ticket 1 記錄逐檔 hash 的輔助檔明示無法判定；沒有新增 runtime hashing。

| 檔案（相對 data-root；checkpoint 除外） | 與 Ticket 1 比對 |
| --- | --- |
| `arrays/6l3r_E_R1_i400ps_chain.npz` | match |
| `arrays/6l3r_E_R1_i400ps.npy` | match |
| `arrays/6l3r_E_R2_i400ps_chain.npz` | match |
| `arrays/6l3r_E_R2_i400ps.npy` | match |
| `arrays/6l3r_E_R3_i400ps_chain.npz` | match |
| `arrays/6l3r_E_R3_i400ps.npy` | match |
| `arrays/6lrd_A_R1_i400ps_chain.npz` | match |
| `arrays/6lrd_A_R1_i400ps.npy` | match |
| `arrays/6lrd_A_R2_i400ps_chain.npz` | match |
| `arrays/6lrd_A_R2_i400ps.npy` | match |
| `arrays/6lrd_A_R3_i400ps_chain.npz` | match |
| `arrays/6lrd_A_R3_i400ps.npy` | match |
| `arrays/6sms_A_R1_i400ps_chain.npz` | match |
| `arrays/6sms_A_R1_i400ps.npy` | match |
| `arrays/6sms_A_R2_i400ps_chain.npz` | match |
| `arrays/6sms_A_R2_i400ps.npy` | match |
| `arrays/6sms_A_R3_i400ps_chain.npz` | match |
| `arrays/6sms_A_R3_i400ps.npy` | match |
| `arrays/6sup_A_R1_i400ps_chain.npz` | match |
| `arrays/6sup_A_R1_i400ps.npy` | match |
| `arrays/6sup_A_R2_i400ps_chain.npz` | match |
| `arrays/6sup_A_R2_i400ps.npy` | match |
| `arrays/6sup_A_R3_i400ps_chain.npz` | match |
| `arrays/6sup_A_R3_i400ps.npy` | match |
| `checkpoint_metadata.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6l3r_E_R1_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6l3r_E_R1_i400ps_partition.npz` | match |
| `partitions/6l3r_E_R2_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6l3r_E_R2_i400ps_partition.npz` | match |
| `partitions/6l3r_E_R3_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6l3r_E_R3_i400ps_partition.npz` | match |
| `partitions/6lrd_A_R1_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6lrd_A_R1_i400ps_partition.npz` | match |
| `partitions/6lrd_A_R2_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6lrd_A_R2_i400ps_partition.npz` | match |
| `partitions/6lrd_A_R3_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6lrd_A_R3_i400ps_partition.npz` | match |
| `partitions/6sms_A_R1_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6sms_A_R1_i400ps_partition.npz` | match |
| `partitions/6sms_A_R2_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6sms_A_R2_i400ps_partition.npz` | match |
| `partitions/6sms_A_R3_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6sms_A_R3_i400ps_partition.npz` | match |
| `partitions/6sup_A_R1_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6sup_A_R1_i400ps_partition.npz` | match |
| `partitions/6sup_A_R2_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6sup_A_R2_i400ps_partition.npz` | match |
| `partitions/6sup_A_R3_i400ps_drift.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `partitions/6sup_A_R3_i400ps_partition.npz` | match |
| `partitions/summary.json` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `summary.json` | match |
| `validation.csv` | 無 Ticket 1 逐檔 hash，無法判定 match／mismatch |
| `atlas.ckpt` | match |

Checkpoint 的完整 SHA-256 為 `bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015`，同時符合 Ticket 1 與凍結契約。

## 分片與 JSON schema

`python -m mdgen.train_step_matrix manifest --output NEW.json --cases-per-shard 1` 產生完整矩陣，ID 由 case 欄位／分片 case ID 清單的 SHA-256 決定，不依執行順序或結果變動。調整粒度不改 case ID；每個分片可作為一個獨立 Job。矩陣涵蓋[代表設定](raw-tensor-cases.json)及[計畫摘要](planned-matrix.json)列出的 replica、t／noise seed／幀層、halo 與控制。

[計畫摘要 JSON](planned-matrix.json)記錄預設粒度：6,642 筆候選紀錄，1,620 個可執行分片；4,536 筆完整涵蓋重複候選映射至同 halo 的第一個候選，另有 486 筆長蛋白 250 幀記為 `phase1_infeasible`，附既有 OOM 證據、不重跑。長蛋白兩圈僅列核准的 group 95；其餘兩圈候選的不可行原因保留於 `selection_inventory`，不補抽。基準每個 halo 保留第一個完整涵蓋候選；基準分別執行兩個 halo 的控制。重複與不可行紀錄不配置 GPU 分片，也不以其他結果補值。

各介面使用 `schema_version: 1`：

| 產物 | 主要欄位 |
| --- | --- |
| manifest | `manifest_id`、`contract_version`、`contract_sha256`、`inventory_sha256`、`selection_inventory`、`cases`、`shards` |
| case | `case_id`、protein／replica／frames／t／noise_seed／halo／group／control_type、`coverage`、`paired_common_subset`、`plan_status`、必要的 `duplicate_of`／OOM `evidence` |
| shard | `manifest_id`、`shard_id`、唯一 `run_id`、`slurm_job_id`、`checkpoint_sha256`、`status`、`cases` |
| 成對 probe | `alignment`（兩側 hash、分母、loss／Training region 全蛋白索引、t）、`warmup`、`measured_repeats`、`measurements`、`status` |
| 每次量測 | `warmup`、`full`／`candidate`（loss 分子／分母、精度、時間、顯存、nonfinite、初始權重 hash、optimizer／clipping／缺梯度）、`comparison`（loss 差、逐群梯度與更新 norm／RMS／cosine／near-zero／控制判定） |
| 原始 tensor | 預設 `probe/ITER-SIDE.pt` 的 `pre_clip_gradients`（逐參數）、`updates`（逐參數群）；同名 `.json` 保留已完成側證據 |
| 失敗 | `failure_category`：`oom`／`missing_gpu`／`timeout`／`program_error`；`failure.json` 保存 traceback、stage、side、iteration、可取得的峰值；case record 保存 exit code、命令、log 路徑 |
| 彙整 | `status`、`missing_shards`、`strata`（逐層 counts、原始 records、`statistics.metrics` 的 median／p95／max／有效量測數、`statistics.counts`、`statistics.failed_diagnostics`、去重與 loss 殘基聯集比例） |

**契約限制**：training loss 與參數群梯度皆為整個固定 loss-residue 集合的總量，不能分解成逐邊界圈數結果。`boundary_ring: all_loss_residues` 記錄此限制，並非逐邊界圈數的量測；本次不修改契約檔案。彙整另依蛋白、replica、幀、t、noise seed、halo、完整涵蓋／截斷、控制類型與共同子集分層，不作跨蛋白 pooled 通過判定。量測重複次數不當獨立空間樣本。失敗與 `no_supervision` 個別計數；長蛋白若出現重複 loss-group 集合，以 `deduplicated: true` 與 `duplicate_of` 標記，保留原 `status`／失敗證據並排除統計。完成但失敗的控制保留在 `failed_diagnostics`，不混入通過／描述性樣本；zero、one-sided-zero、near-zero 與 nonfinite 使用明確 counts，非有限數值不進入分位數。環境設定失敗停止 launcher 並保留 `paired-setup-*/setup.log`／`setup-failure.txt`；只有結構化 `missing_gpu` preflight 失敗交給分片保存分類。

## 後續提交指令（本次未執行）

先在目前 worktree 的 WSL 執行本機 manifest 產生，這不會連線或提交：

```bash
PY=/home/luketou/.local/share/mdgen/venvs/cpu-py311/bin/python
"$PY" -m mdgen.train_step_matrix manifest --output results/t5-manifest.json
```

以下為後續操作範例，數值為計畫參數而非新量測；模型參數見[離線設定紀錄](planned-configuration.json)，提交節流上限見[儲存預估](storage-estimate.json)的 `submission_in_flight_cap_example`。

目前 `scripts/hpc_run.py` 是 Windows 端呼叫 WSL ssh／scp 的工具；以下在此 worktree 的 **Windows PowerShell** 執行。它會保存同一份不可寫程式快照；後續所有 Job 用 `--run` 共用該快照。先執行一個基準 4 幀分片作為 launcher smoke，核對 Job 與 JSON；`deployment_ready: false` 的正式 launcher／資料路徑尚未經本分支 GPU 驗證。

```powershell
$manifest = Get-Content results/t5-manifest.json -Raw | ConvertFrom-Json
$smokeCase = $manifest.cases | Where-Object {
    $_.protein -eq '6sms_A' -and $_.frames -eq 4 -and
    $_.control_type -eq 'paired' -and $_.plan_status -eq 'ready'
} | Select-Object -First 1
$smoke = $manifest.shards | Where-Object { $_.case_ids -contains $smokeCase.case_id }
$data = '/share/home/r15543056/MDgen/mdgen/data/atlas_validation_ticket1_20261023b'
$checkpoint = '/share/home/r15543056/MDgen/mdgen/ckpt/atlas.ckpt'
$reply = & py -3 scripts/hpc_run.py submit --tag issue43-t5-smoke `
    job/submit_training_region_probe.sh --paired-shard --shard-id $smoke.shard_id `
    --data-root $data --checkpoint $checkpoint
if ($LASTEXITCODE -ne 0) { throw '提交未知或失敗，依 job_history 查核，勿自動重送' }
$reply
$runId = [regex]::Match(($reply -join "`n"), 'submitted job_id=\d+ run_id=(\S+)').Groups[1].Value
if (!$runId) { throw '未取得 run ID' }
$runId | Set-Content results/t5-run-id.txt
py -3 scripts/hpc_run.py status $runId
py -3 scripts/hpc_run.py tail $runId
```

Job 結束時 stdout 最後印出獨立 `.../results/paired/SHARD/RUN/shard.json` 路徑；用 `fetch` 取回該檔、核對 `status`、控制與實際 precision／GPU。硬 Slurm 逾時可能來不及寫 `shard.json`。未來彙整可傳 `--scheduler-log SHARD_ID=STDERR_PATH`（可重複指定）；缺片且該 stderr 出現 `DUE TO TIME LIMIT` 才分類為 `timeout` 並保存路徑及原文，否則維持 `missing_shard`。程序內 case timeout 記錄 非 null 的 `exit_code`（Python timeout 以 SIGKILL 終止子程序）、最後已落盤的 `stage`／side／iteration 與來源；無可讀進度則記 `unknown` 及原因，同時寫入 case record 與 `failure.json`。已完成側 JSON 只能證明最後完成的 step，無法還原正在執行的階段。這些診斷只適用於使用修正版的未來執行，不回填目前 formal run 的既有產物。

```powershell
# 以 stdout 的確切 SHARD/RUN 替換，取回小型 JSON；不要取整個含梯度的目錄。
py -3 scripts/hpc_run.py fetch $runId results/t5-collected results/paired/SHARD/RUN/shard.json
```

Smoke 完成並通過後，用以下迴圈分批提交其餘分片，各一個獨立 Job。範例採人工節流：同時 pending／running 上限為 4 個；每批達上限後停止提交，核對本批全部 Job 已結束，再輸入 `completed` 繼續。執行期間不要從其他終端向同一 run 提交，以維持上限。同一分片重跑會產生新的 UUID 子目錄，不覆寫舊結果。連線或提交未知時迴圈停止，先查 `job_history`／排程器，勿整個迴圈重跑。

```powershell
$maxInFlight = 4
$inFlight = 0
foreach ($shard in $manifest.shards) {
    if ($shard.shard_id -eq $smoke.shard_id) { continue }
    if ($inFlight -ge $maxInFlight) {
        py -3 scripts/hpc_run.py status $runId
        if ($LASTEXITCODE -ne 0) { throw '查詢失敗，停止提交' }
        $confirmed = Read-Host '核對排程器與 log，確認本批全部結束後輸入 completed；否則停止'
        if ($confirmed -ne 'completed') { throw '尚未確認容量，停止提交' }
        $inFlight = 0
    }
    py -3 scripts/hpc_run.py submit --run $runId --tag $shard.shard_id `
        job/submit_training_region_probe.sh --paired-shard --shard-id $shard.shard_id `
        --data-root $data --checkpoint $checkpoint
    if ($LASTEXITCODE -ne 0) { throw "停止提交：$($shard.shard_id)，先核對狀態" }
    $inFlight++
}
```

較粗粒度需重新產生 `--cases-per-shard N` 的 manifest，並在每個 launcher 指令加 `--cases-per-shard N`。也可用 `--manifest` 指定 Compute Node 可讀的同一份 manifest；未指定時從凍結契約重新產生，hash 必須相同。

取回每個分片的確切 `shard.json` 後，在 WSL 彙整（每片只選一個明確 run；工具拒絕多個 run，不挑最好結果）。沒有取回的片保留為未完成：

```bash
"$PY" -m mdgen.train_step_matrix aggregate --manifest results/t5-manifest.json \
  --output results/t5-report.json --results results/t5-collected/results/paired/*/*/shard.json
```

仍需 GPU 完成：launcher smoke、三個基準所有控制、`6sup_A` 的 4／64 幀固定可行候選及同路徑控制。250 幀長蛋白沿用契約的第一期不可行紀錄，不以較短幀或 forward 填補。控制容差外推至其他 replica／t／seed、空 Adam 單步不代表恢復訓練、資料獨立性及 checkpoint 長度／時間外推限制照契約保留。

## CPU 驗證

[CPU 驗證紀錄](cpu-validation.json)保存實際指令、Python／PyTorch 與 pass／skip 計數；[SDPA 原始 JSON](cpu-sdpa.json)保存獨立比對結果。

```bash
MDGEN_CPU_VENV=/home/luketou/.local/share/mdgen/venvs/cpu-py311 bash scripts/check.sh
```

原始提交紀錄：271 passed、0 failed、2 skipped、14 deselected；獨立 CPU SDPA 6 組通過。測試涵蓋真正截斷／完整涵蓋、分母不等、零監督、非有限值、固定 halo loss groups、控制相對容差、原始梯度／更新保存、實際 backward hook 注入 OOM、preflight 缺 GPU、分片穩定性／完整性、重跑不覆寫及彙整缺口。這些都是 CPU 工具測試，沒有真實 GPU 資源或蛋白數值驗收結果。

先前審查修正的 [CPU 紀錄](review-fix-validation.json)與 [SDPA 結果](review-fix-sdpa.json)：獨立完整 CPU pytest 為 290 passed、0 failed、2 skipped、14 deselected，SDPA 為 6 組通過；新增反例先重現失敗再修正。指定的 `scripts/check.sh` 退出碼仍為 2：工作樹與 index 的空白檢查通過，但歷史 `903971b` 的 README EOF 空白仍被 `HEAD^1..HEAD` 檢查擋下。該段為先前驗證紀錄；本輪結果見下方。

本輪[完整 CPU 檢查紀錄](review-20261024-validation.json)與[原始 log](review-20261024-check.log)：指定的 `scripts/check.sh` 退出碼為 0，305 passed、2 skipped、14 deselected；[獨立 SDPA 結果](review-20261024-sdpa.json)為 6 組通過。[逾時／排程反例](review-20261024-red.log)及[具名結構／有限性反例](review-20261024-red-api.log)保存修正前的失敗。未提交 HPC Job、commit 或 push。
