# GPU 驗證：#2 Ticket 4 與 #3 Ticket 5（2026-10-15）

本紀錄回答兩個問題：

1. #2 Ticket 4（local 層改為覆寫空間運算方法）在 GPU 上，global／local 的輸出與梯度是否與凍結基準一致。
2. #3 Ticket 5（`job/*.sh` 路徑參數化）的新提交方式能否在 Slurm 上正常執行。

結論：兩者在 **RTX 3080** 上皆通過。L40S 尚未執行（見「尚未完成」）。

## 執行環境

| 項目 | 值 |
|---|---|
| Slurm job | 48342，`COMPLETED`、`ExitCode=0:0`，運行 26:48 |
| 節點／GPU | node-171，NVIDIA GeForce RTX 3080 10 GB，UUID `6f4d0e3a-816a-522a-921f-62ee5cfadde7` |
| 程式版本 | `refactor/issue-2-local-global` @ `aa740ba` |
| 軟體 | modules `python/3.11.12-gpu cuda/12.8`；torch `2.11.0+cu128`；float32、TF32 關閉 |
| checkpoint | ATLAS `atlas.ckpt`，SHA256 `bc90c66e…aff08015` |
| `mdgen/rigid_utils.py` | SHA256 `25b1ce32…4290d5`，`rot_to_quat` 含 `isfinite` |

L40S 分區（node-201 與 `llm` 分區的 node-181）四張卡皆被 12 小時的 job 佔用，Slurm 預估最早 10-15 23:06 才能開始，因此先改用 rtx3080 分區。原 L40S job 48330 仍在排隊。

提交指令（以 sbatch 參數覆寫分區與時間，未修改腳本）：

```bash
MDGEN_VENV=/share/home/r15543056/.venvs/mdgen sbatch \
  --job-name=neighbor-tests-3080 --partition=rtx3080 --account=l40s \
  --gres=gpu:1 --cpus-per-task=2 --mem=20G --time=00:40:00 \
  --output=logs/neighbor_tests_3080_%j.out --error=logs/neighbor_tests_3080_%j.err \
  --wrap 'if [ -n "$GPU_DEVICE_ORDINAL" ]; then export CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ORDINAL; fi; bash job/submit_neighbor_tests.sh'
```

`--wrap` 中的 `CUDA_VISIBLE_DEVICES` 設定即為本次加入 `job/env.sh` 的修正，見下文。

## #2 Ticket 4：輸出與梯度對凍結基準

### pytest

`tests/test_neighbor_integration.py`、`tests/test_ode_endpoint_storage.py`、`tests/test_pretrained_local_core_only.py`：**51 passed**、7 warnings（皆為 `torch.jit.script` deprecation），8:44。CPU 上 skip 的 CUDA 測試在此全部執行。

其中顯存相關輸出：

- ODE 只存端點：峰值 173,993,984 → 157,865,984 bytes，NFE 20／20。
- 大批次推論：峰值 41,760,256 → 9,410,048 bytes（0.2253 倍）。

### 正式 checkpoint 對照（`validate_neighbor_checkpoint.py`）

104 殘基；梯度比較 4 frames，推論比較 250 frames；容差 atol 2e-5、rtol 2e-4。

| 模式 | 凍結基準 | 250 frames 輸出 | 4 frames 輸出 | 輸入梯度 | 權重梯度 | 結果 |
|---|---|---:|---:|---:|---:|---|
| global | `tests/reference/global2026` | 0 | 0 | 0 | 0 | passed |
| local | `tests/reference/localbaseline` | 0 | 7.87e-6 | 1.11e-9 | 2.33e-8 | passed |

數值為 max abs difference。global 與凍結基準逐位元相同；local 的 250 frames 推論逐位元相同，4 frames 梯度路徑的前向差異 7.87e-6 在容差內。

與重構前 L40S（job 48275，`docs/validation/neighbor/`）比較：local 輸入梯度 1.09e-10、權重梯度 2.61e-8，4 frames 前向為 0。本次輸入與權重梯度同量級；4 frames 前向的 7.87e-6 屬 3080 與 L40S 的 kernel 差異或重構造成，需 L40S 結果才能區分。

### SDPA 等價性（`test_sdpa.py`）

| 案例 | 輸出相對誤差 | 梯度相對誤差 | 結果 |
|---|---:|---:|---|
| residue_no_padding | 5.60e-7 | 7.92e-7 | pass |
| residue_padding | 5.12e-7 | 6.80e-7 | pass |
| time_no_padding | 3.75e-7 | 5.47e-7 | pass |
| time_padding | 3.56e-7 | 6.14e-7 | pass |
| time_full_row_padding | 3.70e-7 | 5.68e-7 | pass |
| no_rope_padding | 2.89e-7 | 8.99e-6 | pass |

容差：輸出 1e-5、梯度 1e-4。`no_rope_padding` 的 manual／SDPA padding invariance 皆為 0。

### ODE 端點保存（`validate_ode_endpoints.py`）

250 frames、104 殘基、dopri5（atol 1e-6、rtol 1e-3）。保存 50 點與 2 點的最終 atom14 **完全相同**（max abs 0），NFE 122／122，輸出全部 finite，shape `[1, 250, 104, 14, 3]`。L40S（job 48275）的 NFE 為 116，不同 GPU 的自適應步數不同屬預期。

### `sim_inference.py` CLI

5e3e_A，4 frames、1 rollout、seed 137。global 與 local（`--local_spatial --no-cell_summaries`）皆產出 `5e3e_A.pdb`、`5e3e_A.xtc`、`5e3e_A_atom14.npy`，無錯誤。此步僅確認流程可執行，不比較數值。

## #3 Ticket 5：新提交方式

| 檢查項目 | 結果 |
|---|---|
| 從 repository 根目錄提交，`job/env.sh` 通過 `mdgen/wrapper.py` 檢查 | 正常 |
| `MDGEN_VENV` 啟用 virtualenv、`module load python/3.11.12-gpu cuda/12.8` | 正常 |
| `MDGEN_RESOURCE_ROOT` 未設定時沿用 `../mdgen` 的 `ckpt/`、`data/`、`splits/` | 正常 |
| 以 sbatch 參數覆寫 `--partition`、`--time`、`--output` | 正常 |
| 腳本內無寫死的個人路徑或 `--nodelist` | 正常 |

發現的問題與本次修正：

1. **時間上限不足**：腳本原設 `--time=00:20:00`，3080 實際需 26:48，L40S 上是否足夠未經驗證。改為 `00:40:00`。
2. **node-201 第二張卡的 `CUDA_VISIBLE_DEVICES`**：node-201 的單 GPU job 分到第二張 L40S 時，Slurm 把 `CUDA_VISIBLE_DEVICES` 設為主機編號 1，但 job 內只看得到 device 0，CUDA 初始化失敗（`CUDA_ERROR_NO_DEVICE`，2026-10-09 於 rvalue 專案觀察到並以相同方式修正）。`job/env.sh` 改為在 `GPU_DEVICE_ORDINAL` 有值時以它設定 `CUDA_VISIBLE_DEVICES`。本次 job 在 node-171 執行，沒有實際觸發此問題。
3. `logs/` 需事先建立，否則 Slurm 無法寫 log（`job/README.md` 已註明）。

## 尚未完成

- **L40S 驗證**：Ticket 4 的 spec 要求 L40S。job 48330（以修正前的腳本提交）仍在排隊；建議取消後以修正後的腳本重送。
- 3080 與 L40S 的 kernel 不同，L40S 結果出來後應比對 local 4 frames 前向差異是否仍為 7.87e-6。

## 檔案

- `sdpa_summary.json`：`test_sdpa.py` 完整輸出
- `checkpoint_parity.json`：正式 checkpoint 的 global／local 對照
- `ode_endpoint_parity.json`：ODE 端點保存對照

完整 log（未提交）：`logs/neighbor_tests_3080_48342.{out,err}`；生成座標（未提交）：`results/neighbor_tests_48342/`。
