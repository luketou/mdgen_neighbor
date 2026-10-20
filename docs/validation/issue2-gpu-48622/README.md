# L40S 驗證：#2 Ticket 4（2026-10-20）

補上 `docs/validation/issue2-gpu-48342/` 留下的 L40S 驗證。

結論：**L40S 上全部通過**。global 與凍結基準逐位元相同；local 前向逐位元相同，梯度差異在容差內。

## 執行環境

| 項目 | 值 |
|---|---|
| Slurm job | 48622，分區 `l40s`（node-201），`--time=01:30:00` |
| GPU UUID | `a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb` |
| 程式版本 | `refactor/issue-2-local-global` @ `25eef76` |
| checkpoint | ATLAS `atlas.ckpt`，SHA256 `bc90c66e…aff08015` |
| 精度 | float32，TF32 關閉 |

提交指令：

```bash
MDGEN_VENV=/share/home/r15543056/.venvs/mdgen sbatch --time=01:30:00 job/submit_neighbor_tests.sh
```

先前的 L40S job 48330 的 pytest 已通過（51 passed），但後續步驟撞到腳本預設的 40 分鐘時限被取消，因此以較長時限重跑。

## 結果

- pytest：**51 passed**，7 warnings（`torch.jit.script` deprecation），10:35。
- 正式 checkpoint 對照（`checkpoint_parity.json`，104 殘基，容差 atol 2e-5、rtol 2e-4）：

| 模式 | 凍結基準 | 250 frames 輸出 | 4 frames 輸出 | 輸入梯度 | 權重梯度 | 結果 |
|---|---|---:|---:|---:|---:|---|
| global | `tests/reference/global2026` | 0 | 0 | 0 | 0 | passed |
| local | `tests/reference/localbaseline` | 0 | 0 | 1.09e-10 | 1.01e-8 | passed |

- ODE 只存端點（`ode_endpoint_parity.json`）：250 frames 輸出逐位元相同，NFE 116／116，皆有限。
- SDPA 等價性（`sdpa_summary.json`）：5 種 padding 情境的輸出相對誤差 ≤ 5.6e-7、梯度相對誤差 ≤ 8.7e-7，皆有限。
- `sim_inference.py` global／local 各產生 4 frames 的 pdb／xtc／atom14。

## 與 RTX 3080（job 48342）比較

3080 上 local 4 frames 前向差異為 7.87e-6；L40S 上為 0，與重構前 L40S（job 48275）一致。

如 `issue2-gpu-48342/README.md` 所述，3080 上的比較是凍結基準與候選模型在同一張卡上進行，因此不能單純歸因於兩種 GPU 的 kernel 不同；較可能是開梯度路徑在 3080 上選到不同的實作（例如 #2 Ticket 1 讓 local 層經過 `torch.utils.checkpoint`）。L40S 上差異為 0，表示 Ticket 4 的重構在 L40S 上沒有改變數值；3080 差異的確切來源本次未再追查。
