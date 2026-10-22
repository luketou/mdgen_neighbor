# Slurm job 腳本

所有 `job/*.sh` 都從 repository 根目錄提交，並透過 `job/env.sh` 載入 modules 與 virtualenv。Slurm 以提交時的目錄作為工作目錄；`logs/` 需事先建立。

```bash
mkdir -p logs
MDGEN_VENV=/path/to/venv sbatch job/submit_neighbor_tests.sh
```

## 環境變數

| 變數 | 必要 | 用途 |
| --- | --- | --- |
| `MDGEN_VENV` | 是 | 要啟用的 virtualenv |
| `MDGEN_MODULES` | 否 | 要 `module load` 的 modules，預設 `python/3.11.12-gpu cuda/12.8`；設為空字串則略過 |
| `MDGEN_ATLAS_CKPT` | `submit_mdgen_l7500_*.sh` | ATLAS checkpoint 路徑 |
| `MDGEN_RESOURCE_ROOT` | 否 | 含 `ckpt/`、`data/`、`splits/` 的資源目錄，預設 `../mdgen`（與 `scripts/diagnostics` 相同） |
| `MDGEN_EXPECTED_GPU_UUID` | 否 | `submit_neighbor_scaling.sh` 會檢查 GPU UUID 是否相符；未設定時只印出 UUID |

`job/env.sh` 會在 `GPU_DEVICE_ORDINAL` 有值時把 `CUDA_VISIBLE_DEVICES` 設為它：`node-201` 的單 GPU job 若分到第二張卡，Slurm 會把 `CUDA_VISIBLE_DEVICES` 設成主機編號 1，但 job 內只看得到 device 0，導致 `CUDA_ERROR_NO_DEVICE`。

`--partition=l40s` 與 `--account=l40s` 仍寫在腳本中；其他叢集可在提交時覆寫，例如 `sbatch --partition=gpu --account=myacct job/...`。

## 重現既有量測

#1 的同卡比較固定在 `node-201` 的同一張 L40S。重現時於提交指令指定節點並設定 UUID：

```bash
MDGEN_VENV=/share/home/r15543056/.venvs/mdgen \
MDGEN_EXPECTED_GPU_UUID=a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb \
sbatch --nodelist=node-201 job/submit_neighbor_scaling.sh
```

原本各腳本使用的 virtualenv：`submit_neighbor_*.sh` 為 `/share/home/r15543056/.venvs/mdgen`；`submit_flrh_*.sh` 與 `submit_mdgen_l7500_*.sh` 為 `/share/home/r15543056/.venvs/mdgen2026`，l7500 的 checkpoint 為 `/share/home/r15543056/.cache/mdgen/atlas.ckpt`。

## 共用 GPU 健康檢查

`job/env.sh` 在 module 與 venv 載入後，使用 venv 的絕對 Python 路徑執行 GPU preflight；CUDA 不可用或矩陣運算失敗即停止。最小獨立診斷、不可變快照與 Slurm 投遞工具見 [HPC 操作入口](../docs/validation/hpc/README.md)。
