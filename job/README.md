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
| `MDGEN_ATLAS_CKPT` | `submit_mdgen_l7500_*.sh`、`submit_halo_error.sh`、`submit_generation_profile.sh` | ATLAS checkpoint 路徑 |
| `MDGEN_HALO_DATA`、`MDGEN_HALO_SPLIT` | `submit_halo_error.sh` | NPY 目錄與 split CSV |
| `MDGEN_RESOURCE_ROOT` | 否 | 含 `ckpt/`、`data/`、`splits/` 的資源目錄，預設 `../mdgen`（與 `scripts/diagnostics` 相同） |
| `MDGEN_EXPECTED_GPU_UUID` | 否 | `submit_neighbor_scaling.sh` 會檢查 GPU UUID 是否相符；未設定時只印出 UUID |

本機提交一律用 `scripts/hpc_run.py`（經 WSL 的 ssh／scp）：建立唯讀快照、核對 SHA-256、把 `src/results`、`src/logs` 連到 run 目錄後提交，並把每個 job 記到 `<run>/job_history`。用法見 `python scripts/hpc_run.py -h`，例如：

```bash
python scripts/hpc_run.py submit --tag pytest job/submit_pytest.sh tests/test_halo_error.py
python scripts/hpc_run.py submit --run <run_id> --env MDGEN_ATLAS_CKPT=... job/submit_halo_error.sh --pdb_id 3ilw_A
python scripts/hpc_run.py status <run_id>
python scripts/hpc_run.py fetch <run_id> <本機目錄> results/halo_error_<job>/halo_error.json logs/<job>.out
```

`job/env.sh` 會清掉從 SSH 繼承的 `SSH_CLIENT`（否則 `module load` 會載入 `~/.bashrc` 而在 compute node 失敗），並在工作目錄唯讀且未設 `MODEL_DIR` 時，把 `MODEL_DIR` 指到 `$MDGEN_RUN_DIR/workdir/<job>`。

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


## Training region 訓練步量測（#9 Ticket 3）

`submit_training_region_probe.sh` 以 `mdgen.train_step_probe` 對 5 層 `--local_spatial` ATLAS 組態（`--sim_condition --prepend_ipa --num_frames 250 --batch_size 1`）跑一次 `fp32-reference` training step；輸入是從 7,500 殘基合成鏈切出的一個 Training region（預設 `--region_residues 1024 --halo_rings 1`）。報告寫到 `results/training_region_probe_<job>/probe.json`，log 末尾印出 region、峰值記憶體與 step 時間。OOM 時 job 以非零結束，traceback 即為結果。

```bash
mkdir -p logs
MDGEN_VENV=/path/to/venv sbatch job/submit_training_region_probe.sh
# 其他 train.py 參數直接附在後面，例如層間 checkpointing 或不同 region 大小：
MDGEN_VENV=/path/to/venv sbatch job/submit_training_region_probe.sh --grad_checkpointing --region_residues 512
```

合成鏈每幀為獨立隨機漫步，loss 數值無意義，只用於記憶體與時間量測。

## Halo 近似誤差（#12 Ticket 5）

`submit_halo_error.sh` 以 `mdgen.full_length.halo_error` 比較 `--halo_rings 1 2` 與完整蛋白在 loss core 上的模型輸出差（同一雜訊、t、loss group）。報告寫到 `results/halo_error_<job>/halo_error.json`，log 末尾印出每種設定與每個邊界圈數的相對誤差。方法與欄位見 `docs/validation/halo-error.md`。

```bash
mkdir -p logs
MDGEN_VENV=/path/to/venv MDGEN_ATLAS_CKPT=/path/to/atlas.ckpt \
MDGEN_HALO_DATA=/path/to/npy MDGEN_HALO_SPLIT=/path/to/split.csv \
sbatch job/submit_halo_error.sh --pdb_id NAME --replicate 1 --region_residues 1024 --regions 8
```

蛋白須讓完整蛋白 forward 裝得下 GPU；`--partition_dir` 可沿用 `mdgen.full_length.partition_store` 的存檔。

## 生成速度 profiling（#9 Ticket 8）

`submit_generation_profile.sh` 以 `mdgen.full_length.generation_profile` 量測一段完整長度 FP32 rollout（`--local_spatial`，ODE 方法沿用 checkpoint 的 `sampling_method`）的 wall-clock 與 NFE，並以 torch.profiler 分析單次 forward 的 kernel 數、GPU 忙碌比例與 local group 發出的 launch。預設為合成 7,500 殘基鏈；報告寫到 `results/generation_profile_<job>/profile.json`。方法與欄位見 `docs/validation/generation-profile.md`。

```bash
mkdir -p logs
MDGEN_VENV=/path/to/venv MDGEN_ATLAS_CKPT=/path/to/atlas.ckpt sbatch job/submit_generation_profile.sh
# 比較不同 group 批次大小，或改用真實蛋白：
MDGEN_VENV=/path/to/venv MDGEN_ATLAS_CKPT=/path/to/atlas.ckpt sbatch job/submit_generation_profile.sh --group_batch_size 16
```
