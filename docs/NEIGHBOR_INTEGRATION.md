# mdgen2026 local 整合說明

## 現在能做什麼

同一份 mdgen2026 可選 global 或 local 空間 attention。預設 global；使用 `--local_spatial --no-cell_summaries` 選 core/ghost 鄰域。兩者共用原生時間 SDPA、主層 FFN 推論分批、可設定 ODE 容差與座標重建。

- `mdgen/full_length/partition.py`：初始結構分區、core 唯一 ownership、ghost 與必要同鏈編號鄰居；容量不足拆 core，單殘基仍放不下則明確報錯。
- `mdgen/full_length/attention.py`：core-only Query、ghost Key/Value、按 group batch gather、全域位置 RoPE。
- `mdgen/full_length/local_ipa.py`：局部 IPA；這部分仍先 gather 全部 group，未宣稱也已逐批 gather。
- `mdgen/full_length/model.py`：2026 模型的 local subclass，重用原生時間 attention 與 FFN。
- `mdgen/full_length/checkpoint.py`、`sim_inference.py`：相容權重載入與報告；CLI 拒絕非 summary 缺失、不相容 shape 或額外權重。
- `mdgen/wrapper.py`：建立初始固定分區；僅 eval/no_grad/dopri5 保存 ODE 起終點。`num_frames` 仍是生成軌跡的長度。
- `mdgen/rigid_utils.py`：CUDA 大批次 4×4 `eigh` 每批最多 16384，避免 cuSOLVER 批次限制。

Global 的 dataset、前處理、runtime、主層 FFN／全域 IPA 分批與 manual fallback 保留。Local 不支援 `manual_attention`、`no_frames` 或自訂 `position_ids`。Summaries 預設關閉；打開未訓練 summary 不代表品質改善。未移植 geometry loss，未新增 chemistry encoder 或顯式 ligand 模型。

## 環境與簡單執行

本輪量測使用 PyTorch `2.11.0+cu128`；local 需要提供 `scaled_dot_product_attention` 的 PyTorch。README 下方原論文的舊版安裝指令是歷史環境，不適用於這次 local 整合。本 HPC 使用既有 modules 加 venv，不另建 Conda 環境：

```bash
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu cuda/12.8
source /share/home/r15543056/.venvs/mdgen/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
```

Checkpoint 與 NPY 不放 Git。可使用原 README 的官方下載位置；本次既有資源在 sibling `../mdgen`。先用真實 104 殘基系統跑少量 frames：

```bash
python sim_inference.py \
  --sim_ckpt ../mdgen/ckpt/atlas.ckpt \
  --data_dir ../mdgen/data/atlas_data \
  --split ../mdgen/splits/mini_atlas_all.csv \
  --pdb_id 5e3e_A --suffix _R1_i40 \
  --num_frames 4 --num_rollouts 1 --inference_seed 137 \
  --local_spatial --no-cell_summaries \
  --ode_atol 1e-6 --ode_rtol 1e-3 \
  --xtc --save_atom14 --out_dir results/local_smoke
```

Global 對照省略 `--local_spatial --no-cell_summaries`；若 checkpoint 本身為 local 可明確指定 `--no-local_spatial`。在其他機器修改資源與環境路徑；Slurm job 的提交方式、環境變數與重現同卡比較所需的 node／GPU UUID 見 [job/README.md](../job/README.md)。

## 可重現驗證

```bash
python -m pytest tests/test_neighbor_integration.py \
  tests/test_pretrained_local_core_only.py \
  tests/test_ode_endpoint_storage.py tests/test_spatial_partition.py -q
python scripts/diagnostics/test_sdpa.py --device cuda --output_dir results/sdpa
export MDGEN_RESOURCE_ROOT=/path/to/resource_directory
python scripts/diagnostics/validate_neighbor_checkpoint.py --outdir results/checkpoint
python scripts/diagnostics/validate_ode_endpoints.py --outdir results/ode
```

資源根目錄包含 `ckpt/atlas.ckpt`、`data/atlas_data/5e3e_A_R1_i40.npy`、`splits/mini_atlas_all.csv`；未設定環境變數時採用 sibling `../mdgen`。`splits/mini_atlas_all.csv` 隨 repo 提供，可複製到外部資源根目錄的 `splits/`；只需準備其中 `5e3e_A` 的 NPY。若 checkpoint 與資料直接放在本 repo 的 `ckpt/`、`data/`，設定 `MDGEN_RESOURCE_ROOT="$PWD"`。需要 CUDA 的測試在無 GPU 時 skip，不能把 CPU-only 結果當作完整驗證。

凍結參考位於 `tests/reference/global2026` 與 `localbaseline`；`manifest.json` 記錄整合前兩邊來源 SHA256，基準模型不使用候選的模型原始碼。額外 frozen helper 用於 core-only 與 ODE 保存比較。正式 checkpoint SHA256：`bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015`。

已驗證非零模型權重的輸出、輸入梯度、權重梯度及 train/eval fallback。正式 checkpoint：global 與整合前 2026 輸出／梯度完全一致；local 與整合前 mdgen 輸出完全一致，job 48275 的輸入梯度最大差異約 `1.09e-10`，權重梯度約 `2.61e-8`。50 點改 2 點的最終 atom14 完全一致，NFE 都是 116；這些是指定 fixture 的實測結果，非所有輸入的數學保證。

## 大型資源量測

`job/submit_neighbor_scaling.sh` 在同一實體 L40S 分別測 global/local、L=5000/7500、250 frames 的 forward 與完整 generation。每例獨立 Python process；FP32、TF32 off、seed 137、官方 checkpoint、FFN chunk 65536、dopri5、atol=1e-6、rtol=1e-3、summary off。forward 為 warmup 2 次後 5 次的中位數；完整生成一次，包含 GPU 前處理、ODE 與 atom14 重建，不含權重載入與 fixture CPU→GPU 搬移。

```bash
MDGEN_VENV=/path/to/venv MDGEN_EXPECTED_GPU_UUID=a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb \
  sbatch --nodelist=node-201 job/submit_neighbor_scaling.sh
python scripts/diagnostics/summarize_neighbor_scaling.py results/neighbor_scaling_JOBID
```

每例 JSON 記錄 source/checkpoint/input SHA、GPU UUID、時間、allocated/reserved 顯存、NFE、finite 及失敗階段；audit 核對共同條件。**Fixture 是把 5e3e_A 重複放在相隔的 3D lattice，作工程容量測試；成功不表示真實 7500 殘基蛋白或 protein+ligand 的物理品質已驗證。** 此 fixture 的 `chain_id` 用於分區；既有 `atom37_to_torsions` 仍沿連續 tensor 編號計算，沒有跨 chain 斷鍵處理。因而它也不是多鏈生物構型驗證。Global/local 鄰域不同，不能把兩者差異解讀為數值誤差。

主要套件版本記錄於 `docs/validation/neighbor/environment.json`（null 表示沒有 distribution metadata；分析環境未納入本輪推論驗證）。

`docs/reports/` 中既有 `LOCAL_*`、`ODE_ENDPOINT_STORAGE_REPORT.md` 等報告保留先前 sibling mdgen 的歷史量測；此次共同基底的結果以本輪驗證紀錄為準。

本輪測試、量測與 review 的最終摘要記錄於 `docs/NEIGHBOR_VALIDATION.md`。
