# MDGen Neighbor（基於 mdgen2026）

此分支保留 mdgen2026 的 global SDPA 預設，新增來自 sibling mdgen 的 local 空間計算與省顯存推論。操作、驗證與量測限制見 [整合說明](docs/NEIGHBOR_INTEGRATION.md)，固定範圍見 [spec](docs/specs/neighbor-integration.md)。歷次實作與量測報告見 [docs/reports](docs/reports/README.md)。目前環境見下方 Installation；HPC 上的 module＋venv 設定見整合說明。原論文模型、資料與授權保留如下。

Implementation of [Generative Modeling of Molecular Dynamics Trajectories](https://arxiv.org/abs/2409.17808) by Bowen Jing*, Hannes Stark*, Tommi Jaakkola, and Bonnie Berger.

We introduce generative modeling of molecular trajectories as a paradigm for learning flexible multi-task surrogate models of MD from data. By conditioning on appropriately chosen frames of the trajectory, such generative models can be adapted to diverse tasks such as forward simulation, transition path sampling, and trajectory upsampling. By alternatively conditioning on part of the molecular system and inpainting the rest, we also demonstrate the first steps towards dynamics-conditioned molecular design. We validate these capabilities on tetrapeptide simulations and show initial steps towards learning trajectories of protein monomers. Methodological details and further evaluations can be found in the paper. Please feel free to reach out to us at bjing@mit.edu, hstark@mit.edu with any questions.

**Note:** This repository is provided for research reproducibility and is not intended for usage in application workflows.

![mdgen.png](mdgen.png)

## Installation

Current environment (Python 3.11, PyTorch 2.11; versions match `docs/validation/neighbor/environment.json`):

```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128  # or .../whl/cpu
pip install -r requirements.txt
```

訓練入口 `train.py` 與 `train-runtime.py` 支援 `--precision fp32-reference`（預設：FP32、`highest` matmul、TF32 關閉）及 `--precision bf16-train`（BF16 autocast、FP32 master weights、`high` matmul、TF32 開啟）。亦接受 Lightning 別名 `32-true` 與 `bf16-mixed`。每次訓練／驗證啟動時，`MODEL_DIR/log.out` 記錄實際的 autocast dtype、參數 dtype、matmul 精度及 CUDA matmul／cuDNN 的 TF32 狀態。此設定尚不代表通過 Precision parity 驗證。

`python -m mdgen.train_step_probe --output probe.json --configs fp32-reference bf16-train -- <train.py 參數>` 以同一 batch、權重與 seed 跑一次完整 training step，輸出峰值記憶體、時間、loss、每個參數群的梯度摘要與 cosine、NaN／Inf 計數；詳見 `docs/validation/train-step-probe.md`。

Run the CPU test suite with `pip install pytest==8.4.2 && python -m pytest tests -q`; the same checks run in GitHub Actions (`.github/workflows/tests.yml`). The analysis scripts additionally need PyEMMA; see the conda environments in `envs/environment-mdgen2026-*-pyemma.yml`.

Original paper environment (historical, PyTorch 1.12):

```
pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
pip install wandb dm-tree einops torchdiffeq fair-esm pyEMMA
pip install matplotlib==3.7.2 numpy==1.21.2
```

## Datasets

1. The tetrapeptide MD datasets (explicit and implicit solvent) are available on HuggingFace: https://huggingface.co/datasets/bjing-mit/tetrapeptide-sims

2. Download the ATLAS simulations via https://github.com/bjing2016/alphaflow/blob/master/scripts/download_atlas.sh to `data/atlas_sims`.
3. Preprocess the tetrapeptide simulations
```
# Forward simulation and TPS, prep with interval 100 * 100fs = 10ps
python -m scripts.prep_sims --splits splits/4AA.csv --sim_dir data/4AA_sims --outdir data/4AA_data --num_workers [N] --suffix _i100 --stride 100

# Upsampling, prep with interval 100fs
python -m scripts.prep_sims --splits splits/4AA_implicit.csv --sim_dir data/4AA_sims_implicit --outdir data/4AA_data_implicit --num_workers [N]

# Inpainting, prep with interval 100fs 
python -m scripts.prep_sims --splits splits/4AA.csv --sim_dir data/4AA_sims --outdir data/4AA_data --num_workers [N]
```
4. Preprocess the ATLAS simulations
```
# Prep with interval 40 * 10 ps = 400 ps
python -m scripts.prep_sims --splits splits/atlas.csv --sim_dir data/atlas_sims --outdir data/atlas_data --num_workers [N] --suffix _i40 --stride 40
```

## Training

Commands similar to these were used to train the models presented in the paper.
```
# Forward simulation
python train.py --sim_condition --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv --data_dir data/4AA_data/ --num_frames 1000 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 --wandb --run_name [NAME]

# Interpolation / TPS
python train.py --tps_condition --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv --data_dir data/4AA_data/ --num_frames 100 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 --wandb --run_name [NAME]

# Upsampling
python train.py --sim_condition --train_split splits/4AA_implicit_train.csv --val_split splits/4AA_implicit_val.csv --data_dir data/4AA_data_implicit/ --num_frames 1000 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 20 --val_repeat 25 --cond_interval 100 --batch_size 8 --epochs 10000 --wandb --run_name [NAME]

# Inpainting / design
python train.py --inpainting --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv --data_dir data/4AA_data --num_frames 100 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 100 --val_repeat 25 --batch_size 32 --design --sampling_method euler --epochs 10000 --frame_interval 10 --no_aa_emb --no_torsion --wandb --run_name [NAME]

# ATLAS
python train.py --sim_condition --train_split splits/atlas_train.csv --val_split splits/atlas_val.csv --data_dir share/data_atlas/ --num_frames 250 --batch_size 1 --prepend_ipa --crop 256 --val_repeat 25 --epochs 10000 --atlas --ckpt_freq 10 --suffix _i40 --wandb --run_name [NAME]
```

## Model weights

The model weights used in the paper are available on HuggingFace: https://huggingface.co/bjing-mit/mdgen

## Inference

Commands similar to these were used to obtain the samples analyzed in the paper.
```
# Forward simulation
python sim_inference.py --sim_ckpt forward_sim.ckpt --data_dir share/4AA_sims --split splits/4AA_test.csv --num_rollouts 10 --num_frames 1000 --xtc --out_dir [DIR]

# Interpolation / TPS
python tps_inference.py --sim_ckpt interpolation.ckpt --data_dir share/4AA_sims --split splits/4AA_test.csv --num_frames 100 --suffix _i100 --mddir data/4AA_sims  --out_dir /data/cb/scratch/share/results/0506_tps_1ns 

# Upsampling 
python upsampling_inference.py --ckpt upsampling.ckpt --split splits/4AA_implicit_test.csv --out_dir outpdb/0505_100ps_upsampling_3139 --batch_size 10 --xtc --out_dir [DIR]

# Inpainting / design for high flux transitions
python design_inference.py --sim_ckpt inpainting.ckpt --split splits/4AA_test.csv --data_dir data/4AA_data/ --num_frames 100 --mddir data/4AA_sims --random_start_idx --out_dir [DIR] 

# Inpainting / design for random transitions
python design_inference.py --sim_ckpt inpainting.ckpt --split splits/4AA_test.csv --data_dir data/4AA_data/ --num_frames 100 --mddir data/4AA_sims --out_dir [DIR] 

# ATLAS forward simulation # note no --xtc here!
python sim_inference.py --sim_ckpt atlas.ckpt --data_dir share/data_atlas/ --num_frames 250 --num_rollouts 1 --split splits/atlas_test.csv --suffix _R1 --out_dir [DIR]
```

## Analysis

We run analysis scripts that produce a pickle file in each sample directory.
```
# Forward simulation
python -m scripts.analyze_peptide_sim --mddir data/4AA_sims --pdbdir [DIR] --plot --save --num_workers 1

# Interpolation / TPS
python -m scripts.analyze_peptide_tps --mddir data/4AA_sims --data_dir data/4AA_sims  --pdbdir [DIR] --plot --save --num_workers 1 --outdir [DIR]

# Upsampling
python -m scripts.analyze_upsampling --mddir data/4AA_sims_implicit --pdbdir [DIR] --plot --save --num_workers 1

# Inpainting / design
python -m scripts.analyze_peptide_design --mddir data/4AA_sims --data_dir data/4AA_data --pdbdir [DIR]
```
To analyze the ATLAS rollouts, follow the instructions at https://github.com/bjing2016/alphaflow?tab=readme-ov-file#Evaluation-scripts.

Tables and figures in the paper are extracted from these pickle files.

## License

MIT. Additional licenses may apply for third-party source code noted in file headers.

## Citation
```
@misc{jing2024generativemodelingmoleculardynamics,
      title={Generative Modeling of Molecular Dynamics Trajectories}, 
      author={Bowen Jing and Hannes Stärk and Tommi Jaakkola and Bonnie Berger},
      year={2024},
      eprint={2409.17808},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2409.17808}, 
}
```
