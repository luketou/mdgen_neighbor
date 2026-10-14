# MDGen-2026 r2.1 Review Reply

本文件記錄 `mdgen2026-r2` 至 `mdgen2026-r2.1` 的修改、驗證與回覆。

## R2 — C19 conda 環境建立 log

目的：使用兩份包含 PyEMMA 的環境鎖定檔，分別建立全新的 CUDA 12.6 與 CUDA 13.0 conda 環境，並保存完整的環境建立輸出，作為環境可重現性的證據。

```bash
conda env create \
  --name mdgen2026-r2-a6-cu126 \
  --file my_new/environment-mdgen2026-cu126-pyemma.yml \
  2>&1 | tee my_new/r2/C19_M12-cu126-conda-create.log

conda env create \
  --name mdgen2026-r2-a6-cu130 \
  --file my_new/environment-mdgen2026-cu130-pyemma.yml \
  2>&1 | tee my_new/r2/C19_M12-cu130-conda-create.log
```
保存產物：
(a) `my_new/r2/C19_M12-cu126-conda-create.log`
(b) `my_new/r2/C19_M12-cu130-conda-create.log`

執行結果：兩份 C19 conda 建立 log 已保存並納入 commit `8cb3fa2`。

## R3 — 大型產物移出 Git 追蹤
處置：將 `*.npy`、`*.npz`、`*.pdb`、`*.pickle` 加入 `.gitignore`，並以 `git rm --cached` 取消既有大型產物的 Git 追蹤。此操作只修改 Git index，不刪除工作目錄中的實體檔案；JSON、log、txt、code 與 CSV 均繼續保留於 repository。
取消追蹤清單摘要：
(a) `.npy`：52 個
(b) `.npz`：26 個
(c) `.pdb`：2 個
(d) `.pickle`：1 個
原始檔保存位置：81 個實體檔保留於執行主機的原始實驗工作目錄 `/mnt/hdd/jeff/mdgen-piezo/model/mdgen2026/`；這些檔案由 `.gitignore` 排除，只保留於工作目錄，不納入 repository 追蹤。
執行狀態：Git index 整理與實體檔保留檢查已完成；Git 已不再追蹤上述四種大型產物，原始實驗工作目錄仍保有 52 個 NPY、26 個 NPZ、2 個 PDB 與 1 個 pickle。R3 的 repository 整理由獨立 commit `0c87584` 完成，原始產物的保存位置亦已記錄於本報告。
### C40 — R3 repository 大型產物整理

```bash
git ls-files -z -- '*.npy' '*.npz' '*.pdb' '*.pickle' \
  | xargs -0 git rm --cached --

git add .gitignore
git ls-files -- '*.npy' '*.npz' '*.pdb' '*.pickle'
git status --short
git commit -m "R3: stop tracking generated artifacts"
```

## R4 — position_ids 檢查移到 MHA 入口

處置：將非 `None` 的 `position_ids` 檢查由 RoPE 實作移至 `MultiheadAttention.forward` 入口。RoPE／no-RoPE、manual／SDPA 收到非 `None` 的 `position_ids` 時，現在都會拋出 `NotImplementedError`，不再於 no-RoPE 路徑靜默忽略輸入。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/model/mha.py` | `position-id-rope-interface`（r1 已存在；r2.1 修改並保留） | 將非 `None` 的 `position_ids` 拒絕檢查移到 `MultiheadAttention.forward` 入口，使檢查不再依賴 RoPE 是否啟用。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；r2.1 修改並保留） | 在既有 RoPE manual／SDPA rejection case 之外，新增 no-RoPE manual／SDPA case；四條路徑收到非 `None` 的 `position_ids` 時都必須拋出 `NotImplementedError`。 |

### C41 — R4 position_ids CPU regression

```bash
conda activate mdgen2026-r2-a6-cu126
mkdir -p my_new/r2.1

python test_sdpa.py \
  --device cpu \
  --seed 137 \
  --output_dir my_new/r2.1/C41_R4-position-ids \
  2>&1 | tee my_new/r2.1/C41_R4-position-ids.log
```

執行結果：完成；summary 為 `passed=true`，RoPE／no-RoPE、manual／SDPA 四條路徑收到非 `None` 的 `position_ids` 時均成功拋出 `NotImplementedError`。

## R5 — 底層類別的 use_sdpa 預設改為 True

處置：將底層 attention 與 layer 類別的 `use_sdpa` 建構預設改為 `True`，使直接建構底層類別時也符合「未指定 attention 路徑即使用 SDPA」。`AttentionWithRoPE` 經由 `*args, **kwargs` 建構 `MultiheadAttention`，因此沿用 MHA 的新預設；上層 `NewMDGenWrapper` 與 `LatentMDGenModel` 在 r2 已預設使用 SDPA。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/model/mha.py` | `sdpa-route`（r1 已存在；r2.1 修改並保留） | 將 `MultiheadAttention.__init__` 的 `use_sdpa` 預設由 `False` 改為 `True`；`AttentionWithRoPE` 未明確傳值時會經 kwargs 使用此預設。 |
| `mdgen/model/latent_model.py` | `sdpa-route`（r1 已存在；r2.1 修改並保留） | 將 `IPALayer.__init__` 與 `LatentMDGenLayer.__init__` 的 `use_sdpa` 預設由 `False` 改為 `True`。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；本輪確認、不修改） | manual 組均已明確傳入 `use_sdpa=False`，SDPA 組均明確傳入 `use_sdpa=True`，不依賴底層預設值。 |

### C42 — R5 bottom-layer default SDPA CPU smoke test

```bash
conda activate mdgen2026-r2-a6-cu126
mkdir -p my_new/r2.1

python - <<'PY' 2>&1 | tee my_new/r2.1/C42_R5-default-sdpa.log
from mdgen.model.latent_model import AttentionWithRoPE, IPALayer, LatentMDGenLayer
from mdgen.model.mha import MultiheadAttention

embed_dim = 32
num_heads = 4
ipa_args = {
    "c_s": embed_dim,
    "c_z": 0,
    "c_hidden": 4,
    "no_heads": num_heads,
    "no_qk_points": 2,
    "no_v_points": 2,
    "dropout": 0.0,
}

mha = MultiheadAttention(embed_dim, num_heads)
attention_with_rope = AttentionWithRoPE(embed_dim, num_heads)
ipa_layer = IPALayer(
    embed_dim=embed_dim,
    ffn_embed_dim=4 * embed_dim,
    mha_heads=num_heads,
    ipa_args=ipa_args,
)
latent_layer = LatentMDGenLayer(
    embed_dim=embed_dim,
    ffn_embed_dim=4 * embed_dim,
    mha_heads=num_heads,
    num_frames=2,
)

checks = {
    "MultiheadAttention": mha.use_sdpa,
    "AttentionWithRoPE": attention_with_rope.attn.use_sdpa,
    "IPALayer": ipa_layer.use_sdpa and ipa_layer.mha_l.attn.use_sdpa,
    "LatentMDGenLayer": (
        latent_layer.use_sdpa
        and latent_layer.mha_t.attn.use_sdpa
        and latent_layer.mha_l.attn.use_sdpa
    ),
}

for name, uses_sdpa in checks.items():
    print(f"{name}: use_sdpa={uses_sdpa}")
    assert uses_sdpa is True

print("passed=true")
PY
```

執行結果：完成；四個底層建構檢查均為 `use_sdpa=True`，最後輸出 `passed=true`。

## R6 — 環境鎖定檔整理

處置：兩份包含 PyEMMA 的環境檔使用不會與 r1 環境衝突的名稱；兩份不含 PyEMMA 的舊環境檔於檔頭標記為「r1 歷史環境，勿用」。正式環境為 `my_new/environment-mdgen2026-cu126-pyemma.yml`；cu130 版本保留供 eigh／Blackwell 追蹤。

| file_path | block_name | 說明改動 |
|---|---|---|
| `my_new/environment-mdgen2026-cu126-pyemma.yml` | `runtime-environment`（r2 已存在；r2.1 修改並保留） | `name:` 設為 `mdgen2026-r2-a6-cu126`，避免未傳入 `conda env create --name` 時覆蓋 r1 環境；此檔為正式環境鎖定檔。 |
| `my_new/environment-mdgen2026-cu130-pyemma.yml` | `runtime-environment`（r2 已存在；r2.1 修改並保留） | `name:` 設為 `mdgen2026-r2-a6-cu130`，與 r1 cu130 環境區隔。 |
| `my_new/environment-mdgen2026-cu126.yml`、`my_new/environment-mdgen2026-cu130.yml` | —（r1 歷史環境） | 檔頭新增「MDGen-2026 r1 歷史環境，勿用」註解，保留既有內容供歷史結果重現。 |

執行結果：完成；環境檔整理主要由 commit `9a3e6e0` 完成，最終環境名稱配合 C19 實際建立名稱調整於 commit `d554581`。

## E1 — Trainer 的 weights_only=False 時序

C28 log 的執行時間早於 `weights_only=False` 進入 commit 的時間，僅由既有 log 無法重建當時未提交的 `train.py` 工作樹內容；因此不再以舊 C28 作為這兩行已實跑的證據，改由 r2.1 code 的 C43、C44 分別驗證 validate 與 fit resume 路徑。

| 編號 | 執行內容 | 驗證目的 |
|---|---|---|
| C43 | `train.py --validate --ckpt ckpt/atlas.ckpt`，限制一個 validation batch | 驗證 `Trainer.validate(..., weights_only=False)` 能載入舊 Lightning checkpoint。 |
| C44 | 先以 `train.py` 產生包含 optimizer state 的完整 Lightning checkpoint，再以 `train.py --ckpt` resume 並限制一個 training batch | 驗證 `Trainer.fit(..., weights_only=False)` 能恢復完整訓練狀態並完成一個 resumed training step。`atlas.ckpt` 僅含模型權重，不適用於 fit resume。 |

### C43 — E1 r2.1 validation checkpoint load

```bash
conda activate mdgen2026-r2-a6-cu126
mkdir -p my_new/r2.1

python train.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --atlas \
  --prepend_ipa \
  --num_frames 250 \
  --crop 256 \
  --batch_size 1 \
  --num_workers 0 \
  --validate \
  --val_batches 1 \
  --ckpt ckpt/atlas.ckpt \
  --run_name C43-E1-r21-validation \
  --model_dir workdir \
  2>&1 | tee my_new/r2.1/C43_E1-validation-checkpoint.log
```

執行結果：完成；成功由 `ckpt/atlas.ckpt` 載入模型權重並完成一個 validation batch，`val_loss=1.131419062614441`。

### C44 — E1 r2.1 one-step checkpoint resume

```bash
conda activate mdgen2026-r2-a6-cu126
mkdir -p my_new/r2.1

python train.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --atlas \
  --prepend_ipa \
  --num_frames 250 \
  --crop 256 \
  --batch_size 1 \
  --num_workers 0 \
  --train_seed 137 \
  --no-deterministic \
  --epochs 1 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 1 \
  --run_name C44-E1-r21-source \
  --model_dir workdir \
  2>&1 | tee my_new/r2.1/C44_E1-source-checkpoint.log

SOURCE_CKPT=$(find workdir/C44-E1-r21-source -maxdepth 1 -type f -name '*.ckpt' \
  | sort | tail -n 1)
test -n "${SOURCE_CKPT}"

python train.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --atlas \
  --prepend_ipa \
  --num_frames 250 \
  --crop 256 \
  --batch_size 1 \
  --num_workers 0 \
  --train_seed 137 \
  --no-deterministic \
  --epochs 2 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --ckpt "${SOURCE_CKPT}" \
  --run_name C44-E1-r21-resume \
  --model_dir workdir \
  2>&1 | tee my_new/r2.1/C44_E1-one-step-resume.log
```

執行結果：完成；來源 checkpoint 於 epoch 0、`trainer_step=1` 產生。resume 時顯示 `Restored all states`，並完成 epoch 1，使 `trainer_step` 由 1 前進至 2。

## E2 — 量測輸出記錄執行設定

處置：在 runtime 與 L-scaling 的 JSON 輸出加入實際 matmul precision、deterministic algorithms 狀態，以及各 MHA 最後使用的 attention backend 與 SDPA fallback reason。此次只修改輸出欄位，不重跑 r2 的重量案例。

| file_path | block_name | 說明改動 |
|---|---|---|
| `train-runtime.py` | `sdpa-diagnostics`（r1 已存在；r2.1 修改並保留） | 新增共用的 MHA metadata 收集函式，逐一記錄 `last_attention_backend` 與 `last_sdpa_fallback_reason`。 |
| `train-runtime.py` | `peak_memory`（r1 已存在；r2.1 修改並保留） | peak-memory JSON 新增 `float32_matmul_precision`、`deterministic_algorithms_enabled` 與 `attention_modules`。 |
| `train-runtime.py` | `execution_time`（r1 已存在；r2.1 修改並保留） | execution-time JSON 新增相同的三項 runtime 設定。 |
| `benchmark_l_scaling.py` | `a4-l-scaling`（r1 已存在；r2.1 修改並保留） | L-scaling JSON 新增 matmul precision 與 deterministic algorithms 實際狀態；各 attention module 的欄位名稱明確改為 `last_attention_backend` 與 `last_sdpa_fallback_reason`。 |

### C45 — E2 runtime metadata smoke test

```bash
conda activate mdgen2026-r2-a6-cu126
mkdir -p my_new/r2.1

python benchmark_l_scaling.py \
  --sim_ckpt ckpt/atlas.ckpt \
  --lengths 16 \
  --num_frames 250 \
  --seed 137 \
  --output my_new/r2.1/C45_E2-runtime-metadata.json \
  2>&1 | tee my_new/r2.1/C45_E2-runtime-metadata.log

python - <<'PY'
import json

path = "my_new/r2.1/C45_E2-runtime-metadata.json"
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)

assert "float32_matmul_precision" in report
assert "deterministic_algorithms_enabled" in report
assert report["results"]
assert report["results"][0]["attention_modules"]
for metadata in report["results"][0]["attention_modules"].values():
    assert "last_attention_backend" in metadata
    assert "last_sdpa_fallback_reason" in metadata

print("float32_matmul_precision=", report["float32_matmul_precision"])
print(
    "deterministic_algorithms_enabled=",
    report["deterministic_algorithms_enabled"],
)
print("attention_modules=", len(report["results"][0]["attention_modules"]))
print("passed=true")
PY
```

執行結果：完成；L=16 forward 成功，JSON 記錄 `float32_matmul_precision=highest`、`deterministic_algorithms_enabled=false`，15 個 attention modules 的 `last_attention_backend` 均為 `sdpa`，且每個 module 均包含 `last_sdpa_fallback_reason` 欄位。

## r2 → r2.1 處置總表

| # | 處置 | commit | 說明 |
|---|---|---|---|
| R1 | 完成 | `deee9ff` | 已建立 `CHANGES_vs_base.md`，整合相對上游 `81482a4` 的 r1、r2、r2.1 改動。 |
| R2 | 完成 | `8cb3fa2` | C19 的 cu126／cu130 conda 建立 log 已納入 repository。 |
| R3 | 完成 | `0c87584` | 四類大型產物已加入 `.gitignore` 並由 Git index 移除；實體檔未刪除，保存位置已記錄於本報告。 |
| R4 | 完成 | `e0944f0`；驗證產物 `8cb3fa2` | 非 `None` 的 `position_ids` 統一在 MHA 入口拒絕；C41 通過。 |
| R5 | 完成 | `70e5884` | 底層 attention／layer 預設改為 SDPA；C42 通過。 |
| R6 | 完成 | `9a3e6e0`、`d554581` | PyEMMA 環境名稱與 r1 區隔，舊 YAML 已標記為歷史環境，cu126 PyEMMA YAML 為正式檔。 |
| E1 | 完成 | `d554581` | C43 validation checkpoint load 與 C44 one-step fit resume 均成功，三份 log 已保存。 |
| E2 | 完成 | `c5334f1` | 兩個量測入口的 JSON 已加入 runtime 設定與各 MHA backend；C45 通過。 |
