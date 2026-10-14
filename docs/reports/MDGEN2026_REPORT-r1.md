## 2026-09-25 mdgen2026
> 建立日期：2026-09-24 ｜ 最後更新：2026-09-24  
> 作者：王皇文  
> github帳號：`32176701Jeff`  mail:`r12521230@ntu.edu.tw`

## 主要改進
1. torch版本更新
1. SDPA,FP32
1. positionID傳遞到ROPE
1. seed/deterministic設定

## 交付內容
1. T1-電腦與環境
1. T2-原始code議題確認
1. T3-code_modification
1. T4-等價性測試
1. T5-Performance/Batchsize/residue_length/GPU_memory

## T1-電腦與環境

### T1.1-電腦規格
| 項目 | 規格 |
|---|---|
| 主機名稱 | m5-cluster2 |
| 作業系統 | Ubuntu 24.04.5 LTS |
| 系統架構 | x86_64 |
| Linux kernel | 7.0.0-31-generic |
| CPU | AMD Ryzen 9 9950X |
| CPU核心／執行緒 | 16 cores／32 threads |
| 系統記憶體 | 30 GiB |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU記憶體 | 24,564 MiB |
| NVIDIA driver | 595.71.05 |
| Driver最高支援CUDA版本 | 13.2 |

### T1.2-Conda與PyTorch環境
cu126與cu130環境使用相同的Python及MDGen相依套件，主要差異只有PyTorch使用的CUDA runtime版本。
共同套件版本：
| 項目 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch Lightning | 2.6.5 |
| NumPy | 1.26.4 |
| fair-esm | 2.0.0 |
| torchdiffeq | 0.2.5 |

環境差異：
| 項目 | cu126環境 | cu130環境 |
|---|---|---|
| 環境設定檔 | [environment-mdgen2026-cu126.yml](../../envs/environment-mdgen2026-cu126.yml) | [environment-mdgen2026-cu130.yml](../../envs/environment-mdgen2026-cu130.yml) |
| Conda環境名稱 | `mdgen2026-cu126` | `mdgen2026-cu130` |
| PyTorch | 2.12.1+cu126 | 2.12.1+cu130 |
| PyTorch CUDA runtime | 12.6 | 13.0 |
| Torchvision | 0.27.1+cu126 | 0.27.1+cu130 |

### T1.3-相依性檢查
| 檢查項目 | 結果 |
|---|---|
| PyTorch Lightning 是否阻擋 PyTorch 2.12.1 | 否；實際安裝 PyTorch Lightning 2.6.5 |
| torchaudio | 未安裝，MDGen 未使用 |
| pip openfold | 未安裝 |
| flash-attn | 非必要相依；本次使用 PyTorch 內建 SDPA |


## T2-原始code議題確認

T2檢查原始MDGen在導入SDPA前需要確認的實作細節；對應的MDGen-2026修改方式記錄於T3。

原始MDGen：[https://github.com/bjing2016/mdgen](https://github.com/bjing2016/mdgen)

### T2.1-SDPA前置確認

| 編號 | 確認項目 | 原始code確認結果 | 對應T3項目 |
|---|---|---|---|
| Q1 | Scaling | Q已在外部乘上`head_dim**-0.5` | Scaling |
| Q2 | Padding mask | Dataset、MHA與SDPA使用的mask極性不同 | Mask極性與fast path |
| Q3 | add_bias_kv | K/V會增加一個learned token，mask也會同步延長 | bias K/V |
| Q4 | Time-axis mask | Time-axis attention有傳入mask | Time-axis mask |
| Q5 | Dropout | Manual attention只在training啟用dropout | Dropout |
| Q6 | 輸出比較位置 | MHA在`out_proj`後的共同輸出為`attn` | Tensor layout與輸出 |

#### Q1-Scaling位置

確認結果：

SDPA內部會自動執行scaling。原始MDGen已在外部對Q乘上`head_dim**-0.5`，因此導入SDPA時需要避免重複scaling。

原始code：

```
# mdgen/model/mha.py, line 102
self.scaling = self.head_dim**-0.5

# mdgen/model/mha.py, line 263
q *= self.scaling

# mdgen/model/mha.py, line 359
attn_weights = torch.bmm(q, k.transpose(1, 2))
```

#### Q2-Padding

確認結果：

Dataset的mask中，1代表有效residue，0代表padding。原始MDGen使用`1 - mask`產生`key_padding_mask`，因此有效位置變成0，padding位置變成1；在manual attention中，`key_padding_mask=True`代表需要遮蔽。PyTorch SDPA的boolean mask語意剛好相反，True代表允許參與attention，False代表遮蔽，因此傳入SDPA前必須對`key_padding_mask`取反。
| 位置 | Dataset mask | key_padding_mask | SDPA boolean mask |
|---|---:|---:|---:|
| 有效residue | 1 | False／0 | True／1 |
| Padding | 0 | True／1 | False／0 |

Padding mask的作用是：所有位置先計算Q、K、V；完成QKᵀ後，padding key對應的attention score會在softmax前設為`-inf`，softmax後權重變成0，因此對應的V不會貢獻到最終output。

原始code：

[1] Dataset建立mask
```
# mdgen/dataset.py, line55-56
L = frames.shape[1]
mask = np.ones(L, dtype=np.float32)

# mdgen/dataset.py, line80–89, L<cropping window
mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])
```

[2] 轉換成key_padding_mask
```
# mdgen/model/latent_model.py，line 458–461
mask.reshape(B * T, L)

# mdgen/model/latent_model.py, line 325–327
x, _ = self.attn(
    query=x,
    key=x,
    value=x,
    key_padding_mask=1 - mask
)
```

[3] Manual attention套用mask
```
# mdgen/model/mha.py，line 251–263
q = self.q_proj(query)
k = self.k_proj(key)
v = self.v_proj(value)
q *= self.scaling

# mdgen/model/mha.py，line 359
attn_weights = torch.bmm(q, k.transpose(1, 2))

# mdgen/model/mha.py，line 370–375
if key_padding_mask is not None:
    attn_weights = attn_weights.view(
        bsz, self.num_heads, tgt_len, src_len
    )
    attn_weights = attn_weights.masked_fill(
        key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
        float("-inf")
    )

# mdgen/model/mha.py，line 381
attn_weights_float = utils_softmax(
    attn_weights, dim=-1, onnx_trace=self.onnx_trace
)

# mdgen/model/mha.py，line 383–389
attn_probs = F.dropout(
    attn_weights_float.type_as(attn_weights),
    p=self.dropout,
    training=self.training,
)
attn = torch.bmm(attn_probs, v)
```

#### Q3-add_bias_kv

確認結果：

`bias_k`和`bias_v`是兩個可以訓練的額外向量。假設原本序列有L個token，啟用`add_bias_kv`後，程式會在K和V的尾端各加上一個learned token。Q長度仍為L，K和V長度變成L+1，padding mask也必須同步延長為L+1。

Bias token不是padding，所以原始`key_padding_mask`在尾端補0，代表不遮蔽。SDPA本身沒有`add_bias_kv=True`參數，只接收已經準備好的Q、K、V。

原始code：

```
# mdgen/model/latent_model.py, line 343, IPA
self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

# mdgen/model/latent_model.py, line 401, LatentMDGenLayer
self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)
```

#### Q4-Time-axis attention有沒有使用mask

確認結果：

Time-axis attention有使用mask。

原始code：

```
# mdgen/dataset.py，line 80–89
mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

# mdgen/wrapper.py，line 353–364, 原始 batch['mask'] 的形狀是 B × L。這裡沿著時間軸展開成 B × T × L
'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1)

# mdgen/model/latent_model.py，line 472–475
x = self.mha_t(
    x.transpose(1, 2).reshape(B * L, T, C),
    mask=mask.transpose(1, 2).reshape(B * L, T)
)
```

#### Q5-Dropout

確認結果：

原始manual attention透過`training=self.training`控制dropout；SDPA需要由呼叫端明確設定dropout機率。

原始code：

```
# mdgen/model/mha.py，line 381–389
attn_weights_float = utils_softmax(
    attn_weights,
    dim=-1,
    onnx_trace=self.onnx_trace
)
attn_probs = F.dropout(
    attn_weights_float.type_as(attn_weights),
    p=self.dropout,
    training=self.training,
)
attn = torch.bmm(attn_probs, v)
```

#### Q6-輸出比較位置

確認結果：

Manual attention與SDPA皆以`out_proj`後的`attn`作為數值比較位置。比較時另外區分diffusion time、layer及residue／frame／IPA attention。

原始code：

```
mdgen/model/mha.py，line 397，variable_name = attn
```

### T2.2-Data representation與cropping順序

確認結果：

原始MDGen先計算frame與torsion，再執行crop。

原始code：

```
# mdgen/dataset.py, line79-84
torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
torsion_mask = torsion_mask[0]
if self.args.atlas:
    if L > self.args.crop:
```


# T3.0 — 修改方向

MDGen-2026的程式修改分成五個方向：

| 方向 | 目的 | 主要內容 |
|---|---|---|
| SDPA（T3.2） | 將原本的manual attention改為PyTorch SDPA，同時保留數學語意與manual比較路徑 | SDPA routing、scaling、mask、bias K/V、dropout、tensor layout及FP32設定 |
| Position ID（T3.3） | 先建立後續fork共用的RoPE位置介面 | `position_ids=None`介面，以及CSV、Dataset、model與inference之間的欄位傳遞 |
| Deterministic／seed（T3.4） | 讓manual與SDPA在相同條件下比較 | Training/inference seed、deterministic algorithms及cuDNN benchmark控制 |
| Other（T3.5） | 收納不屬於以上三類的相容性及輔助修改 | PyTorch/checkpoint相容性、資料前處理修正及model輸出目錄 |
| Runtime驗證與new files（T3.6） | 集中介紹新增檔案，以及產生T4/T5所需證據的工具 | 實際SDPA backend、peak memory、sec/step、模組等價性、L-scaling及輔助工具 |

老師的規格定義功能及驗收條件，沒有指定`block_name`名稱。本報告以`block_name`標記`origin/master`之後的程式修改：(a) 完整新增function/class時，在定義前一行標記`# <block_name>`，不再以`:start`／`:end`包住整個function/class；(b) 既有function/class內只修改一行時，直接在該行行尾標記`# <block_name>`，argument、import或常數的單行修改亦同；(c) 既有function/class內連續修改多行時，只以`# <block_name>:start`／`# <block_name>:end`緊貼包住實際修改的連續範圍；(d) 若兩段修改中間包含未修改的舊程式碼，應拆成兩組`:start`／`:end`，兩組可以沿用相同`block_name`，不得為了合併成一大段而把舊程式碼一起包入；(e) `:start`／`:end`不得跨越function或class；(f) 刪除程式碼時直接刪除原程式碼及其標記，不保留`delete_`、`removed`或空白區塊，刪除項目由報告表格及版本diff記錄。

## T3.1 — 新增項目

### T3.1.1 — New files

此處只介紹新增檔案及新增原因；各檔案自己的CLI參數留在該檔案說明或`--help`中，不列入T3.1.2。

| 新增檔案 | 方向（T3.x） | 新增原因 | 類別 |
|---|---|---|---|
| `scripts/prep-protein-csv.py` | Position ID（T3.3） | 原始MDGen沒有從PDB建立`seqres`與`position_ids` CSV的工具 | Position ID輔助 |
| `train-runtime.py` | Runtime驗證與new files（T3.6） | 將本次training backend、peak memory及sec/step量測與下一階段正式`train.py`隔離 | Runtime驗證 |
| `test_sdpa.py` | Runtime驗證與new files（T3.6） | 原始repository沒有manual與SDPA的MHA forward/backward等價性測試 | Runtime驗證 |
| `benchmark_l_scaling.py` | Runtime驗證與new files（T3.6） | 原始repository沒有逐L量測單次model forward峰值、OOM及實際SDPA backend的工具 | Runtime驗證 |
| `scripts/analyze_ensemble.py` | Runtime驗證與new files（T3.6） | 從AlphaFlow取得的ensemble分析工具 | 非本次驗收主線 |
| `scripts/print.py` | Runtime驗證與new files（T3.6） | 彙整ensemble分析pickle結果，AlphaFlow下載 | 非本次驗收主線 |
| `.gitignore` | Runtime驗證與new files（T3.6） | 排除checkpoint、workdir、wandb及暫存檔 | Repository設定 |

### T3.1.2 — New args in existing files

本節只列出原有檔案中新增的args，不列新增檔案自己定義的args。Training runtime args仍由原有的`mdgen/parsing.py`定義，因此列於此表；正式`train.py`不使用這三個量測args，只有`train-runtime.py`會讀取並執行對應功能。

| 新增args | 原有檔案 | 預設行為 | 方向（T3.x） | 新增原因／block_name |
|---|---|---|---|---|
| `--use_sdpa` | `mdgen/parsing.py` | 未提供時使用manual attention | SDPA（T3.2） | 保留同環境A/B比較路徑；`sdpa-route` |
| `--print_sdpa_backend <path>` | `mdgen/parsing.py` | 預設不啟用profiler | Runtime驗證（T3.6） | 由`train-runtime.py`輸出實際SDPA backend；`sdpa-diagnostics` |
| `--peak_memory <file_path>` | `mdgen/parsing.py` | 預設不輸出JSON | Runtime驗證（T3.6） | 由`train-runtime.py`紀錄training CUDA peak memory與OOM；`peak_memory` |
| `--execution_time <file_path>` | `mdgen/parsing.py` | 預設不輸出JSON | Runtime驗證（T3.6） | 由`train-runtime.py`紀錄排除warm-up後的sec/step；`execution_time` |
| `--train_seed <int>` | `mdgen/parsing.py` | 137 | Deterministic／seed（T3.4） | 控制training seed；`seed-deterministic-args` |
| `--deterministic`／`--no-deterministic` | `mdgen/parsing.py` | deterministic開啟 | Deterministic／seed（T3.4） | 控制deterministic algorithms；`seed-deterministic-args` |
| `--benchmark`／`--no-benchmark` | `mdgen/parsing.py` | cuDNN benchmark關閉 | Deterministic／seed（T3.4） | 控制cuDNN autotuner；`seed-deterministic-args` |
| `--model_dir <path>` | `mdgen/parsing.py` | 使用`workdir/<run_name>` | Other（T3.5） | 可選training輸出根目錄；`model-output-dir` |
| `--use_sdpa` | `sim_inference.py` | 未提供時使用manual attention | SDPA（T3.2） | inference SDPA路徑；`sdpa-route` |
| `--inference_seed <int>` | `sim_inference.py` | 137 | Deterministic／seed（T3.4） | 控制inference seed；`seed-deterministic-args` |
| `--deterministic`／`--no-deterministic` | `sim_inference.py` | deterministic開啟 | Deterministic／seed（T3.4） | 控制deterministic algorithms；`seed-deterministic-args` |
| `--benchmark`／`--no-benchmark` | `sim_inference.py` | cuDNN benchmark關閉 | Deterministic／seed（T3.4） | 控制cuDNN autotuner；`seed-deterministic-args` |
| `--atlas_dir` | `scripts/prep_sims.py` | 舊`--sim_dir`仍可作為alias | Other（T3.5） | 統一程式實際使用的欄位名稱；`data-preprocess-fixes` |

`--precision`原本已存在，因此不是new arg；MDGen-2026只把允許值限制為`32-true`，詳細修改列於T3.2.2。

## T3.2 — SDPA

### T3.2.1 — Attention實作

MDGen-2026將原本的`bmm → mask → softmax → dropout → bmm`改為`F.scaled_dot_product_attention`，Q/K/V projection、residue/time兩軸attention、RoPE、bias K/V、dropout及output projection語意維持不變。

| 項目 | 處理方式 | 修改檔案 | block_name |
|---|---|---|---|
| SDPA route | 由wrapper將`use_sdpa`傳入各attention layer；未開啟時維持manual路徑，SDPA不適用時記錄fallback原因 | `mdgen/wrapper.py`、`mdgen/model/latent_model.py`、`mdgen/model/mha.py` | `sdpa-route` |
| Scaling | 保留原本`q *= self.scaling`，SDPA指定`scale=1.0`，避免double scaling | `mdgen/model/mha.py` | `sdpa-scaling` |
| Mask | 將fairseq的padding mask反轉成SDPA保留mask；mask為no-op時傳入`None` | `mdgen/model/mha.py` | `sdpa-mask` |
| bias K/V | MDGen確實使用`add_bias_kv=True`，因此保留K/V append及mask延長流程 | `mdgen/model/mha.py` | `sdpa-bias-kv` |
| Dropout | Training使用原dropout，eval傳入`0.0` | `mdgen/model/mha.py` | `sdpa-dropout` |
| Layout/output | SDPA前轉成`B,H,L,D`，完成後還原並通過原本`out_proj` | `mdgen/model/mha.py` | `sdpa-layout-output` |
| Time-axis mask | Time-axis attention沿用相同的mask及SDPA轉換流程 | `mdgen/model/latent_model.py` | `sdpa-time-axis-mask` |

### T3.2.2 — FP32執行條件

本次規格全程使用FP32，不啟用bf16、AMP或TF32。這些是本次SDPA等價性與benchmark的固定條件，不另外提供切換到其他dtype的args。

| 項目 | 處理方式 | 修改檔案 | block_name |
|---|---|---|---|
| Training precision | 將既有`--precision`限制為`32-true` | `mdgen/parsing.py` | `fp32-training-precision` |
| Matmul precision | Training與inference均設定為`highest` | `train.py`、`train-runtime.py`、`sim_inference.py` | `fp32-matmul-precision` |
| TF32 | 明確關閉CUDA matmul及cuDNN TF32 | `train.py`、`train-runtime.py`、`sim_inference.py` | `fp32-disable-tf32` |
| Inference model | checkpoint載入後使用`model.eval().float()` | `sim_inference.py` | `fp32-inference-model` |
| Data pipeline | NPY座標及inference輸入使用float32，避免先經float16量化 | `scripts/prep_sims.py`、`sim_inference.py` | `fp32-data-pipeline` |

## T3.3 — Position ID

老師要求的核心範圍是RoPE加入`position_ids=None`介面，且None路徑保持數值等價。目前實作另外把欄位由CSV傳到residue-axis MHA，作為後續fork的介面準備；RoPE仍沿用原本的連續位置計算，因此本次不宣稱true-gap或multi-chain語意已完成。

| 項目 | 處理方式 | 修改檔案 | block_name |
|---|---|---|---|
| RoPE介面 | MHA及RoPE接受`position_ids=None`；目前保持原始連續位置計算 | `mdgen/model/mha.py` | `position-id-rope-interface` |
| Dataset讀取 | CSV有欄位時讀取並檢查長度，沒有時使用`arange(L)` | `mdgen/dataset.py` | `position-id-dataset` |
| Crop/padding | 與sequence同步crop；padding時同步補齊 | `mdgen/dataset.py` | `position-id-crop-padding` |
| Wrapper routing | Training、validation及inference加入model kwargs | `mdgen/wrapper.py` | `position-id-wrapper` |
| Model routing | 經latent model傳到residue-axis attention | `mdgen/model/latent_model.py` | `position-id-model-routing` |
| Residue layout | 將`B,L`展開成residue attention使用的`B×T,L`；time-axis不使用 | `mdgen/model/latent_model.py` | `position-id-residue-layout` |
| Inference input | 從CSV讀取；沒有欄位時使用連續位置 | `sim_inference.py` | `position-id-inference-input` |
| Inference routing | 檢查CSV、NPY residue數後經rollout傳入model | `sim_inference.py` | `position-id-inference-routing` |

## T3.4 — Deterministic與seed

MDGen-2026預設seed為137、開啟deterministic algorithms並關閉cuDNN benchmark；使用者仍可透過T3.1.2列出的args調整。

| 項目 | 處理方式 | 修改檔案 | block_name |
|---|---|---|---|
| Args | 新增training/inference seed、deterministic及cuDNN benchmark控制 | `mdgen/parsing.py`、`sim_inference.py` | `seed-deterministic-args` |
| Seed初始化 | 使用`seed_everything(..., workers=True)`固定Python、NumPy、PyTorch及DataLoader worker | `train.py`、`train-runtime.py`、`sim_inference.py` | `seed-initialization` |
| Deterministic執行 | Training交由Lightning Trainer；inference呼叫`torch.use_deterministic_algorithms` | `train.py`、`train-runtime.py`、`sim_inference.py` | `deterministic-execution` |
| cuDNN benchmark guard | 禁止deterministic與cuDNN benchmark同時開啟 | `mdgen/parsing.py`、`train.py`、`train-runtime.py`、`sim_inference.py` | `deterministic-benchmark-guard` |

## T3.5 — Other block names

| 修改檔案 | block_name | 修改內容 | 定位 |
|---|---|---|---|
| `mdgen/model/primitives.py`、`mdgen/tensor_utils.py`、`sim_inference.py` | `runtime-compatibility` | 對legacy FlashAttention加入optional import guard、將新版PyTorch不接受的list indexing改為tuple，並以`weights_only=False`載入舊Lightning checkpoint | PyTorch 2.x與checkpoint相容性所需 |
| `mdgen/wrapper.py` | `checkpoint-runtime-args` | `save_hyperparameters`只保存原有`args`，避免`use_sdpa`等runtime選項成為checkpoint建構需求 | 新增runtime args後的checkpoint相容處理 |
| `mdgen/parsing.py`、`train.py`、`train-runtime.py` | `runtime-args-separation` | 正式`train.py`只註冊一般訓練參數與`use_sdpa`；backend profiler、peak memory及execution time三個量測參數只由`train-runtime.py`註冊與使用 | 將正式訓練入口與benchmark instrumentation分離 |
| `mdgen/dataset.py`、`scripts/prep_sims.py` | `data-preprocess-fixes` | 修正frame不足錯誤及最後合法window未被抽到；統一`atlas_dir`欄位並保留舊`--sim_dir` alias | 額外bug fix，不是SDPA主線要求 |
| `mdgen/parsing.py` | `model-output-dir` | 支援`--model_dir`指定training輸出根目錄 | 額外便利功能；未提供時維持原路徑 |
| `.gitignore` | `repository-artifacts` | 排除大型輸出及暫存檔 | 不影響model runtime |

## T3.6 — Runtime驗證與new files

### T3.6.1 — `train-runtime.py`與runtime instrumentation

`train-runtime.py`由完成本次量測功能的training入口保留下來，專門執行A2、A3及T5 training benchmark。正式`train.py`已移除profiler、peak-memory與execution-time實作，只保留一般training流程。兩者目前共用`mdgen/parsing.py`，但只有`train-runtime.py`讀取並使用三個量測args。

| 功能 | 執行方式與輸出 | 修改檔案 | block_name |
|---|---|---|---|
| SDPA backend | `--print_sdpa_backend`啟用profiler，輸出實際operator/backend及各attention module的manual/SDPA path | `train-runtime.py`、`benchmark_l_scaling.py`、`mdgen/model/mha.py` | `sdpa-diagnostics` |
| Peak memory | `--peak_memory`在training前reset CUDA peak，成功或OOM時輸出allocated/reserved、完成step及執行設定 | `mdgen/parsing.py`、`train-runtime.py` | `peak_memory` |
| Execution time | `--execution_time`使用CUDA Event量測完整training step，排除前5個warm-up step後輸出mean/median/min/max | `mdgen/parsing.py`、`train-runtime.py` | `execution_time` |

Training backend由`train-runtime.py`確認；inference條件下的backend則由`benchmark_l_scaling.py`以A4單次model forward確認。Profiler會影響效能及memory，因此backend確認與正式memory/time量測應分開執行。

### T3.6.2 — `test_sdpa.py`

`test_sdpa.py`是獨立的MHA模組級等價性測試，以固定seed建立權重相同的manual與SDPA MHA，使用相同FP32 input、mask及上游gradient，在`eval()`模式下比較forward output、input gradient及parameter gradients。測試涵蓋residue/time axis各自的padding及no-padding case，並要求SDPA組不能fallback到manual。

輸出包含各case的output NPY、input-gradient NPY、parameter-gradient NPZ、`result.json`及總結`summary.json`。對應block name為`sdpa-equivalence-test`，T4/A1使用此檔案。

### T3.6.3 — `benchmark_l_scaling.py`

`benchmark_l_scaling.py`載入既有checkpoint，以FP32、B=1、無gradient checkpointing及`torch.inference_mode()`執行完整`LatentMDGenModel`單次forward。預設量測L=256/1000/2500/5000/7500，逐case輸出CUDA peak allocated/reserved、attention path及OOM資訊；首次OOM後預設停止。

此檔案另提供`--print_sdpa_backend <path>`：以第一個指定L額外執行一次獨立profiling forward，輸出實際PyTorch SDPA operator/backend與各attention module路徑。這代表與A4相同B/T/L、FP32、mask及inference mode條件下的單次model forward，不宣稱量測完整ODE rollout；正式peak-memory數據應另一次不開profiler執行。對應block name為`a4-l-scaling`及`sdpa-diagnostics`，T5/A4使用此檔案。

### T3.6.4 — `scripts/prep-protein-csv.py`

此工具從PDB建立`name,seqres,position_ids` CSV。每條chain分別將第一個PDB residue number正規化為0並保留chain內gap，例如10、11、14輸出為`[0,1,4]`。它只負責測試資料準備，不是SDPA實作；對應block name為`position-id-csv`。

### T3.6.5 — Ensemble工具

`scripts/analyze_ensemble.py`及`scripts/print.py`是從AlphaFlow取得的下游ensemble分析與結果彙整工具，對應`ensemble-analysis-tool`及`ensemble-report-tool`。兩者不參與老師指定的T4模組等價性或T5硬體驗收，只有被獨立執行時才生效。

# T4-等價性測試
## code執行
cu126
```
conda activate mdgen2026
python test_sdpa.py \
  --device cuda \
  --seed 137 \
  --output_dir my_new/t4-cu126
```
cu130
```
conda activate mdgen2026-cu130
python test_sdpa.py \
  --device cuda \
  --seed 137 \
  --output_dir my_new/t4-cu130
```
## result
| CUDA環境 | Case | Output relative error | Input grad relative error | Parameter grad relative error | 結果 |
|---|---|---:|---:|---:|---|
| cu126 | Residue / no padding | 5.595 × 10⁻⁷ | 4.854 × 10⁻⁷ | 8.642 × 10⁻⁷ | Pass |
| cu126 | Residue / padding | 5.120 × 10⁻⁷ | 5.119 × 10⁻⁷ | 8.243 × 10⁻⁷ | Pass |
| cu126 | Time / no padding | 3.754 × 10⁻⁷ | 3.994 × 10⁻⁷ | 5.065 × 10⁻⁷ | Pass |
| cu126 | Time / padding | 3.164 × 10⁻⁷ | 4.229 × 10⁻⁷ | 5.416 × 10⁻⁷ | Pass |
| cu130 | Residue / no padding | 5.595 × 10⁻⁷ | 4.854 × 10⁻⁷ | 8.642 × 10⁻⁷ | Pass |
| cu130 | Residue / padding | 5.120 × 10⁻⁷ | 5.119 × 10⁻⁷ | 8.243 × 10⁻⁷ | Pass |
| cu130 | Time / no padding | 3.754 × 10⁻⁷ | 3.994 × 10⁻⁷ | 5.065 × 10⁻⁷ | Pass |
| cu130 | Time / padding | 3.164 × 10⁻⁷ | 4.229 × 10⁻⁷ | 5.416 × 10⁻⁷ | Pass |


# T5-Benchmark 與硬體驗收

## T5.0-Benchmark protocol 與測試條件
```
GPU：RTX 4090 24 GB
PyTorch：2.12.1
CUDA wheel：cu126 / cu130
dtype：FP32
TF32：off
T：250
seed：137
data：同一份 CSV 與 NPY
validation：off
profiler 與 memory/time 測量分開
```
測試變因
```
CUDA：cu126 / cu130
Attention：manual / SDPA
Gradient checkpointing：off / on
Batch size：由 1 往上
```
## T5.1-Datapreprocess
本次測試使用`1a62_A`與`1bkp_A`兩個蛋白質，並將training的`crop`設定為256。兩個蛋白質分別短於及長於cropping window，可同時測試padding與cropping兩種資料處理路徑。
| 蛋白質 | Residue數量 | 與crop=256的關係 | Dataset處理方式 | 測試目的 |
|---|---:|---|---|---|
| `1a62_A` | 130 | 小於256 | 保留完整序列並padding至256 | 驗證padding mask及SDPA mask轉換 |
| `1bkp_A` | 278 | 大於256 | 從原始序列crop出256個residues | 驗證cropping |

下載方式，下載後的檔案會存在./data/pdbxtc中
```
bash ./my_new/download_atlas.sh
```
### making csv
```
python scripts/prep-protein-csv.py \
  --input_dir data/pdbxtc \
  --output_csv data/proteins-mdgen2026.csv
```
### making npy
```
python scripts/prep_sims.py \
  --split data/proteins-mdgen2026.csv \
  --atlas_dir data/pdbxtc \
  --outdir data/npy \
  --num_workers 4 \
  --stride 40 \
  --atlas
```

## T5.2-實際 SDPA backend 確認

### code執行
cu126
```
conda activate mdgen2026
python train-runtime.py \
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
  --epochs 1 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name t5-backend-cu126-sdpa \
  --model_dir workdir \
  --use_sdpa \
  --print_sdpa_backend my_new/t5/backend/cu126-sdpa.txt
```
cu130
```
conda activate mdgen2026-cu130

python train-runtime.py \
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
  --epochs 1 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name t5-backend-cu130-sdpa \
  --model_dir workdir \
  --use_sdpa \
  --print_sdpa_backend my_new/t5/backend/cu130-sdpa.txt
```
### result
| CUDA環境 | PyTorch | Requested path | Selected backend | 結果 |
|---|---|---|---|---|
| cu126 | 2.12.1+cu126 | SDPA | EFFICIENT_ATTENTION | Pass |
| cu130 | 2.12.1+cu130 | SDPA | EFFICIENT_ATTENTION | Pass |

## T5.3-A2/A3
### 定義要跑項目
| 編號 | CUDA環境 | Attention | Gradient checkpointing | Batch size | 目的 |
|---|---|---|---|---:|---|
| C1 | cu126 | Manual | Off | 1 | cu126原始attention基準 |
| C2 | cu126 | SDPA | Off | 1 | cu126正式SDPA主結果 |
| C3 | cu130 | Manual | Off | 1 | cu130原始attention基準 |
| C4 | cu130 | SDPA | Off | 1 | cu130正式SDPA主結果 |
| C5 | cu126 | Manual | On | 1 | checkpointing對manual的影響 |
| C6 | cu126 | SDPA | On | 1 | checkpointing對cu126 SDPA的影響 |
| C7 | cu130 | Manual | On | 1 | checkpointing對cu130 manual的影響 |
| C8 | cu130 | SDPA | On | 1 | checkpointing對cu130 SDPA的影響 |

### code執行
C1
```
conda activate mdgen2026
python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C1-cu126-manual-gc-off \
  --model_dir workdir \
  --peak_memory my_new/t5/A2A3/C1-cu126-manual-gc-off-memory.json \
  --execution_time my_new/t5/A2A3/C1-cu126-manual-gc-off-time.json
```
C2
```
conda activate mdgen2026

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C2-cu126-sdpa-gc-off \
  --model_dir workdir \
  --use_sdpa \
  --peak_memory my_new/t5/C2-cu126-sdpa-gc-off-memory.json \
  --execution_time my_new/t5/A2A3/C2-cu126-sdpa-gc-off-time.json
```
C3
```
conda activate mdgen2026-cu130

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C3-cu130-manual-gc-off \
  --model_dir workdir \
  --peak_memory my_new/t5/A2A3/C3-cu130-manual-gc-off-memory.json \
  --execution_time my_new/t5/A2A3/C3-cu130-manual-gc-off-time.json
```
C4
```
conda activate mdgen2026-cu130

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C4-cu130-sdpa-gc-off \
  --model_dir workdir \
  --use_sdpa \
  --peak_memory my_new/t5/A2A3/C4-cu130-sdpa-gc-off-memory.json \
  --execution_time my_new/t5/A2A3/C4-cu130-sdpa-gc-off-time.json
```
C5
```
conda activate mdgen2026

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C5-cu126-manual-gc-on \
  --model_dir workdir \
  --grad_checkpointing \
  --peak_memory my_new/t5/A2A3/C5-cu126-manual-gc-on-memory.json \
  --execution_time my_new/t5/A2A3/C5-cu126-manual-gc-on-time.json
```
C6
```
conda activate mdgen2026

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C6-cu126-sdpa-gc-on \
  --model_dir workdir \
  --use_sdpa \
  --grad_checkpointing \
  --peak_memory my_new/t5/A2A3/C6-cu126-sdpa-gc-on-memory.json \
  --execution_time my_new/t5/A2A3/C6-cu126-sdpa-gc-on-time.json
```
C7
```
conda activate mdgen2026-cu130

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C7-cu130-manual-gc-on \
  --model_dir workdir \
  --grad_checkpointing \
  --peak_memory my_new/t5/A2A3/C7-cu130-manual-gc-on-memory.json \
  --execution_time my_new/t5/A2A3/C7-cu130-manual-gc-on-time.json
```
C8
```
conda activate mdgen2026-cu130

python train-runtime.py \
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
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C8-cu130-sdpa-gc-on \
  --model_dir workdir \
  --use_sdpa \
  --grad_checkpointing \
  --peak_memory my_new/t5/A2A3/C8-cu130-sdpa-gc-on-memory.json \
  --execution_time my_new/t5/A2A3/C8-cu130-sdpa-gc-on-time.json
```
### result
| Case | CUDA | Attention | Grad ckpt | Steps | Peak allocated | Peak reserved | sec/step | 結果 |
|---|---|---|---|---:|---:|---:|---:|---|
| C1 | cu126 | Manual | Off | 0 | ≥21.670 GiB | 22.715 GiB | — | OOM |
| C2 | cu126 | SDPA | Off | 105 | 20.541 GiB | 21.295 GiB | 0.5050 | Pass |
| C3 | cu130 | Manual | Off | 0 | ≥22.034 GiB | 22.848 GiB | — | OOM |
| C4 | cu130 | SDPA | Off | 1 | ≥20.282 GiB | 21.041 GiB | — | OOM |
| C5 | cu126 | Manual | On | 105 | 8.586 GiB | 16.311 GiB | 0.8803 | Pass |
| C6 | cu126 | SDPA | On | 105 | 8.586 GiB | 16.311 GiB | 0.6668 | Pass |
| C7 | cu130 | Manual | On | 1 | ≥16.520 GiB | 16.713 GiB | — | OOM |
| C8 | cu130 | SDPA | On | 1 | ≥16.520 GiB | 16.713 GiB | — | OOM |

## T5.4-A3-Batchsize
### 定義
| 編號 | CUDA | Attention | Grad ckpt | Batch size | 執行條件 |
|---|---|---|---|---:|---|
| C9 | cu126 | SDPA | Off | 2 | B=1已成功，測B=2 |
| C10 | cu126 | Manual | On | 2 | B=1已成功，測B=2 |
| C11 | cu126 | SDPA | On | 2 | B=1已成功，測B=2 |

### 執行code
C9
```
conda activate mdgen2026

python train-runtime.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --atlas \
  --prepend_ipa \
  --num_frames 250 \
  --crop 256 \
  --batch_size 2 \
  --num_workers 0 \
  --train_seed 137 \
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C9-cu126-sdpa-gc-off-b2 \
  --model_dir workdir \
  --use_sdpa \
  --peak_memory my_new/t5/A2A3/C9-cu126-sdpa-gc-off-b2-memory.json \
  --execution_time my_new/t5/A2A3/C9-cu126-sdpa-gc-off-b2-time.json
```
C10
```
conda activate mdgen2026

python train-runtime.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --atlas \
  --prepend_ipa \
  --num_frames 250 \
  --crop 256 \
  --batch_size 2 \
  --num_workers 0 \
  --train_seed 137 \
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C10-cu126-manual-gc-on-b2 \
  --model_dir workdir \
  --grad_checkpointing \
  --peak_memory my_new/t5/A2A3/C10-cu126-manual-gc-on-b2-memory.json \
  --execution_time my_new/t5/A2A3/C10-cu126-manual-gc-on-b2-time.json
```
C11
```
conda activate mdgen2026

python train-runtime.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --atlas \
  --prepend_ipa \
  --num_frames 250 \
  --crop 256 \
  --batch_size 2 \
  --num_workers 0 \
  --train_seed 137 \
  --epochs 105 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name C11-cu126-sdpa-gc-on-b2 \
  --model_dir workdir \
  --use_sdpa \
  --grad_checkpointing \
  --peak_memory my_new/t5/A2A3/C11-cu126-sdpa-gc-on-b2-memory.json \
  --execution_time my_new/t5/A2A3/C11-cu126-sdpa-gc-on-b2-time.json
```
### result
| Case | 結果 | 錯誤類型 | 發生位置 |
|---|---|---|---|
| C9 | 未產生JSON | `torch._C._LinAlgError` | `prep_batch → get_offsets → rot_to_quat → torch.linalg.eigh` |
| C10 | 未產生JSON | `torch._C._LinAlgError` | 同上 |
| C11 | 未產生JSON | `torch._C._LinAlgError` | 同上 |
CUSOLVER_STATUS_INVALID_VALUE
B=1：1 × 250 × 256 = 64,000
B=2：2 × 250 × 256 = 128,000

## T5.5-L-scaling單次forward
### 定義
| 編號 | CUDA | Attention | Grad ckpt | Residue lengths |
|---|---|---|---|---|
| C12 | cu126 | Manual | Off | 256、1000、2500、5000、7500 |
| C13 | cu126 | SDPA | Off | 256、1000、2500、5000、7500 |
固定條件
```
B=1
T=250
FP32
torch.inference_mode()
gradient checkpointing=False
synthetic inputs
```
### 執行code
C12
```
conda activate mdgen2026

python benchmark_l_scaling.py \
  --sim_ckpt ckpt/atlas.ckpt \
  --lengths 256 1000 2500 5000 7500 \
  --num_frames 250 \
  --seed 137 \
  --output my_new/t5/A4/C12-cu126-manual-l-scaling.json
```
C13
```
conda activate mdgen2026

python benchmark_l_scaling.py \
  --sim_ckpt ckpt/atlas.ckpt \
  --lengths 256 1000 2500 5000 7500 \
  --num_frames 250 \
  --seed 137 \
  --use_sdpa \
  --output my_new/t5/A4/C13-cu126-sdpa-l-scaling.json
```
### results
| L | Manual peak allocated | Manual結果 | SDPA peak allocated | SDPA結果 |
|---:|---:|---|---:|---|
| 256 | 2.616 GiB | Pass | 1.894 GiB | Pass |
| 1000 | ≥16.889 GiB | OOM | 6.978 GiB | Pass |
| 2500 | 未執行 | — | 17.236 GiB | Pass |
| 5000 | 未執行 | — | ≥20.428 GiB | OOM |
| 7500 | 未執行 | — | 未執行 | — |

# T5.6-A6 環境可複製驗收
## 定義
| 編號 | 驗證內容 | 是否必做 | 完成判準 |
|---|---|---|---|
| C14 | 從環境鎖定檔建立全新環境 | 必做 | Conda建立成功 |
| C15 | 版本與import檢查 | 必做 | `import mdgen`成功，版本正確，GPU可見 |
| C16 | ATLAS training smoke test | 必做 | 目標SDPA設定完成10 steps |
| C17 | Manual inference一次 | 必做 | 產生輸出PDB |
| C18 | SDPA inference一次 | 建議 | 產生輸出PDB |

## 執行code
C14
```
cd /mnt/hdd/jeff/mdgen-piezo/model/mdgen2026
conda env create \
  --name mdgen2026-a6-cu126 \
  --file my_new/environment-mdgen2026-cu126.yml
```
C15
```
conda activate mdgen2026-a6-cu126

python -c "import torch; import pytorch_lightning as pl; import mdgen; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('Lightning:', pl.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('import mdgen: PASS')"
```
C16
```
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
  --epochs 10 \
  --train_batches 1 \
  --no_validate \
  --ckpt_freq 999 \
  --run_name A6-cu126-sdpa-train-smoke \
  --model_dir workdir \
  --use_sdpa
```
C17
```
python sim_inference.py \
  --sim_ckpt ckpt/atlas.ckpt \
  --data_dir data/npy \
  --split data/proteins-mdgen2026.csv \
  --pdb_id 1a62_A \
  --suffix _R1 \
  --num_frames 250 \
  --num_rollouts 1 \
  --inference_seed 137 \
  --out_dir my_new/t5/A6/inference-manual
```
C18
```
python sim_inference.py \
  --sim_ckpt ckpt/atlas.ckpt \
  --data_dir data/npy \
  --split data/proteins-mdgen2026.csv \
  --pdb_id 1a62_A \
  --suffix _R1 \
  --num_frames 250 \
  --num_rollouts 1 \
  --inference_seed 137 \
  --out_dir my_new/t5/A6/inference-sdpa \
  --use_sdpa
```
