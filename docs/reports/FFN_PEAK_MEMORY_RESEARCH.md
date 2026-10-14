# FFN 展開中間張量的峰值記憶體：文獻整理

整理日期：2026-10-13  
範圍：Transformer／MLP 的 FFN 中間張量過大，導致單次 forward 或長序列生成的 GPU 峰值記憶體暴增。優先採用原始論文與官方文件；論文中的效能數字是作者在其模型、硬體和資料集上的結果，不代表 MDGen 一定能得到相同幅度。

## 結論先講

有，而且最直接、最容易套到既有 checkpoint 的方法已經很成熟：**沿著彼此獨立的 token／位置分批計算同一個 FFN**。Reformer 論文把這種方法明確用在 FFN，並指出它只改變計算實作，數學上的 layer 計算不變；分批後，最大的暫存由「所有位置 × FFN 寬度」縮成「單批位置 × FFN 寬度」。

較新的研究則分成幾條路：融合 kernel，讓中間結果留在 GPU 片上記憶體；改 FFN 架構，只保留部分活化通道；或在多 GPU 長序列訓練時做 sequence chunking、pipeline 和 activation checkpointing。後幾種通常牽涉自訂 kernel、分散式設定或重新訓練，不能直接視為替換目前 checkpoint 的小改動。

就目前的 MDGen 問題，優先順序是：**先做 inference-only FFN 分批，保留 FP32 和原權重；再看 profiler 決定是否值得做融合 kernel；最後才評估 BF16 或改 FFN 架構。**

## MDGen 的實際瓶頸

`mdgen/model/latent_model.py` 的 forward 會執行 `fc2(gelu(fc1(x)))`（可看 [latent_model.py](../../mdgen/model/latent_model.py)）。若輸入有 `N` 個位置、FFN 隱藏寬度是 `d_ff`，第一層 linear 和 GELU 會處理約 `N × d_ff` 個元素。FP32 單一同尺寸 tensor 的理想大小是 `4 × N × d_ff` bytes；GELU 實作還可能產生除以 √2、erf 和乘法等暫存，所以峰值可能同時包含多個中間結果。

專案的 [L-scaling 量測報告](LOCAL_SDPA_L_SCALING_REPORT.md) 記錄：完整生成、250 frames 時，L=5000 在 GELU 配置 `torch.erf(...)` 結果時因還需要約 7.15 GiB 而 OOM；L=7500 在 GELU 的 `x / sqrt(2)` 暫存配置時因還需要約 10.73 GiB 而 OOM。trace 指向 FFN GELU，沒有指向 SDPA attention。這個觀察與「減少 FFN 展開暫存」直接吻合。

## 相關論文與方法

| 工作 | 解決哪一種記憶體 | 方法與結果 | 對目前 MDGen 的適用性 |
|---|---|---|---|
| [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451), ICLR 2020 | FFN 的單次展開暫存；另處理訓練時跨層 activation | 把位置拆成 chunks，逐批執行 position-wise FFN。作者指出 FFN chunking 只改實作、計算數學上不變；可降低 FFN 內部的 peak activation。它也用了 reversible residual layers，減少訓練時跨層保存 activation。 | **最直接的參考。** 可保留目前 FFN 權重，只分小批輸入；適合先做 inference-only。可先不採用 Reformer 的 LSH attention 或 reversible block，因為那是另外兩種不同改動。 |
| [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370), 2023 preprint | 長序列 Transformer 的 attention 和 FFN 暫存 | 用 blockwise attention 與 FFN fusion 降低記憶體；作者報告可訓練比 vanilla Transformer 長 32 倍、比先前 memory-efficient 方法長 4 倍的序列。 | 支持「按區塊執行整個長序列」的方向，但主要是長序列訓練設計，改動和驗證範圍比只把 FFN 拆 chunk 大。 |
| [Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer (FPDT)](https://proceedings.mlsys.org/paper_files/paper/2025/file/d5a655b8b373737b4f2aea8f78e5e754-Paper-Conference.pdf), MLSys 2025 | 多 GPU 長序列訓練的 activation／中間 buffer | 對 attention 和 FFN 用不同 chunk 粒度。論文在 GPT/Llama 測試中發現，FFN chunk 數設為 attention 的兩倍足以讓 attention 成為主要記憶體限制；2.7B 案例中，兩個 sequence chunks 將 activation memory 從 27 GB 降為 18 GB。這是多 GPU pipeline/sequence-parallel 系統。 | 有助於理解 FFN 和 attention 的峰值要分開量，也證明 chunk 數需要依算子調整；不是單卡 inference 的直接程式碼配方。 |
| [Flash Multi-Head Feed-Forward Network (FlashMHF)](https://arxiv.org/abs/2512.06989), 2025 arXiv preprint | FFN 內部中間狀態和 HBM 流量 | 以不同的 multi-head FFN 架構搭配 fused、blockwise kernel，線上計算 SwiGLU，避免把完整中間 hidden state 寫到 HBM。作者報告在 128M–1.3B 語言模型上降低 3–5 倍 peak memory。 | 直接針對 FFN 中間狀態，但它**改了架構並需要專用 kernel／訓練驗證**；不是載入現有 MDGen checkpoint 後就能開啟的推論選項。 |
| [Mixture-of-Channels (MoC)](https://arxiv.org/abs/2511.09323), 2025 arXiv preprint | 訓練時保存的 FFN activation，也降低推論權重讀取 | 逐 token 選 Top-K FFN channels，訓練時只儲存被選中的部分 activation。作者以 LLaMA-style FFN 配置報告，activation 記憶體由 `11.67bsd` 降至 `3.67bsd`（保留約 20% channels），並報告推論效能改善。 | 很貼近「FFN 活化太大」的研究問題，但會改 FFN 的函數與稀疏模式，需重新訓練或做品質／相容性研究；不適合當作第一個 checkpoint-preserving 修正。 |
| [GACT: Activation Compressed Training for Generic Network Architectures](https://proceedings.mlr.press/v162/liu22v.html), ICML 2022 | 訓練時為 backward 保留的 activation | 將保存的 activation 壓縮；作者在 CNN、Transformer、Graph NN 任務上報告最高 8.1× activation memory reduction，並稱精度損失很小。 | 是訓練記憶體方案，不是降低 inference 中某個 GELU 暫存的直接方案；壓縮會讓梯度成為近似值，需另外驗證 MDGen 訓練穩定性。 |
| [ActNN](https://proceedings.mlr.press/v139/chen21z.html), ICML 2021 | 訓練時保存的 activation | 以隨機量化方式儲存 backward 所需 activation；作者在其視覺任務中報告平均 2-bit activation、12× activation memory reduction。 | 目標偏訓練，且作者主要評估視覺任務；不直接處理本次 inference 的 GELU 單次 allocation。 |
| [Out of the Memory Barrier (OOMB)](https://arxiv.org/abs/2602.02108), 2026 arXiv preprint | 百萬 token 長序列訓練的整體 activation/KV 記憶體 | chunk-recurrent training、on-the-fly recomputation、paged KV/gradient 管理和 offload；作者報告在單張 H200 訓練 Qwen2.5-7B、4M context。 | 代表 2026 年長上下文訓練系統的方向，但它不是單一 FFN GELU allocation 的局部修補；對當前單卡 MDGen inference 屬較遠期參考。 |

## 常見技術要分清楚

### 1. Token-wise FFN chunking：優先做

把 `x` 沿位置維切成 `x₁, x₂, …`，每次只算 `fc2(gelu(fc1(xᵢ)))`，最後將輸出接回完整位置順序。position-wise FFN 不會讓不同位置互相混合，所以在數學上可以逐塊算同一個函數。若 chunk 含 `C` 個位置，FFN 最大工作暫存大致由 `N × d_ff` 降成 `C × d_ff`；模型完整輸出仍需保留。

分塊前後使用同一個 checkpoint、同一 dtype 時，沒有刻意改變模型權重或運算公式。不過因 GEMM 的矩陣形狀變了，浮點加總順序可能不同，實測應比較誤差容限，而非要求 bitwise 一模一樣。若在 training 使用一般 autograd，所有 chunk 的反向所需 activation 仍可能被保留；要壓 training 的跨步保存量，需要再搭配 checkpoint/recompute。對 inference，`eval()` 下沒有 dropout 的 FFN chunking 最直接。

### 2. Fusion／`torch.compile`：可能減少暫存，需量測

PyTorch 官方文件說 TorchInductor 可以融合可融合的 pointwise／reduction 算子；`epilogue_fusion` 可在 `max-autotune` 下測試。這值得做成另一個對照實驗，但「能融合 GELU 等 pointwise op」不保證特定模型一定把 `fc1` 的整個輸出 tile 化，或保證峰值就降到某個數值；要看實際生成的 kernel、GPU backend、dtype、shape，並重新量 `max_memory_allocated`。`reduce-overhead` CUDA graph 也可能增加 workspace memory，文件明確提醒有些情況會多用記憶體。

### 3. BF16／FP16：能減少每個元素的 bytes，會改變精度

FP32 activation 每元素 4 bytes；BF16／FP16 通常是 2 bytes，所以同一個 activation tensor 的理論儲存量約減半。PyTorch AMP 會依算子選用精度；這不是「完全不影響結果」的精確性優化。對 MDGen 需把 FP32 當基準，檢查輸出、幾何量、trajectory 統計和長 rollout 是否出現累積漂移，再決定能否接受。

### 4. Activation checkpointing：主要救 training 保存量

PyTorch 定義 checkpointing 為用額外計算換記憶體：forward 不保留區段內所有中間 activation，backward 需要時再重算。它有助於多層 training 的 activation 保存量；如果單次 inference 的 FFN forward 本身就要配置一個放不下的 GELU 暫存，checkpointing 通常不會消除該算子執行當下的 allocation 峰值。

### 5. Attention 方法不是 FFN 修正

FlashAttention／SDPA 把 attention 內的中間矩陣和記憶體流量當作目標。若 traceback 是 `fc1 -> GELU/erf`，再調 SDPA 不會直接縮小這個 FFN 張量；本專案的 L-scaling 報告也觀察到 OOM 發生在 FFN GELU。

## 對 MDGen 的建議實驗次序

1. **FP32、固定 checkpoint 和固定輸入，做 full FFN vs inference chunked FFN。** 先在短序列比較輸出 `max_abs/max_rel`，再量 L=2500/5000/7500、250 frames 的單次 forward 和完整生成。記錄時間、peak allocated/reserved、OOM 的階段。chunk 大小逐步調整，找能容納且不過度拖慢的值。
2. **如果分塊正確但太慢，再單獨測 fusion／`torch.compile`。** 比較生成 kernel 和記憶體曲線；不要把 compile time 算進穩態 inference latency，也留意 shape guard/recompile 和 workspace。
3. **需要更多容量時才試 BF16/FP16。** FP32 對照不能拿掉；除了 tensor output，還要比較生成 trajectory 的幾何與物理合理性。
4. **若改做 training，再另外量 backward。** 分別評估 chunking、activation checkpointing；若追求更大 context，再看 FPDT/OOMB 這類分散式／recurrent 方法。
5. **把 MoC／FlashMHF 留作模型研究方向。** 只有在你願意重訓或做蒸餾、並能驗證 MDGen 結構品質時，才進入架構比較。

最重要的公平性原則：兩個 FFN 實作必須使用同一 checkpoint、輸入、dtype、gradient/inference mode 和 rollout 條件；記憶體與速度要分開報告。L40S 的結果也不能直接當成 4090 的可跑上限，因為可用顯存與 kernel 執行條件不同，最終要在目標卡重測。

## 主要來源

- Kitaev, Kaiser, Levskaya. *Reformer: The Efficient Transformer*. ICLR 2020 / [arXiv:2001.04451](https://arxiv.org/abs/2001.04451).
- Liu, Abbeel. *Blockwise Parallel Transformer for Large Context Models*. [arXiv:2305.19370](https://arxiv.org/abs/2305.19370).
- Yao et al. *Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer*. MLSys 2025, [paper PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/d5a655b8b373737b4f2aea8f78e5e754-Paper-Conference.pdf).
- Zhang et al. *Flash Multi-Head Feed-Forward Network*. [arXiv:2512.06989](https://arxiv.org/abs/2512.06989), preprint.
- Wu et al. *Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference*. [arXiv:2511.09323](https://arxiv.org/abs/2511.09323), preprint.
- Liu et al. *GACT: Activation Compressed Training for Generic Network Architectures*. ICML 2022, [PMLR](https://proceedings.mlr.press/v162/liu22v.html).
- Chen et al. *ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training*. ICML 2021, [PMLR](https://proceedings.mlr.press/v139/chen21z.html).
- Li et al. *Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts*. [arXiv:2602.02108](https://arxiv.org/abs/2602.02108), preprint.
- PyTorch, [Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint), [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide), [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile), and [Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/accelerator/amp.html).
