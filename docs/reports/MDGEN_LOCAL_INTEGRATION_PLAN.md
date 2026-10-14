# 從 mdgen 移入 local 功能至 mdgen2026：整合盤點

2026-10-14；以兩個目錄的目前工作檔案比較，包含未提交修改。下表記錄整合前差異；實作狀態請見 docs/NEIGHBOR_INTEGRATION.md。

目前不是「mdgen 完整繼承 mdgen2026，再加入 local」。兩邊各自發展，不能直接以資料夾名稱判斷版本先後或品質。

使用者已修正方向：mdgen2026 是實作目的地與共同基底，sibling mdgen 是 local 功能來源及凍結比較基準。此前「整合到 mdgen」的方向已撤銷；本輪已依確認的 spec 與四個 tickets 實作。

| 範圍 | mdgen2026 | sibling mdgen | 整合處置 |
|---|---|---|---|
| Attention | MHA 原生 SDPA 路由、manual fallback、backend 診斷 | 舊 MHA 加 pretrained local／full-sequence SDPA adapter；global 空間路徑未完整接入 2026 SDPA | 以 2026 全域路徑作基準，接入 local adapter；避免重複 scaling 或重複 RoPE |
| Local | 無 neighbor-cell adapter | partition、core-only Query、逐批 gather、local IPA、summary | 保留 local 擴充；summary 預設關閉供基準比較 |
| FFN | 主層推論分批 | 已移植主層分批 | 統一條件、chunk size 與 fallback；不把 conditioning IPA layer 說成已全部分批 |
| IPA | 全域推論 Query 分批 | local IPA 與 attn_mask 擴充，沒有 2026 的全域 Query 分批路徑 | 合併介面並保留全域分批；local IPA 另驗證 |
| ODE | wrapper 明確傳入 ode_atol／ode_rtol；原保存 50 點 | eval/no_grad/dopri5 保存 2 點，但 wrapper 沒有傳入可設定容差 | 同時保留容差設定及 endpoint 優化；其他求解器與 grad-enabled fallback 保留 |
| 時間窗資料抽樣 | 最後合法起點、恰好等長、短軌跡檢查已修正 | 舊 np.random.choice(arange(N-T)) 邏輯仍在 | 保留 2026 修正；不從 mdgen 覆蓋 dataset；N=T、N<T、最後合法起點需驗證 |
| 前處理 | atlas_dir／sim_dir alias；float16／float32 選擇 | 僅 sim_dir 與 float16 | 沿用 2026 工具與預設 |
| Runtime | deterministic／benchmark guard、manual_attention、model_dir、train-runtime 分工 | 不同 CLI；local checkpoint 相容載入與報告 | 保留 2026 共同入口，新增 local 選項；驗證 checkpoint keys |
| CUDA 幾何 | rot_to_quat 未包含 local 的 CUDA 分塊 | 大批次 eigh 分塊 | 保留分塊；同一幾何實作供 global/local 比較 |
| 幾何訓練 | 無此 geometry_loss 擴充 | 選用 C-N／CA 損失與 clean prediction recovery | 不屬於 local 推論必要相依；列為後續獨立 ticket，首輪保持 2026 loss |
| 分析與依賴 | 有 MDTraj／PyEMMA 相容處理及新版分析工作 | 分析檔與 optional import 的處置不同 | 保留 2026；僅移入 local 所需相容處理，不整檔覆蓋 |

部分差異只是註解、docstring 或格式；不可把 diff 檔案數当作模型能力差異。

## 目標結構

```text
mdgen2026 的共同基底
  checkpoint / dataset / geometry / ODE / FFN / runtime
                       |
            spatial attention 選擇
               /                \
         global SDPA          local SDPA
         所有殘基             core + ghost
                               core-only Q
                               streamed gather
                       |
             共用時間 SDPA 與輸出重建
```

程式整合目標放在 mdgen2026；sibling mdgen 保留作來源與 local 對照。保留 mdgen2026 現有 global 預設，再新增明確 local 開關；local 基準比較關閉 summaries。這是行為與介面的整合，不必覆蓋資料、環境或重寫 Git 歷史。

## 實作順序與驗收

1. 保存當前兩邊快照與來源 hash，固定官方 checkpoint、輸入、FP32、solver、seed。
2. 固定 mdgen2026 現有 global、dataset、前處理、容差 CLI 與 runtime 行為，用現有測試及時間窗邊界案例建立基準。
3. 移入 local 必要模組及 checkpoint 相容載入，接入空間 attention 與 local IPA；保留既有全域 SDPA、全域 IPA 分批、共用時間 SDPA 與主層 FFN 分批。整合版 global 和凍結 2026 比較輸出、輸入梯度、權重梯度；整合版 local 和凍結目前 mdgen local 比較。
4. 移入 CUDA 四元數分塊與 ODE endpoint 優化，保留 2026 可設定容差；新增最終輸出／NFE／fallback 比較。幾何訓練損失獨立處理，不與此輪推論整合混合。
5. 相同單一鄰域、summary off 才測 global/local 等價；一般 local 鄰域不同，不要求與 global 完全相同。
6. 官方 checkpoint 小系統完整推論驗證，再同 L40S 重測 forward／generation。資源與動態品質分開判斷。
7. 依 ask-matt 路線先形成 spec，再切出帶依賴的 tickets；實作逐項 TDD，最後 code-review 以整合前快照核對 Standards 與 Spec。

未新增訓練，未改化學 encoder，未實作顯式 ligand 狀態。這兩個目錄不能直接宣稱為完整 protein+ligand 軌跡模型。

關鍵程式：兩邊 mdgen/model/latent_model.py、model/mha.py、model/ipa.py、wrapper.py、dataset.py、parsing.py、rigid_utils.py；入口 sim_inference.py、train.py、scripts/prep_sims.py；local 擴充 mdgen/full_length/。
