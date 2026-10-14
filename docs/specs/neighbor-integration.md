# feat: 將 local 推論整合至 mdgen2026

## Problem Statement
使用者需要在同一個 mdgen2026 中選擇 global 與 local 計算；目前 local 功能在 sibling mdgen，兩者的資料、runtime 與推論設定已分岔。

## Solution
以當前 mdgen2026 為共同基底，移入 neighbor-cell/core/ghost、core-only Query、逐批 gather、local IPA、相容 checkpoint 載入、CUDA 四元數分塊與 ODE endpoint 保存。整合結果發布到 luketou/mdgen_neighbor。

## User Stories
1. 作為模型使用者，我要預設維持 global SDPA，確保既有指令與模型可使用。
2. 作為模型使用者，我要明確選擇 local 空間運算，處理較大系統。
3. 作為研究者，我要繼續沿完整時間軸計算 SDPA，保留時間資訊。
4. 作為模型使用者，我要重用官方 checkpoint 權重，避免重新訓練才能推論。
5. 作為研究者，我要看見新增、缺少或 shape 不相容的權重報告，避免靜默錯載。
6. 作為模型使用者，我要每個殘基只有一個 core owner，避免重複輸出。
7. 作為模型使用者，我要 ghost 提供資訊且梯度可回傳，保留邊界資訊。
8. 作為模型使用者，我要必要同鏈編號鄰居不被容量截斷，避免靜默遺失。
9. 作為模型使用者，我要 local 推論沿用 core-only Query 和逐批 gather，控制工作顯存。
10. 作為研究者，我要保留 global IPA 分批與主層 FFN 分批，避免移植造成容量退化。
11. 作為研究者，我要訓練與需要梯度時保留可微分路徑。
12. 作為模型使用者，我要 dopri5 推論少存中間結果但仍產生完整 frames。
13. 作為研究者，我要保留可設定 atol/rtol，並核對最终輸出及 NFE。
14. 作為模型使用者，我要大型 CUDA 四元數轉換分塊，避免 backend 批次限制。
15. 作為資料使用者，我要保留合法時間窗、短軌跡檢查與 NPY dtype 選項。
16. 作為研究者，我要 global 和凍結 2026、local 和凍結 mdgen 比較輸出及梯度。
17. 作為研究者，我要同實體 L40S、FP32、250 frames 下量測時間、顯存及 OOM。
18. 作為程式維護者，我要公開可重現測試、文件與資源結果，而非提交大型資料。

## Implementation Decisions
- 原生全域 MHA SDPA、manual fallback、position_ids 拒絕語意及 runtime 分工保留。
- Local 僅限制空間鄰域；時間 SDPA 沿用 2026 實作。Global/local 不要求不同鄰域下完全等價。
- Local adapter 重用既有 projection、RoPE、mask、bias K/V；不得雙重 scaling。
- Global 為預設；local summaries 預設關閉。必要新 summary 權重可初始化並報告，其他缺少／不相容權重需拒絕。
- 分區依初始條件建立；編號鄰居仍為 fallback，不宣稱真 topology 或完整多鏈 CSV 支援。
- Eval/no_grad/dopri5 保存起終點；其他情況保留 50 點，Sampler 公開預設不改；atol/rtol 不改。
- 不把另一目錄的舊 dataset、分析或 CLI 整檔覆蓋過來。
- 發布至指定空 repository，保留來源歷史及既有授權；不提交 checkpoint、資料、暫存或完整生成產物。

## Testing Decisions
- 公開介面沿用使用者先前確認的 LatentMDGenModel.forward_inference、run_pretrained_local_attention、Sampler.sample_ode 回傳函式與 NewMDGenWrapper.inference。
- 比較真實模型與求解器，使用非零權重、非均勻 cotangent；檢查輸出、輸入梯度及所有使用權重梯度。
- 凍結整合前兩邊原始碼，記錄 SHA256；基準不依賴候選修改。
- 每個 ticket 依序 RED→GREEN，不先寫全部測試；官方 checkpoint 小系統再驗證。
- 同卡工程量測分開 forward／generation，固定輸入、checkpoint、FP32、TF32 off、250 frames、seed 137、dopri5 與容差；記錄 allocated/reserved、NFE、finite、OOM 階段。
- 大尺寸 fixture 是合成資源測試，不能推論真實大蛋白或 MD 品質；單次量測不主張統計加速。
- Review 起點為本輪修改前快照，Standards 與 Spec 分別審查。

## Out of Scope
化學 encoder、顯式 ligand、幾何 loss 移植、新微調、鄰域重建策略、訓練容量宣稱與物理品質改善。

## Further Notes
使用 module 加現有 venv。四個依序 tickets：①雙基準/global回歸；②local/checkpoint/CLI；③CUDA分塊/ODE endpoints；④同卡量測/review/發布。②依賴①，③依賴②，④依賴③。使用者已確認此拆分。

## Tickets（使用者已確認）
- [x] Ticket 1：建立雙基準與 global 回歸；Blocked by：無。保留兩邊 source hash，global 輸出／梯度符合凍結 2026。
- [x] Ticket 2：local 模型、checkpoint 與 CLI 整合；Blocked by：Ticket 1。同一入口選 local，輸出／梯度符合凍結 mdgen，global 保持預設。
- [x] Ticket 3：CUDA 四元數分塊與 ODE endpoints；Blocked by：Ticket 2。容差、最後輸出與 NFE 不變，fallback 正確。
- [ ] Ticket 4：同卡 L40S、雙軸 review 與 GitHub 發布；Blocked by：Ticket 3。保存量測條件、處理 findings、推送可重現整合版本。
