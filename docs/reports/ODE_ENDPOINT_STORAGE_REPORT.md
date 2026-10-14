# Dopri5 推論保存起終點：TDD 與同卡資源比較

四個同卡案例全部成功：L5000峰值24.06→19.36GiB，L7500峰值36.00→28.96GiB，皆節省約19.5%。同L的50／2點策略最終座標與序列雜湊完全相同、模型評估次數不變；時間差不到0.3%，主要收益是顯存。

## 行為與修改

[NewMDGenWrapper.inference](/share/home/r15543056/MDgen/mdgen/mdgen/wrapper.py:717)僅在eval且停用梯度、求解器為dopri5時，把保存點數從50改成2。dopri5仍依原本atol1e-6／rtol1e-3自行選擇內部積分步長；2個保存點不代表2次模型評估。訓練、grad-enabled或其他求解器維持50。

Sampler.sample_ode公開預設仍50，沒有改動integrator、模型權重、GELU、attention、FFN分批設定或250個分子軌跡frames。這輪修改發生在sibling mdgen的local實作，不修改mdgen2026全域模型。推論程式原本只取50個保存結果的最後一個，現在只保存起點／終點並仍取終點。

## TDD與正式checkpoint

使用者已同意Sampler.sample_ode回傳函式和NewMDGenWrapper.inference兩個seam。以本輪修改前凍結的inference函式作50點基準；真實模型與求解器照常執行，沒有stub模型輸出。

小fixture的整段生成峰值由GPU四元數前處理主導，50點／候選2點均1099804672bytes。因此顯存預算測試改量ODE／重建區段：重新RED48188兩側172500480bytes，GREEN48189降至156372480bytes（約164.51→149.13MiB，正好少48個state張量），最終結構一致、NFE20/20。大尺寸benchmark仍量整段生成，未排除前處理。

最終job48191：51tests passed、COMPLETED Exit0:0。包括原45tests與新增6tests：推論輸出／NFE／顯存、FP32與FP64解析解比較、training／grad-enabled／rk4 fallback。

正式ATLAS checkpoint104res、250frames比較：保存50/2點、NFE116/116、最後atom14座標maxabsdiff0，逐元素完全相同，序列一致且輸出finite。這是所測fixture結果，不宣稱所有輸入必定逐bit相同。

[測試](/share/home/r15543056/MDgen/mdgen/tests/test_ode_endpoint_storage.py) · [正式checkpoint結果](/share/home/r15543056/MDgen/mdgen/results/ode_validation_48191/checkpoint_parity.json)

## 固定測量條件

Job48192使用node-201實體L40S UUID a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb。ATLAS checkpoint SHA256 bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015。local SDPA、FFN65536tokens/chunk、groupbatch4、summaries off、B1、T250、FP32、TF32 off、seed137、dopri5 atol1e-6／rtol1e-3、shared quaternion前處理均相同。

50／2保存點×L5000／7500共4case，各使用新程序並新跑baseline。完整生成時間包含GPU前處理、ODE與atom14重建；記錄peak allocated／reserved、NFE、finite、OOM stage與traceback。計時與顯存擷取結束後，對全部最終座標與返回序列計算SHA256，核對同L的結果是否完全相同，hash計算不包含在生成時間內。

輸入沿用5e3e_A第一個結構在72.90625Å間距3D格點複製的工程fixture。相同L的inputhash一致。這不是真實5000／7500res蛋白質，不能據此宣稱MD動態品質、真實系統所需顯存或精確L上限。

## 完整生成測量

Job48192已COMPLETED、ExitCode0:0；2026-10-14 00:11:42–02:09:44。4case全成功、0 OOM、0其他錯誤；每個成功輸出均finite。下表時間為整段生成各一次，顯存單位GiB。

| L | ODE保存點 | 時間（秒） | peak allocated | peak reserved | NFE | 狀態 |
|---|---:|---:|---:|---:|---:|---|
|5000|50|1537.47|24.06|34.29|134|成功|
|5000|2|1533.85|19.36|29.39|134|成功|
|7500|50|1888.94|36.00|42.19|110|成功|
|7500|2|1892.18|28.96|37.55|110|成功|

L5000少4.69GiB（19.51%），時間變化-0.24%；L7500少7.04GiB（19.56%），時間變化+0.17%。兩側時間基本相同，單次量測不主張統計上顯著的加速。allocated是活躍tensor峰值，reserved是allocator持有空間，不等於活躍用量。

同L的完整座標與返回序列SHA256完全相同，NFE也相同：L5000為134／134，L7500為110／110。這直接支持所測大尺寸fixture最後答案未變，不只依賴小模型或104res checkpoint比較。

- L5000輸出SHA256：4630edb93b051356e412fb0f546606f9fedcee451fc1e31c3c366debc8654eab。
- L7500輸出SHA256：29c6aaeac5cb27c94ade132eebce3c2c8e826330a26170085e9aa6d9e8d1920f。

![ODE保存方式的時間與顯存比較](/share/home/r15543056/MDgen/mdgen/results/ode_storage_48192/scaling.png)

圖左為完整生成時間，兩條線幾乎重疊；圖右為峰值allocated顯存，橘色2點比藍色50點低約20%。兩者都生成250frames。

[彙總CSV](/share/home/r15543056/MDgen/mdgen/results/ode_storage_48192/summary.csv) · [條件audit](/share/home/r15543056/MDgen/mdgen/results/ode_storage_48192/audit.json)

### 限制與下一步

本輪只做L5000／7500的完整生成資源比較，沒有再量單次forward，因模型forward本身未改動。這是合成工程fixture，不代表已驗證真實大蛋白的MD品質，也不能把L40S結果直接當作其他GPU的容量保證。

目前下一個方向是用真實大系統確認密集鄰域下的顯存與輸出品質；若仍要降低資源，再量測temporal attention／IPA／FFN的剩餘峰值，依量測選擇分批位置。此次沒有縮小鄰域、減少frames、降低FP32精度或改求解容差。

## 重現與原始資料

進入 `/share/home/r15543056/MDgen/mdgen`，執行 `sbatch job/submit_ode_tests.sh`、`sbatch job/submit_ode_storage.sh`。benchmark腳本檢查相同實體GPU UUID，50點baseline透過診斷程式覆蓋保存點數，正式推論eval/no_grad/dopri5預設2點。

[測試job](/share/home/r15543056/MDgen/mdgen/job/submit_ode_tests.sh) · [benchmark job](/share/home/r15543056/MDgen/mdgen/job/submit_ode_storage.sh) · [benchmark](/share/home/r15543056/MDgen/mdgen/scripts/diagnostics/scale_sdpa.py) · [audit／圖表](/share/home/r15543056/MDgen/mdgen/scripts/diagnostics/summarize_ode_storage.py)

原始結果目錄 `/share/home/r15543056/MDgen/mdgen/results/ode_storage_48192/`。

## Review

起點為本輪修改前檔案快照。兩位獨立review agent已完成程式、測試、最終量測與報告的唯讀複核。Standards：總findings 0、未修正0、最高嚴重度無；Spec：總findings 0、未修正0、最高嚴重度無。
