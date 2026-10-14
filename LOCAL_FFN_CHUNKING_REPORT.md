# Local FFN 推論分批：TDD 與同卡 L40S 比較

分批FFN讓本次測試的L=2500、5000、7500全部完成250frames生成。原本完整FFN只有L=2500能完成生成，L=5000/7500在FFN GELU中間張量配置時OOM。主要收益是顯存與可跑尺寸；沒有證據顯示加速。這是工程資源測試，不是MD品質驗證。

## 改了什麼

把 mdgen2026 既有 FFN 分批方式移植到 sibling mdgen 的 local 主層。[實作](/share/home/r15543056/MDgen/mdgen/mdgen/model/latent_model.py:580)每批預設65536 tokens；維持原權重、GELU、FP32與attention設定。FFN對各token分別運算，因此可以分批，並在推論時重用已計算完的buffer，減少擴張到FFN hidden dimension的中間張量。

只在 `eval()` 且停用梯度、token數超過門檻時啟用。訓練或任何需要梯度的呼叫仍走原本完整FFN，不改checkpoint參數鍵值。梯度比較驗證的是此fallback；分批推論本身沒有backward。

## TDD 與數值驗證

公開介面沿用 `LatentMDGenModel.forward_inference`，以本輪修改前的凍結local模型為基準。測試使用非零權重，避免零初始化輸出掩蓋錯誤。

- RED48165：完整/候選兩者workspace都是158799872bytes，顯存目標失敗。
- GREEN48166：158799872→74692096bytes（約151.44→71.23MiB）。初始含RoPE的小fixture由其他workspace主導峰值，因此顯存測試改用無RoPE模型；含RoPE的數值比較與正式checkpoint測試保留。
- 最終48167：45tests passed，包括既有40tests與新增5tests。train/eval、RoPE/plain、非連續輸入、分批尾端均涵蓋。
- 正式ATLAS checkpoint：104res、4frames比較輸入梯度最大差1.3097e-10、289個權重梯度最大差8.1491e-9。
- 正式checkpoint推論：104res、250frames、513tokens/chunk確實啟用分批，最大輸出差9.7752e-6；全部finite。

這支持在FP32容許誤差內等價，不保證逐bit相同，也不證明長軌跡的MD分布品質相同。

[測試](/share/home/r15543056/MDgen/mdgen/tests/test_local_ffn_chunking.py) · [正式checkpoint比較](/share/home/r15543056/MDgen/mdgen/results/ffn_validation_48167/checkpoint_parity.json)

## 固定比較條件

Job48168：node-201同一張NVIDIA L40S，GPU UUID `a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb`，CUDA可見44.39GiB；PyTorch2.11.0+cu128。venv `/share/home/r15543056/.venvs/mdgen` 搭配module python/3.11.12-gpu、cuda/12.8。

兩側皆local SDPA、B1、250frames、FP32、TF32關閉、seed137、summaries關閉、group batch4、相同前處理與輸入。正式 `atlas.ckpt` SHA256：`bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015`。生成固定dopri5、atol1e-6、rtol1e-3。full設定threshold>T×L，chunked設定65536。

每案例獨立新程序。forward暖機2次、5次取median；完整生成各量一次，包含GPU前處理、ODE與atom14結構重建。量測peak allocated與reserved；失敗保留stage、traceback、NFE。NFE為嘗試呼叫次數，OOM時含失敗呼叫。

輸入是104res的5e3e_A第一個結構，複製排列在72.90625Å間距3D格點，再截取至指定L。相同L的兩策略及兩operation使用相同輸入hash。這是資源測量fixture，並非真正5000/7500res蛋白質，也不能據此判定MD物理品質。

## 測量結果

Job48168已COMPLETED、ExitCode0:0，2026-10-13 19:50:17–21:21:25。12案例audit通過：9成功、3個預期full策略OOM、0其他錯誤。chunked六案例皆成功且輸出finite。

### 單次forward

時間為暖機後5次median，顯存為peak allocated，單位GiB。

| L | 完整FFN時間 | 完整FFN顯存 | 分批FFN時間 | 分批FFN顯存 |
|---|---:|---:|---:|---:|
|2500|5.559s|17.26|5.620s|8.49|
|5000|11.429s|34.35|11.470s|16.80|
|7500|OOM（warmup）|40.72†|17.111s|25.12|

L2500/5000的forward顯存下降約50.8%/51.1%；時間增加約1.1%/0.4%，不主張加速。

### 完整生成250frames

每策略每L量一次，包含前處理、ODE、結構重建；時間不能只按L線性推算，因dopri5的NFE也不同。

| L | 完整FFN時間／NFE | 完整FFN顯存 | 分批FFN時間／NFE | 分批FFN顯存 |
|---|---:|---:|---:|---:|
|2500|680.57s／122|20.89|685.67s／122|12.12|
|5000|OOM（第2次模型呼叫）|39.46†|1534.84s／134|24.06|
|7500|OOM（第1次模型呼叫）|37.64†|1885.57s／110|36.00|

L2500完整生成顯存下降42.0%，時間增加0.75%，NFE同為122。L5000/7500的full失敗，不把失敗峰值當成成功生成的所需顯存，也不以此計算節省比例。

†OOM時峰值只是成功配置過的下界，不含下一筆無法配置的記憶體。L5000生成／L7500 forward在GELU的erf配置失敗；L7500生成在GELU的除法中間張量配置失敗，均為full FFN路徑。

L7500分批完整生成的peak reserved為42.19GiB，peak allocated為36.00GiB；兩者意義不同。reserved是PyTorch allocator持有的記憶體，不等於活躍tensor用量。L7500成功輸出shape為[1,250,7500,14,3]且全部finite；只保存資源指標，未另寫PDB/XTC。

![同卡L40S資源比較](/share/home/r15543056/MDgen/mdgen/results/ffn_scaling_48168/scaling.png)

圖中藍色為full、橘色為chunked；上排forward、下排完整生成。右側越低表示顯存越省；OOM叉號是失敗前峰值下界。左下生成時間也受到NFE影響。

[原始彙總CSV](/share/home/r15543056/MDgen/mdgen/results/ffn_scaling_48168/summary.csv) · [條件audit](/share/home/r15543056/MDgen/mdgen/results/ffn_scaling_48168/audit.json)

### 結果的適用範圍與下一步

在本次固定條件與fixture下，成功測到的最大L已從完整生成2500提高至7500；這不是7500為精確上限，因為未測更大L。此次只比較local full/chunked，沒有重跑global或不同精度。訓練仍保留full FFN，不能把此推論容量套用到訓練。

下一步可用真正較大的protein或protein+ligand輸入確認實際鄰域密度、顯存與生成品質；先保留目前checkpoint、求解器與FP32，把實際系統的瓶頸和物理品質分開驗證。這輪沒有新增訓練或改變化學encoder／鄰域定義。

## 重現與原始資料

[提交測試](/share/home/r15543056/MDgen/mdgen/job/submit_ffn_tests.sh) · [提交同卡benchmark](/share/home/r15543056/MDgen/mdgen/job/submit_ffn_scaling.sh) · [量測程式](/share/home/r15543056/MDgen/mdgen/scripts/diagnostics/scale_sdpa.py) · [audit與圖表程式](/share/home/r15543056/MDgen/mdgen/scripts/diagnostics/summarize_ffn_scaling.py)

執行前進入 `/share/home/r15543056/MDgen/mdgen`，使用 `sbatch job/submit_ffn_tests.sh` 與 `sbatch job/submit_ffn_scaling.sh`。後者要求相同GPU UUID，不會自動接受另一張卡作為同卡比較。

結果目錄 `/share/home/r15543056/MDgen/mdgen/results/ffn_scaling_48168/`，含full/chunked JSON、traceback、summary.csv、audit.json、scaling.png/svg。

## Review

本輪修改前快照為起點，全部案例完成後已由兩個獨立review agent唯讀複核。Standards：累計0 findings、未修0；Spec：累計0 findings、未修0。程式、測試、測量條件、原始結果及報告均納入。兩軸均無最嚴重問題。
