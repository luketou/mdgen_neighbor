# Local temporal SDPA 與同卡 L40S L-scaling

狀態：已完成。job 48152 結束；24 個正式案例的控制條件與有限值 audit 通過，18 個成功、6 個 OOM、0 個其他 error。

## 結論

在相同 checkpoint、FP32、250 frames、完整 FFN 與同一實體 L40S 下，global/local 的單次 forward 都成功到測過的 L=5000、L=7500 都 OOM；完整生成都成功到 L=2500、L=5000/7500 都 OOM。這只是已測尺寸的最大成功值，不是精確臨界值。

本次 local 時間 SDPA 的輸出與梯度驗證通過，但目前 local 的端到端速度仍慢於 global，總峰值顯存幾乎相同。完整生成的大尺寸 OOM traceback 指向 FFN 的 GELU 暫存張量；下一個容量優化應優先處理 FFN 分批，速度瓶頸則仍需 profiling 區分 gather、ghost K/V 重複處理與多次 kernel 呼叫。

## 修改與 TDD

- production 只改 sibling mdgen 的 `mdgen/full_length/attention.py` 與 `mdgen/model/latent_model.py`：加入既有權重的全序列 SDPA adapter，讓 local 時間 attention 預設使用 SDPA，沿用 Q scaling、RoPE、bias KV、mask、dropout 與 out projection。未新增可學習參數；既有 manual 選項保留。
- RED job 48145：輸出／梯度通過，但時間 attention 工作顯存 old=new=39,719,424 bytes，節省要求失敗。GREEN job 48147：40 tests passed。
- 250 frames 小模型的整合測試比較輸出、輸入梯度與所有可用權重梯度；工作顯存 39,719,424 → 19,624,448 bytes（49.41%）。這不是完整 ATLAS 模型的總顯存。
- 正式 ATLAS checkpoint 的梯度比較使用 104 殘基、4 frames：輸出最大差 9.954e-6、輸入梯度 1.630e-9、289 組權重梯度最大差 9.686e-8，通過預定容許誤差。這不是大 L backward 容量測試。
- 104 殘基／250 frames profiler 確認 45 次 SDPA 呼叫，包含 5 層時間 attention；此輸入的時間沒有混入下列 scaling 表。

## 固定條件與測量方法

GPU：NVIDIA L40S，UUID `a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb`，CUDA 可見顯存 44.39 GiB；torch 2.11.0+cu128。官方 ATLAS checkpoint SHA256 `bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015`；B=1、T=250、FP32、TF32 關閉、seed=137、summaries 關閉。global 使用 mdgen2026，local 使用 sibling mdgen。

輸入是將真實 5e3e_A 的第一個結構複製到分離的三維格點，裁切到各 L，250 個 conditioning frames 重複同一結構。每個 L 的兩種模式、兩種操作均核對原始輸入 SHA256。這是工程資源測試，不是實際大蛋白或 protein+ligand 的生成品質／MD 一致性驗證，也不能直接推論 RTX 4090 容量。

單次 forward 在 CPU 準備 conditioning，再移到 GPU，eval/no_grad、暖機 2 次、測 5 次取中位數。完整生成則使用正常 wrapper.inference，含 GPU 前處理、dopri5 ODE 與 atom14 結構重建，測 1 次，atol=1e-6、rtol=1e-3。NFE 是模型呼叫次數；local/global 向量場不同，採相同自適應容差不保證 NFE 相同。

顯存表列 peak allocated（GiB）；reserved、baseline、OOM elapsed、NFE 與 traceback 保存在原始 JSON/CSV。成功輸出的 finite 檢查在計時與峰值擷取後進行；較早成功案例另做獨立驗證，驗證時間不取代正式量測。

## 單次 forward

| L | Global 秒 | Global GiB | Local 秒 | Local GiB |
|---:|---:|---:|---:|---:|
| 256 | 0.211 | 1.90 | 0.582 | 1.92 |
| 500 | 0.432 | 3.56 | 1.174 | 3.59 |
| 1000 | 0.968 | 6.98 | 2.323 | 7.00 |
| 2500 | 3.295 | 17.24 | 5.580 | 17.26 |
| 5000 | 9.181 | 34.33 | 11.425 | 34.35 |
| 7500 | OOM（暖機） | 40.70 下界 | OOM（暖機） | 40.72 下界 |

## 完整生成：250 frames

| L | Global 秒 | Global GiB | NFE | Local 秒 | Local GiB | NFE |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 24.93 | 4.34 | 116 | 78.63 | 4.37 | 134 |
| 500 | 53.22 | 4.36 | 122 | 137.28 | 4.38 | 116 |
| 1000 | 130.72 | 8.43 | 134 | 284.37 | 8.46 | 122 |
| 2500 | 416.12 | 20.87 | 128 | 681.93 | 20.89 | 122 |
| 5000 | OOM | 39.44 下界 | 2 | OOM | 39.46 下界 | 2 |
| 7500 | OOM | 37.67 下界 | 1 | OOM | 37.64 下界 | 1 |

兩種模式的 L5000 完整生成在 model_forward、第 2 次模型呼叫時，FFN GELU 的 torch.erf(...) 結果 allocation OOM，要求額外 7.15 GiB；L7500 在第 1 次呼叫，GELU 的 x / sqrt(2) 暫存配置 OOM，要求額外 10.73 GiB。它們已通過 GPU 前處理，失敗點沒有指向 SDPA attention。OOM 的已配置峰值只是下界；L7500 更早失敗，因此峰值比 L5000 小，不表示需求較小。

## 圖表解讀

左側是時間，越低越快；右側是峰值 allocated，越低越省。藍色 global、橙色 local；顯存曲線幾乎重疊。叉號代表 OOM 前已成功配置的下界，不能當作完成所需顯存。上排單次 forward，下排完整生成，兩者不可互相代替。

![同卡 L-scaling](/share/home/r15543056/MDgen/mdgen/results/sdpa_scaling_48148/scaling.png)

## 必須保留的比較條件修正

1. mdgen2026 已有 inference FFN chunking（65,536 tokens），local 原本使用完整 FFN。初始 global L≥500 的優勢包含 FFN 分批效果，先前把 global L7500 成功／local OOM 直接解讀成 attention 容量差是不成立的。受影響資料封存於 unmatched_ffn_chunking/，正式 global 案例已補跑；兩者均使用完整 FFN。local 與 global L256 原本就是完整 FFN，保留原始時間並記錄來源驗證。production FFN 未修改。
2. local 在本輪前已有 CUDA rot_to_quat 分塊，global 沒有；原生 global 完整生成 L256/500 在 torch.linalg.eigh 發生 cuSOLVER INVALID_VALUE，NFE=0。正式生成對兩者採 shared_chunked：載入同一既有函式、核對來源 SHA256，仍走正常 inference；production geometry 未修改。原生結果留在 native_preprocessing/。
3. 有限 4×4 單位矩陣的 backend probe：batch 26,000 成功且 eigenvalues 正確；32,767、32,768、64,000 都出現 cuSOLVER INVALID_VALUE。這支持目前環境存在大批次 backend 限制，不能把該錯誤直接歸因於 NaN，也沒有測出精確臨界值。

## 下一步

優先把 mdgen2026 的 inference FFN chunking 移到 local，先比較完整 FFN 與分批 FFN 的輸出，再在同卡重測 L5000/7500 forward 與完整生成。若要支援訓練，再單獨驗證輸入／權重梯度與 backward 顯存。速度方面，用 profiler 量出時間 attention、local gather、ghost K/V、IPA 與 FFN 的比例後，再選優化點；此輪不把全部 local 慢速都歸因於 ghost。

## 證據與 review

- [測試與正式 checkpoint log](/share/home/r15543056/MDgen/mdgen/logs/core_final_48147.out)
- [規格與本輪快照起點](/share/home/r15543056/MDgen/mdgen/docs/specs/temporal-sdpa-l-scaling.md)
- [控制條件 audit](/share/home/r15543056/MDgen/mdgen/results/sdpa_scaling_48148/audit.json)
- [完整數據 CSV](/share/home/r15543056/MDgen/mdgen/results/sdpa_scaling_48148/summary.csv)
- [原始 JSON、封存資料及獨立驗證](/share/home/r15543056/MDgen/mdgen/results/sdpa_scaling_48148/)
- [生成與後處理 job](/share/home/r15543056/MDgen/mdgen/job/submit_sdpa_generation_resume.sh)

Review 起點使用本輪修改前快照。Standards：歷史 2 項 P2 與 1 項有限值建議已處理；Spec：1 項 P2 已處理。最終數據與本報告的最後唯讀複查已通過，兩軸均無未修正問題。
