# 比較契約 v1（已凍結）

使用者於 2026-10-24 確認：「照最終矩陣草案凍結：長蛋白成對訓練在 64 幀（另保留 4 幀短幀層），250 幀記為第一期不可行；未量測的 replica 標為同形狀未實測；控制容差選 B（梯度相對上限 5e-4、更新 2e-2）。」機器可讀版本見 [frozen-contract-v1.json](frozen-contract-v1.json)。Ticket 3–5 照此版本執行；任何修改都要另立新版本並重新確認，不能依正式結果調整。

## 固定設定

ATLAS checkpoint SHA-256 `bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015`、Ticket 1 資料（`_i400ps`、frame 0 起始視窗）、FP32（matmul `highest`、TF32 關閉）、dropout 0、batch 1、local spatial（cell width／contact radius 16 Å、core／ghost capacity 128、cell summaries 關閉、`group_batch_size` 4、`ffn_chunk_size` 65536）、region 預算 896、暖機 1 次加量測 3 次、單張 L40S。forward 關閉 gradient checkpointing，訓練步開啟。

## 種子 group

CPU `torch.randperm`（generator seed 43002）的前 8 個候選，halo 1 與 2；不可行或重複的候選不補抽。成對 halo 比較以 2 圈預算選 loss groups 再分別加 1／2 圈；`6sup_A` 的 2 圈只有 group 95 可行。全部種子的可行率（`6sup_A` halo1 189/190、halo2 51/190）照實揭露，不宣稱一般 training startup 有效。

## Forward（Ticket 3、4）

| 項目 | 設定 |
|---|---|
| 蛋白 | Ticket 3：`6sms_A`、`6lrd_A`、`6l3r_E`；Ticket 4：`6sup_A` |
| replica | R1–R3 |
| 幀數 | 250（目標層）；4（獨立的短幀診斷層） |
| t／noise seed | 0.25、0.5、0.75／0、1、2 |
| 門檻 | [核准的 forward 門檻](approved-forward.json)：截斷 case 相對誤差與 `a/sqrt(N)` 皆 p95 ≤ 0.01、max ≤ 0.05；完整涵蓋逐殘基 `1e-5` 包絡 |

基準蛋白在 896 預算下全部是完整涵蓋，Ticket 3 只提供工具一致性證據；截斷的 forward 證據來自 Ticket 4。

## 成對訓練（Ticket 5）

| 項目 | `6sup_A` | 基準蛋白 |
|---|---|---|
| 幀數 | 64（長蛋白成對訓練層）；4（短幀層）；250 為**第一期不可行（OOM）** | 250（目標層）；4（短幀層） |
| replica | R1–R3 | R1–R3 |
| region | halo1 的 8 個候選；halo2 只有 group 95 | 只有完整涵蓋 |
| t／noise seed | 0.25、0.5、0.75／0、1、2 | 同左 |

截斷 case 的 loss、clipping 前梯度與實際更新只作描述，不發出「訓練近似通過」。控制（同路徑重跑與完整涵蓋）採選項 B：loss 絕對差 ≤ `2e-6`；每個參數群梯度差 RMS ≤ `1e-7`、更新差 RMS ≤ `2e-6`；高於近零 cutoff 的群組另要求梯度相對差 ≤ `5e-4`、更新相對差 ≤ `2e-2`。近零 cutoff：梯度 reference RMS ≤ `1e-7`、更新 ≤ `2e-6`。依據見 [控制校準](control-calibration.md)。

## 容量狀態

- **已量測可行**：見 [最終矩陣草案](final-matrix-proposal.md) 的容量表。
- **同形狀未實測**：先導時間預算內沒跑到的 case（Job 49479 有 11 個、49501 有 28 個）。batch 形狀與已量測的 R1 相同，但容量沒有實測；正式執行若失敗，依失敗政策記錄。
- **第一期不可行**：`6sup_A` 完整蛋白訓練步超過 64 幀（96 幀與 250 幀的三種記憶體設定均 OOM；[49501](../43-49501/README.md)、[49513](../43-49513/README.md)）。64 與 96 幀之間沒有量測。

## 報告與失敗政策

依蛋白、replica、幀數、t、noise seed、halo、邊界圈數、完整涵蓋／截斷、成對共同子集分層，報 median、p95、max、有效樣本數、zero／near-zero 與 nonfinite，並報 attempted、可行、跳過、重複、去重 region 數與 loss 殘基聯集比例。不以跨蛋白或跨邊界的 pooled 統計判定通過；replica 與幀不當成獨立蛋白或空間樣本。非有限值讓該 case 不通過；OOM、缺 GPU、逾時與程式錯誤分開保留；缺少必要結果的分層標為未完成。

## 已知限制

400 ps 間隔有上游論文記載，但 checkpoint 未保存；checkpoint 以 `crop=256`、250 幀訓練，完整蛋白、64 幀與 4 幀都屬外推；四蛋白與訓練資料的獨立性只做過序列篩檢；結果只描述固定可行子集與單步訓練。
