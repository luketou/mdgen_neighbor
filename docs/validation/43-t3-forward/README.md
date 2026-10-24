# Ticket 3：三個基準蛋白的成對 forward

依[比較契約 v1.1](../43-contract/frozen-contract-v1.1.md)執行 Ticket 3 的固定矩陣：`6sms_A`、`6lrd_A`、`6l3r_E`，R1–R3，250 幀與 4 幀，t 0.25／0.5／0.75，noise seed 0／1／2，共 162 個 CLI case；每個 case 以同一組 loss groups 比較 1 圈與 2 圈 halo。162 個 case 全部完成，[契約判定](evaluation.json) 的 324 個分層全部 `pass`，`passed` 為 `true`。逐 Job 計數、時間、記憶體與原始 JSON 的 hash 見 [彙整](outcome.json)。

## 結果

- **覆蓋**：每個 protein／replica／halo 的 8 個候選全部可行，同一個完整涵蓋的 region 去重為 1 個，其餘 7 個記為重複；loss 殘基聯集比例為 1.0。324 個分層全為完整涵蓋（250 幀與 4 幀各 162），沒有截斷 case。
- **誤差**：所有分層的 L2、`a/sqrt(N)` 與相對誤差的最大值都是 0.0，即 region 輸出與完整蛋白輸出逐位元相同；全部殘基都是 regular，沒有 zero／near-zero 參考或 no_supervision。
- **有限性**：1458 次量測 forward（含完整蛋白與 region 路徑），模型輸出非有限值總數為 0。
- **資源**：peak reserved 最大 9284091904 bytes（約 8.65 GiB），peak allocated 最大 3472487936 bytes；單張 L40S。每個 case 的時間：250 幀 36.2–97.5 s（97.5 s 是 Job 49590 的第一個 case），4 幀 21.4–33.5 s；每個 Job 的 case 時間合計 13.5–17.3 分鐘。

## 判讀範圍

基準蛋白在 896 預算下整個放進 region，region 與完整蛋白走同一個計算，零誤差是預期結果。這份紀錄只證明正式 Halo 介面、共享輸入、去重、分層與判定在真實資料上一致，**不是** Training region 截斷近似的證據；截斷證據屬 Ticket 4（`6sup_A`）。反例（非有限值、零參考範數、超出門檻、不可行與重複 region、設定不符、缺少結果）由 CPU 工具測試涵蓋，不在此 GPU 紀錄中重現。

[契約判定](evaluation.json) 的 `execution_failures` 列出全部 162 次執行；每一筆的 `failure_class` 都是 `null`、`returncode` 都是 0。

## 執行與來源

| Job | 分批 | case | 狀態 |
|---|---|---|---|
| 49588 | 規劃（CPU，`--plan_only`） | 162 | [manifest](plan/results/manifest.json)、[log](plan/logs/49588.out) |
| 49590 | shard 0／6 | 27 | [manifest](results/halo_error_49590/manifest.json)、[log](logs/49590.out) |
| 49598 | shard 1／6 | 27 | [manifest](results/halo_error_49598/manifest.json)、[log](logs/49598.out) |
| 50303 | shard 2／6 | 27 | [manifest](results/halo_error_50303/manifest.json)、[log](logs/50303.out) |
| 51194 | shard 3／6 | 27 | [manifest](results/halo_error_51194/manifest.json)、[log](logs/51194.out) |
| 51226 | shard 4／6 | 27 | [manifest](results/halo_error_51226/manifest.json)、[log](logs/51226.out) |
| 51229 | shard 5／6 | 27 | [manifest](results/halo_error_51229/manifest.json)、[log](logs/51229.out) |

GPU Job 使用者核准上限為 7 個（6 個 shard 加 1 次重試），每個 1 張 L40S、2 CPU、32 GiB、30 分鐘，逐一執行；重試未使用。六個 GPU Job 共用 run `issue43-t3-forward-20261024T043915Z-a534fe4` 的唯讀快照（commit `a534fe4`），規劃 Job 使用同一 commit 的另一個快照。各 Job 的 preflight（`results/preflight_<job>.json`）均通過，stdout 以 manifest 完成標記結束；終態依據是完成標記與 manifest 的 `driver_finished`，未保存佇列查詢或 `sacct` 輸出。

manifest 保存命令、checkpoint hash（`bc90c66e…`）、輸入檔 hash 與來源檔 hash。原始逐 case JSON 共 150846454 bytes，留在 server 的 run 目錄；[彙整](outcome.json) 的 `raw_reports` 列出每個檔案的遠端路徑、大小與 SHA-256，[契約判定](evaluation.json) 的 `sources` 另記錄判定時讀取的檔案 hash。判定在本機以 `python -m mdgen.full_length.forward_evaluation <raw JSON> --manifest <manifest> --ticket 3` 執行。
