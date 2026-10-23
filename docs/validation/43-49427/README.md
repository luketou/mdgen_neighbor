# #43 Ticket 1：四蛋白資料驗收（Job 49427）

結論：`6sms_A`、`6lrd_A`、`6l3r_E`、`6sup_A` 共 12 條軌跡完成取得、驗收、前處理與分區，可由正式 `MDGenDataset` 與 `partition_store` 讀取。此結果只代表資料與管線可用，不代表訓練獨立性或任何品質宣稱成立。

## 執行紀錄

- Run ID：`issue43-t1-prepare2-20261023T124139Z-967e437`；Job `49427`，`node-19`，`cpu-2g`，1 CPU、8 GiB，約 4 分鐘，exit 0。
- 快照：commit `967e437`，dirty=true（本 ticket 的未提交變更），310 檔，遠端 SHA-256 manifest 已驗證。
- 指令：`scripts/prepare_atlas_validation.py`，manifest `scripts/atlas_validation_ticket1_manifest.json`。
- 輸出：`/share/home/r15543056/MDgen/mdgen/data/atlas_validation_ticket1_20261023b/`（`arrays/`、`partitions/`、`validation.csv`、`summary.json`、`checkpoint_metadata.json`；512 MiB）。本目錄保存兩份 JSON 的副本。
- 先前失敗：Job `49424`（run `issue43-t1-prepare-20261023T122808Z-967e437`）在 checkpoint 時間設定檢查即停止，未產出資料；其輸出目錄 `atlas_validation_ticket1_20261023/` 只有失敗的 `summary.json`，保留未動。
- 工具測試：Job `49421` 以新測試重現 `prep_sims.py` 缺少 analysis 格式參數（red）；Job `49422` 在修改後通過 `tests/test_atlas_validation_prep.py` 與 `tests/test_chain_numbering.py`，17 passed（green）。Review 後補上反例測試（錯誤時間間隔、序列不符、PDB/XTC 原子數不符時不產出檔案；鏡像檔大小與 SHA-256 不符），Job `49444`（run `issue43-t1-review-fixes-20261023T131749Z-4b0720d`）21 passed。非有限座標的反例未測：XTC 壓縮格式無法可靠寫入 NaN。

## 資料來源

- Hugging Face `sxsaxx/data` revision `a91392e2d08aea15ef19fac00e42c18d97a54c74`，社群鏡像的 ATLAS analysis 版（100 ps），非官方 10 ps protein ZIP。雜湊驗證只證明與鏡像一致，不證明與官方檔案一致。
- 三個基準沿用 Job `49375` 的安裝；`6sup_A` 由本 Job 下載安裝，`installation.json` 記錄各檔 SHA-256 與來源網址。
- `6sup_A` 實測：R1–R3 各 1001 幀、0–100,000 ps、100 ps 間隔，全幀解碼且座標有限；PDB 序列與 manifest 一致（2128 殘基、33,377 原子）。
- `6sup_A/installation.json` 的 `sequence_matches_local_test_split: true` 是錯誤欄位：installer 當時把它寫死，實際只比對了 manifest 序列，`6sup_A` 不在任何 split。之後的 installer 改寫為 `sequence_matches_manifest`；server 上的既有紀錄未改動。
- Job `49362` 使用的官方 ZIP installer（`scripts/install_atlas_validation.py`）因連線逾時從未安裝資料，已從 repo 移除；其快照仍在 run `atlas-test3-install-20261023T101031Z-967e437`。

## 授權與作者

- 四個蛋白的鏡像 `README.txt` 內容相同（SHA-256 `7b3e140e…35527590`）。其中聲明：ATLAS 資料庫以 Creative Commons Attribution-NonCommercial 4.0（CC-BY-NC 4.0）釋出，使用須標示來源且不得商用；商業用途需另行聯絡。
- 作者與聯絡人：Tatiana Galochkina、Jean-Christophe Gelly（Université Paris Cité），ATLAS：https://www.dsimb.inserm.fr/ATLAS。
- 鏡像 repository `sxsaxx/data` 本身的授權標示未查核；授權依據是隨檔附的 ATLAS `README.txt`。

## 時間間隔

- Checkpoint：`/share/home/r15543056/MDgen/mdgen/ckpt/atlas.ckpt`，SHA-256 `bc90c66e…aff08015`，與旁邊 `atlas.source.json` 記錄的 Hugging Face `bjing-mit/mdgen` revision `a92efe5b…` LFS 雜湊一致。
- Checkpoint 只保存 `num_frames=250`、`crop=256`，沒有 `train_split`、`suffix`、`frame_interval` 或物理間隔。
- 選定 400 ps 的依據是上游 `bjing2016/mdgen@81482a4` README 的前處理（stride 40）與原始 ATLAS README「every 10 ps」：10 ps × 40 = 400 ps。100 ps 鏡像因此 stride 4，只降採樣一次（1001 → 251 幀，`--expected_interval_ps 100` 在 stride 前檢查原始間隔）。
- **未證實**：checkpoint 實際以 400 ps 訓練。這是依上游公開流程的推定，記為 `checkpoint_training_interval_ps: null`。

## 拓撲與輸出驗收

| 蛋白 | 殘基 | 鏈 | 編號缺口 | 幾何斷點 | groups / replica | 輸出 shape |
|---|---:|---:|---:|---:|---:|---|
| 6sms_A | 724 | 1 | 0 | 0 | 66 | (251, 724, 14, 3) |
| 6lrd_A | 705 | 1 | 0 | 0 | 70 | (251, 705, 14, 3) |
| 6l3r_E | 669 | 1 | 0 | 0 | 60 | (251, 669, 14, 3) |
| 6sup_A | 2128 | 1 | 0 | 0 | 190 | (251, 2128, 14, 3) |

- 全部殘基為標準殘基且重原子完整（只允許多出 C 端 `OXT`，log 的 `OXT not found` 即此情形）。
- 每條軌跡：float32、座標有限、`_chain.npz` sidecar 與 `_partition.npz` 存在並通過 `load_partition` 一致性檢查；正式 `MDGenDataset` 取 4 幀的 item 全部有限。
- 座標往返檢查（frame 0，3 個 Cα）誤差 0.0 Å：以相同的去氫與剛體對齊重算，確認單位（nm→Å）與幀對應；因採相同運算，不是獨立的對齊驗證。
- 分區參數全部來自目前的 local 預設（cell width 16 Å、contact radius 16 Å、core/ghost capacity 128），checkpoint 未保存這些值。

## 獨立性與重疊

- 上游 split（`bjing2016/mdgen@81482a4`）與本地 `atlas_train/val/test.csv` 雜湊一致（`scripts/atlas_validation_split_audit.json`）。
- 四蛋白都不在 train split 的名稱或完全相同序列中。對全部 train 序列做 local alignment（2/-1/-10/-0.5），每個蛋白最高分的前 3 筆只對齊 0.4–1.7% 的 query 長度，篩檢未發現長片段同源。
- 此篩檢不是完整的 domain／family 審核；checkpoint 的實際訓練成員也無法從檔案得知。四蛋白不認證為獨立測試蛋白。
- `6sup_A` 不在 train/val/test 任何 split，排除原因**未知**，沒有查到可證實的原因。

## 限制

- 三個基準約 700 殘基，可能整個放進一般 Training region 預算；`6sup_A`（2128）是本期唯一的長蛋白系統。
- 單一長蛋白、單鏈、無缺失區段，無法代表多鏈或有斷點的系統。
- Checkpoint 以 `crop=256` 訓練；以完整蛋白長度推論屬外推，Ticket 2 的容量與誤差解讀需考慮這點。
