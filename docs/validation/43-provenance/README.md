# #43 來源查核：ATLAS checkpoint 的幀間隔、`6sup_A` 的 split 與 `crop=256`

結論：上游作者的論文與 README 明文寫出 ATLAS 模型以 400 ps（每 40 幀取 1 幀、250 幀）訓練，Hugging Face 權重即論文所用；`6sup_A` 不在 split 中，符合 AlphaFlow 的規則：測試期內序列長度超過 1024 的蛋白排除；上游推論不裁切，直接以全長生成，但公開評估中最長的測試蛋白只有 724 殘基。本查核只讀文獻與原始碼，未讀 checkpoint 內容、未送 HPC Job。

分級：**已證實**＝有原文或原始碼可直接引用；**推定**＝由已證實的事實推論，但沒有原文直接寫明；**查無**＝已查範圍內找不到。

## 查核範圍

- `bjing2016/mdgen@81482a403b91c5a8437046da1d8d321ba97089cc`（2026-10-23 的 HEAD）：`README.md`、`scripts/prep_sims.py`、`mdgen/dataset.py`、`mdgen/parsing.py`、`mdgen/model/latent_model.py`、`sim_inference.py`、`splits/atlas*.csv`。
- `bjing2016/alphaflow@0408d7c89dac444a43a9089d7427ce470b0a5e67`：`splits/atlas*.csv`、`scripts/download_atlas.sh`、`scripts/prep_atlas.py`、`README.md`；split 檔最後一次修改是 `f62d8eae`（2024-02-12「dataset and training」）。
- MDGen 論文 Jing, Stärk, Jaakkola, Berger，[arXiv:2409.17808](https://arxiv.org/abs/2409.17808)（HTML 版，2026-10-23 讀取）。
- AlphaFlow 論文 Jing, Berger, Jaakkola，[arXiv:2402.04845](https://arxiv.org/abs/2402.04845)（HTML 版，2026-10-23 讀取；只讀到 Appendix B.2，之後的附錄未讀）。
- RCSB Data API `https://data.rcsb.org/rest/v1/core/entry/{id}` 的 `deposit_date` 與 `initial_release_date`。
- 未查：Hugging Face `bjing-mit/mdgen` 的 model card、checkpoint 內的 `hyper_parameters`、上游 GitHub issues。

## 1. Checkpoint 的訓練幀間隔

| 項目 | 分級 | 來源 |
|---|---|---|
| ATLAS 原始軌跡每條 100 ns、10k 幀，即 10 ps／幀 | 已證實 | AlphaFlow Appendix B.2：「3 replicate simulations of length 100 ns are provided, each with 10k frames」；Ticket 1 讀到的 ATLAS `README.txt`「every 10 ps」 |
| 上游下載的是 10 ps 的 `_protein.zip` | 已證實 | [`alphaflow/scripts/download_atlas.sh`](https://github.com/bjing2016/alphaflow/blob/0408d7c89dac444a43a9089d7427ce470b0a5e67/scripts/download_atlas.sh)（`${name}_protein.zip`）；mdgen README 第 25 行指向此腳本 |
| 訓練每 40 幀取 1 幀，得 250 幀、400 ps | 已證實 | MDGen §4.4：「we generate samples with 250 frames and 400 ps timestep, such that a single sample emulates the 100 ns ATLAS reference trajectory」；Appendix B.3：「subsample every 40 frames, obtaining a training target with 250 frames」 |
| 公開的前處理與訓練指令一致 | 已證實 | [README 第 39–40、60 行](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/README.md)：`# Prep with interval 40 * 10 ps = 400 ps`、`--stride 40 --suffix _i40`；訓練 `--num_frames 250 --crop 256 --atlas --suffix _i40` |
| Hugging Face 權重就是論文所用的模型 | 已證實（作者聲明） | README 第 65 行：「The model weights used in the paper are available on HuggingFace: https://huggingface.co/bjing-mit/mdgen」 |
| 本地 `atlas.ckpt` 實際以 400 ps 訓練 | 推定（高度可信） | 上兩列加上本地雜湊與 HF revision `a92efe5b…` 一致，且 checkpoint 存的 `num_frames=250`、`crop=256` 與論文相符；checkpoint 本身沒有存間隔，無法從檔案獨立證明 |

補充：

- README 的 ATLAS 前處理指令無法照抄執行：缺 `--atlas`，非 ATLAS 分支讀 `args.atlas_dir`，parser 只定義 `--sim_dir`（[`scripts/prep_sims.py`](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/scripts/prep_sims.py)）。所以作者實際執行的指令與 README 不完全相同；但 stride 40 另有論文 B.3 獨立佐證，不影響結論。
- 10,001 幀 stride 40 得 251 幀。`mdgen/dataset.py` 第 40 行 `np.random.choice(np.arange(arr.shape[0] - num_frames))` 在 251 幀、`num_frames=250` 時只能取 0，所以訓練永遠用第 0–249 幀（0–99.6 ns）。本地 100 ps 鏡像 stride 4 也得 251 幀，起點條件相同。

## 2. `6sup_A` 不在 train/val/test split

| 項目 | 分級 | 來源 |
|---|---|---|
| mdgen 的四個 ATLAS split 檔與 AlphaFlow 的逐檔相同 | 已證實 | SHA-256 相同（例如 `atlas_test.csv` `2BE92A2F…11BA57E188`）；兩個 repo 的 `splits/atlas*.csv` |
| `6sup_A` 在完整清單 `splits/atlas.csv`（第 1303 行），只是不在三個 split | 已證實 | `splits/atlas.csv` 有 1390 筆；train 1266 + val 39 + test 82 = 1387；不在任何 split 的只有 `6j33_B`（1053 殘基）、`6qfk_A`（1357）、`6sup_A`（2128） |
| Split 依 PDB 釋出日期切分：train 早於 2018-05-01，val 為 2018-05-01 至 2019-05-01，test 為之後 | 已證實 | AlphaFlow §4.1：「Using training and validation cutoffs of May 1, 2018 and May 1, 2019」；CSV 實際日期：train 最晚 2018-04-25、val 2018-05-23 至 2019-04-10、test 2019-05-01 至 2022-03-30 |
| CSV 的 `release_date` 是 RCSB 的 initial release date，不是 deposit date | 已證實 | 抽查 `6sms_A`、`6lrd_A`、`6l3r_E`、`6o2v_A`、`1k5n_A` 五筆皆與 RCSB `initial_release_date` 相同；`6o2v_A` deposit 2019-02-24 卻在 test，所以 AlphaFlow B.2 的「deposited after May 1, 2019」措辭不精確，實際依釋出日期 |
| 測試期內長度超過 1024 的蛋白被排除 | 已證實（規則） | AlphaFlow Appendix B.2：test split 為「all 84 targets whose corresponding PDB entries were deposited after May 1, 2019, minus the two targets with sequence length greater than 1024」；§4.1：「(2 excluded due to length)」 |
| `6sup_A` 因此規則被排除 | 推定（高度可信） | `6sup` 於 2019-12-04 釋出，落在 test 期；長度 2128 > 1024；test 最長 724，train 卻有 1025–1144 殘基的蛋白，表示長度上限只套用在 test。論文沒有列出被排除的名稱 |
| 論文的數目與 CSV 對不上 | 已證實（不一致） | 論文：1265/39/82、排除 2、test 期 84 筆。CSV：1266/39/82、test 期內 85 筆（`6j33_B` 2019-09-18、`6qfk_A` 2019-11-06、`6sup_A` 2019-12-04 釋出，三者都超過 1024）。差異原因查無 |

## 3. `crop=256` 與長蛋白推論

| 項目 | 分級 | 來源 |
|---|---|---|
| 訓練時長於 256 的蛋白隨機取連續 256 殘基；短於 256 的補零並以 mask 遮蔽 | 已證實 | [`mdgen/dataset.py` 第 70–89 行](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/mdgen/dataset.py)；MDGen B.3：「We train with random crops of up to 256 residues」 |
| 推論不裁切，以全長序列生成 | 已證實 | MDGen B.3：「but draw samples for the full protein at inference time」；[`sim_inference.py`](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/sim_inference.py) 的 `get_batch` 以 `L = len(seqres)` 建 batch，沒有任何 crop 或分塊 |
| ATLAS 模型沒有綁定 `crop` 長度的絕對位置編碼 | 推定 | `--abs_pos_emb` 預設關閉（`mdgen/parsing.py` 第 96 行 `store_true`），只有開啟時才建 `(1, crop, embed_dim)` 的 `pos_embed`（`mdgen/model/latent_model.py` 第 115–117、234–235 行）；README 的 ATLAS 訓練指令沒有 `--abs_pos_emb`（四肽指令都有）。本地 checkpoint 的 `hyper_parameters.abs_pos_emb` 未讀取 |
| 上游公開評估的最長蛋白為 724 殘基 | 已證實 | `splits/atlas_test.csv` 的 `seqlen` 最大 724（即 `6sms_A`）；MDGen Table 4 為 n=82 的 test |
| 上游對 1024 殘基以上蛋白的推論行為 | 查無 | 論文與原始碼都沒有說明或評估 |

## 對 `docs/validation/43-49427/README.md` 的影響

本次查核不改動該檔；建議另行更新以下陳述：

- 「時間間隔」第 33 行：「**未證實**：checkpoint 實際以 400 ps 訓練」可改為：作者論文（MDGen §4.4、B.3）與 README 明文寫出 400 ps、每 40 幀取 1 幀，且 README 聲明 HF 權重即論文模型；剩下的缺口只是 checkpoint 檔案未保存間隔。`checkpoint_training_interval_ps: null` 可保留，表示檔案內無此值。
- 「獨立性與重疊」第 54 行：「排除原因**未知**」可改為：符合 AlphaFlow 對 test 期內長度超過 1024 蛋白的排除規則（AlphaFlow §4.1、B.2）；論文未列名稱，且排除數（2）與 CSV（3）不一致。
- 「限制」第 60 行的外推陳述應加強：上游不僅以 256 殘基裁切訓練，公開評估也只到 724 殘基；`6sup_A`（2128）約為已評估最長蛋白的 2.9 倍，而且正好屬於上游因長度刻意排除的那類蛋白。
- 「獨立性」的結論不變：`6sup_A` 因釋出日期晚於 train 截止日而不在 train split，這個時間切分已證實；但 domain／family 層級的獨立性仍未審核。
