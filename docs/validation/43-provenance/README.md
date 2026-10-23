# #43 來源查核：ATLAS checkpoint 的幀間隔、`6sup_A` 的 split 與 `crop=256`

結論：上游作者的論文與 README 明文寫出 ATLAS 模型以 400 ps（每 40 幀取 1 幀、250 幀）訓練，Hugging Face 權重即論文所用；`6sup_A` 不在 split 中，符合 AlphaFlow 的規則：測試期內序列長度超過 1024 的蛋白排除；上游推論不裁切，直接以全長生成，但公開評估中最長的測試蛋白只有 724 殘基。本地 checkpoint 的設定也確認關閉絕對位置編碼（`abs_pos_emb: false`）。除讀取 checkpoint 設定的一個 CPU Job 外，本查核只讀文獻與原始碼。

分級：**已證實**＝有原文或原始碼可直接引用；**推定**＝由已證實的事實推論，但沒有原文直接寫明；**查無**＝已查範圍內找不到。

本文的數字都來自同目錄的 [`audit.json`](audit.json)：其中記錄輸入（上游 commit、RCSB API、論文網址）、split 統計、RCSB 日期、檔案雜湊、原始碼行號與論文原文引句。`audit.json` 由 [`make_audit.py`](make_audit.py) 產生：

```bash
python -I make_audit.py <bjing2016/mdgen clone> <bjing2016/alphaflow clone> audit.json
```

論文引句無法由程式重抄，是 2026-10-23 從 arXiv HTML 版人工摘錄後寫進 `make_audit.py`。

## 查核範圍

- `bjing2016/mdgen@81482a403b91c5a8437046da1d8d321ba97089cc`（2026-10-23 的 HEAD）：`README.md`、`scripts/prep_sims.py`、`mdgen/dataset.py`、`mdgen/parsing.py`、`mdgen/model/latent_model.py`、`sim_inference.py`、`splits/atlas*.csv`。
- `bjing2016/alphaflow@0408d7c89dac444a43a9089d7427ce470b0a5e67`：`splits/atlas*.csv`、`scripts/download_atlas.sh`、`scripts/prep_atlas.py`、`README.md`；split 檔最後一次修改是 `f62d8eae`（2024-02-12「dataset and training」）。
- MDGen 論文 Jing, Stärk, Jaakkola, Berger，[arXiv:2409.17808](https://arxiv.org/abs/2409.17808)（HTML 版，2026-10-23 讀取）。
- AlphaFlow 論文 Jing, Berger, Jaakkola，[arXiv:2402.04845](https://arxiv.org/abs/2402.04845)（HTML 版，2026-10-23 讀取；只讀到 Appendix B.2，之後的附錄未讀）。
- RCSB Data API `https://data.rcsb.org/rest/v1/core/entry/{id}` 的 `deposit_date` 與 `initial_release_date`。
- Checkpoint 內的 `hyper_parameters`：Job `49502`（run `issue43-ckpt-hparams-20261023T163215Z-0b955a3`，`node-19`，CPU，snapshot commit `0b955a3`，dirty=false）以 [`dump_ckpt_hparams.py`](dump_ckpt_hparams.py) 讀取 `/share/home/r15543056/MDgen/mdgen/ckpt/atlas.ckpt`，輸出 [`atlas_hparams_49502.json`](atlas_hparams_49502.json) 與 log [`49502.out`](49502.out)；檔案 SHA-256 與 Ticket 1 記錄的相同。
- 未查：Hugging Face `bjing-mit/mdgen` 的 model card、上游 GitHub issues。

## 1. Checkpoint 的訓練幀間隔

| 項目 | 分級 | 來源 |
|---|---|---|
| ATLAS 原始軌跡每條 100 ns、10k 幀，即 10 ps／幀 | 已證實 | AlphaFlow Appendix B.2：「3 replicate simulations of length 100 ns are provided, each with 10k frames」；Ticket 1 讀到的 ATLAS `README.txt` 亦寫「every 10 ps」（見 [43-49427](../43-49427/README.md#時間間隔)） |
| 上游下載的是 10 ps 的 `_protein.zip` | 已證實 | [`alphaflow/scripts/download_atlas.sh`](https://github.com/bjing2016/alphaflow/blob/0408d7c89dac444a43a9089d7427ce470b0a5e67/scripts/download_atlas.sh) 第 2 行（`${name}_protein.zip`）；mdgen README 第 25 行指向此腳本 |
| 訓練每 40 幀取 1 幀，得 250 幀、400 ps | 已證實 | MDGen §4.4：「we generate samples with 250 frames and 400 ps timestep, such that a single sample emulates the 100 ns ATLAS reference trajectory」；Appendix B.3：「subsample every 40 frames, obtaining a training target with 250 frames」 |
| 公開的前處理與訓練指令一致 | 已證實 | [README 第 39–40、60 行](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/README.md)：`# Prep with interval 40 * 10 ps = 400 ps`、`--stride 40 --suffix _i40`；訓練 `--num_frames 250 --crop 256 --atlas --suffix _i40` |
| Hugging Face 權重就是論文所用的模型 | 已證實（作者聲明） | README 第 65 行：「The model weights used in the paper are available on HuggingFace: https://huggingface.co/bjing-mit/mdgen」 |
| 本地 `atlas.ckpt` 實際以 400 ps 訓練 | 推定（高度可信） | 上兩列，加上 Ticket 1 已驗證本地檔與 HF revision 的 LFS 雜湊一致、checkpoint 存的 `num_frames`、`crop` 與論文相符（見 [43-49427](../43-49427/README.md#時間間隔)）；checkpoint 本身沒有存間隔，無法從檔案獨立證明 |

補充：

- README 的 ATLAS 前處理指令無法照抄執行：缺 `--atlas`，非 ATLAS 分支讀 `args.atlas_dir`，parser 只定義 `--sim_dir`（[`scripts/prep_sims.py`](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/scripts/prep_sims.py)）。所以作者實際執行的指令與 README 不完全相同；但 stride 40 另有論文 B.3 獨立佐證，不影響結論。
- 10,001 幀 stride 40 得 251 幀。`mdgen/dataset.py` 第 40 行 `np.random.choice(np.arange(arr.shape[0] - num_frames))` 在 251 幀、`num_frames=250` 時只有 1 個可選起點（0），所以訓練永遠用第 0–249 幀（0–99.6 ns）。本地 100 ps 鏡像 1001 幀 stride 4 也得 251 幀，起點條件相同。

## 2. `6sup_A` 不在 train/val/test split

| 項目 | 分級 | 來源 |
|---|---|---|
| mdgen 的四個 ATLAS split 檔與 AlphaFlow 的逐檔相同 | 已證實 | SHA-256 相同（例如 `atlas_test.csv` `2be92a2f…11ba57e188`）；兩個 repo 的 `splits/atlas*.csv` |
| `6sup_A` 在完整清單 `splits/atlas.csv`（第 1303 行），只是不在三個 split | 已證實 | `splits/atlas.csv` 有 1390 筆；train 1266 + val 39 + test 82 = 1387；不在任何 split 的只有 `6j33_B`（1053 殘基）、`6qfk_A`（1357）、`6sup_A`（2128） |
| Split 依 PDB 釋出日期切分：train 早於 2018-05-01，val 為 2018-05-01 至 2019-05-01，test 為之後 | 已證實 | AlphaFlow §4.1：「Using training and validation cutoffs of May 1, 2018 and May 1, 2019」；CSV 實際日期：train 最晚 2018-04-25、val 2018-05-23 至 2019-04-10、test 2019-05-01 至 2022-03-30 |
| CSV 的 `release_date` 是 RCSB 的 initial release date，不是 deposit date | 已證實 | 抽查 `6sms_A`、`6lrd_A`、`6l3r_E`、`6o2v_A`、`1k5n_A` 五筆皆與 RCSB `initial_release_date` 相同；`6o2v_A` deposit 2019-02-24 卻在 test，所以 AlphaFlow B.2 的「deposited after May 1, 2019」措辭不精確，實際依釋出日期 |
| 測試期內長度超過 1024 的蛋白被排除 | 已證實（規則） | AlphaFlow Appendix B.2：test split 為「all 84 targets whose corresponding PDB entries were deposited after May 1, 2019, minus the two targets with sequence length greater than 1024」；§4.1：「(2 excluded due to length)」 |
| `6sup_A` 因此規則被排除 | 推定（高度可信） | `6sup` 於 2019-12-04 釋出，落在 test 期；長度 2128 > 1024；test 最長 724，train 卻有 1025–1144 殘基的蛋白（共 4 筆），表示長度上限只套用在 test。論文沒有列出被排除的名稱 |
| 論文的數目與 CSV 對不上 | 已證實（不一致） | 論文：1265/39/82、排除 2、test 期 84 筆。CSV：1266/39/82、test 期內 85 筆（`6j33_B` 2019-09-18、`6qfk_A` 2019-11-06、`6sup_A` 2019-12-04 釋出，三者都超過 1024）。差異原因查無 |

## 3. `crop=256` 與長蛋白推論

| 項目 | 分級 | 來源 |
|---|---|---|
| 訓練時長於 256 的蛋白隨機取連續 256 殘基；短於 256 的補零並以 mask 遮蔽 | 已證實 | [`mdgen/dataset.py`](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/mdgen/dataset.py) 第 71 行（`if L > self.args.crop`）與第 80 行（`elif L < self.args.crop`）；MDGen B.3：「We train with random crops of up to 256 residues」 |
| 推論不裁切，以全長序列生成 | 已證實 | MDGen B.3：「but draw samples for the full protein at inference time」；[`sim_inference.py`](https://github.com/bjing2016/mdgen/blob/81482a403b91c5a8437046da1d8d321ba97089cc/sim_inference.py) 第 41 行 `get_batch` 以 `L = len(seqres)` 建 batch，沒有任何 crop 或分塊 |
| ATLAS 模型沒有綁定 `crop` 長度的絕對位置編碼 | 已證實 | 本地 checkpoint 的 `hyper_parameters.args.abs_pos_emb` 為 `false`（`atlas_hparams_49502.json`）；只有開啟時才建 `(1, crop, embed_dim)` 的 `pos_embed` 並加到輸入（`mdgen/model/latent_model.py` 第 115、235 行）。與 README 的 ATLAS 訓練指令（沒有 `--abs_pos_emb`）一致 |
| 殘基間的位置資訊來自 rotary embedding | 已證實（設定） | 同一 JSON 的 `no_rope` 為 `false`，各層以 `use_rotary_embeddings=not args.no_rope` 建 attention（`mdgen/model/latent_model.py` 第 82、98 行）。Rotary 只編碼相對位置，但訓練時的 crop 只涵蓋 256 殘基內的相對距離；更遠距離的行為未經評估 |
| 上游公開評估的最長蛋白為 724 殘基 | 已證實 | `splits/atlas_test.csv` 的 `seqlen` 最大 724（即 `6sms_A`）；MDGen Table 4 為 n=82 的 test |
| 上游對 1024 殘基以上蛋白的推論行為 | 查無 | 論文與原始碼都沒有說明或評估 |

## 對 `docs/validation/43-49427/README.md` 的影響

同一 PR 已依本查核更新該檔三處陳述，各附連結回本文對應段落：

- 「時間間隔」：原「**未證實**：checkpoint 實際以 400 ps 訓練」改為作者已證實，只保留 checkpoint 檔未保存間隔的缺口；`checkpoint_training_interval_ps: null` 不變，表示檔案內無此值。
- 「獨立性與重疊」：原「排除原因**未知**」改為符合 AlphaFlow 的長度排除規則，並註明論文排除數與 CSV 不一致。
- 「限制」：外推陳述補上上游公開評估的最長長度；`6sup_A`（2128）約為該長度的 2.94 倍，而且正好屬於上游因長度排除的那類蛋白。

未改的結論：`6sup_A` 因釋出日期晚於 train 截止日而不在 train split，這個時間切分已證實；但 domain／family 層級的獨立性仍未審核。
