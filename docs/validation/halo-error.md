# Halo 近似誤差實驗（#12 Ticket 5）

`python -m mdgen.full_length.halo_error` 量測 Training region 的 halo 近似：同一個蛋白先以完整蛋白第 0 幀分區跑一次 forward，再對同一組 loss group 分別加 `--halo_rings` 圈 halo 跑 region forward，兩者共用雜訊與 t，比較 loss core 在未條件化幀上的模型輸出。

```bash
python -m mdgen.full_length.halo_error --sim_ckpt atlas.ckpt --data_dir D --split S \
  --pdb_id NAME --replicate 1 --output report.json \
  [--partition_dir P] [--region_residues 1024] [--halo_rings 1 2] [--regions 4] \
  [--times 0.5] [--seed 0] [--spatial_cell_width ... 其他 local 覆寫]
```

L40S 上用 `job/submit_halo_error.sh`（見 `job/README.md`）。實測結果與 JSON 放在 `docs/validation/12-<job>/`，結論附 Slurm job ID。

## 方法

- loss group 由 `sample_region_groups` 以最深的 `--halo_rings` 選出，因此每種設定都放得進 `--region_residues`；較淺的設定只是 halo 較少，loss group 相同。種子 group 加上最深 halo 就超過預算的 seed 會跳過並記在 `skipped_seeds`；同一蛋白沒有任何 seed 放得下時報錯。跳過的 seed 偏向 halo 較大的 group，結果只代表放得下的 region。
- 每個 region 以 `--seed + region 編號` 選種子 group，選出相同 loss group 的 region 只算一次；模型輸入沿用 checkpoint 的 `frame_interval`、取第 0 幀起的視窗；雜訊以 `--seed` 產生，每個 `--times` 值一份，完整蛋白與所有 region 共用。結果與蛋白在 split 中的順序無關。
- 只比較 loss core 殘基、`x_cond_mask` 為 0 的幀，且只計 loss mask 監督的通道（未定義的 torsion 不計）。逐殘基 L2 = 差值在幀與通道上的 L2；相對誤差 = 該 L2 ÷ 完整蛋白輸出的 L2。
- 邊界圈數：loss group 的 ghost 若由 loss group 以外的 group 擁有，為第 0 圈；ghost 由第 r − 1 圈擁有者為第 r 圈。沒有 ghost 擁有者路徑通到 loss group 以外時（例如整個蛋白都是 loss group）為 `None`。
- 蛋白須讓完整蛋白 forward 裝得下 GPU。

## 報告欄位

- `summary.<k>`：`--halo_rings k` 下所有蛋白、region、t 的 loss 殘基彙整，`l2`／`relative` 各有 `count`、`median`、`max`（只計有限值；完整蛋白輸出為 0 而無法算相對誤差的殘基計入 `nonfinite`）；`by_ring.<r>` 依邊界圈數分組。
- `proteins[]`：`name`、`residues`、`groups`、`summary`、`skipped_seeds`（`seed`、`reason`），以及 `regions[]`：`loss_groups`、`by_halo.<k>`（`halo_groups`、`region_residues`、`loss_residues`、每個 t 的 `l2`／`relative`）。
- `settings`（CLI 參數與模型的 local 參數）、`environment`。

## 判讀

只報告結果、不設硬門檻。結論應寫明 1 圈 halo 是否足夠（例如第 0 圈的相對誤差中位數／最大值，與 2 圈相比是否明顯下降），或建議加深。CPU pytest（`tests/test_halo_error.py`）以隨機小模型確認：halo 覆蓋整個蛋白時誤差為 0、沒有 halo 時邊界誤差大於 0；ATLAS checkpoint 的數值只能在 GPU 上量測。
