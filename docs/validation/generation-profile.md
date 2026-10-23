# 生成速度 profiling（#9 Ticket 8）

`python -m mdgen.full_length.generation_profile` 量測完整長度 FP32 生成：整段 rollout（`NewMDGenWrapper.inference`，含 ODE 求解）的 wall-clock 與 NFE，以及同一輸入上單次 forward 的 torch.profiler 結果，用來判斷瓶頸是否為小 local group 的 kernel launch 開銷。

```bash
python -m mdgen.full_length.generation_profile --sim_ckpt atlas.ckpt --output profile.json \
  (--synthetic_residues 7500 | --pdb_id NAME --data_dir D --split S [--suffix _R1]) \
  [--rollouts 1] [--seed 0] [--ode_atol 1e-6 --ode_rtol 1e-3] [--no_profile] \
  [--group_batch_size ... 其他 local 覆寫]
```

L40S 上用 `job/submit_generation_profile.sh`（預設合成 7,500 殘基鏈，見 `job/README.md`）。實測結果與 JSON 放在 `docs/validation/9-<job>/`，結論附 Slurm job ID。

## 方法

- 一律以 `--local_spatial` 載入 checkpoint（與 `mdgen.full_length.halo_error` 共用載入與檢查），模型 FP32、無 autocast；條件為第 0 幀重複 `num_frames` 次，與 `sim_inference.py` 的 rollout 相同。
- NFE = rollout 期間 `forward_inference` 的呼叫次數；ODE 方法沿用 checkpoint 的 `sampling_method`（報告於 `generation.sampling_method`；dopri5 自行決定步數，含被拒絕的步）。`--num_frames` 預設為 checkpoint 的幀數。
- 先跑一次 forward 暖機，再計時 rollout，接著量一次未 profiling 的 forward（`forward_s`），最後才在 profiler 內跑一次。每層的 `_spatial_attention`／`_spatial_ipa` 以 `mdgen.local_attention`／`mdgen.local_ipa` 標記；這些標記在 GPU 端的區間不算 GPU 工作。
- 合成鏈為無偏隨機漫步，比真實蛋白鬆散（group 較多、較小），結果偏向 launch 受限；下結論前應以真實蛋白（`--pdb_id`）確認。

## 報告欄位

- `prep_s`：一次 `prep_batch`（含建 frame-0 分區）的時間；每段 rollout 內都會再付一次。
- `generation.rollouts[]`：`wall_clock_s`、`nfe`、`s_per_nfe`（整段時間 ÷ NFE，含 `prep_s` 與 atom14 轉換）、`peak_memory_allocated_bytes`；另有 `sampling_method` 與中位數。暖機只跑 forward，第一段 rollout 仍含 ODE 求解器等的首次成本；需要穩定數值時用 `--rollouts 2` 以上看中位數。
- `partition`：`groups`、`core_median`、`ghost_median`、`group_batch_size`、`chunks_per_spatial_call`（每次空間呼叫的 group 批次數）；有 profiling 時另有 `spatial_calls_per_forward`（local attention 與 local IPA 呼叫數）與 `group_chunks_per_forward`。
- `forward_s`：未 profiling 的單次 forward 時間。
- `forward_profile`（僅 GPU 有 kernel 欄位）：
  - `kernel_count`、`kernel_time_us`、`kernel_span_us`（第一個 kernel 開始到最後一個結束）、`gpu_busy_us`（kernel 區間聯集）、`gpu_busy_fraction`、`kernel_median_us`、`kernel_fraction_under_10us`。profiler 的 CPU 開銷會拉大 kernel 間隔，因此另給 `gpu_busy_fraction_of_forward`（`gpu_busy_us` ÷ 未 profiling 的 `forward_s`）；`memory_op_count` 為 memcpy／memset 數，不計入 kernel 統計。
  - `launch_count`：CPU 端 `cudaLaunch*`／`cuLaunch*`／`cudaGraphLaunch*` 呼叫數；`regions.<label>`：`calls`、`host_time_us`、`launch_count`、`launch_fraction`（該區段發出的 launch 佔全部比例）。
  - `top_device_ops`：GPU 時間最多的 15 個 op。

## 判讀

`gpu_busy_fraction` 與 `gpu_busy_fraction_of_forward` 低（GPU 在 forward 期間多半閒置）、`kernel_median_us` 小、且大部分 launch 來自 `mdgen.local_attention`／`mdgen.local_ipa` 時，瓶頸為小 local group 的 kernel launch 開銷，可嘗試加大 `--group_batch_size` 比較；`gpu_busy_fraction` 接近 1 則為計算受限。只報告結果、不設硬門檻。CPU pytest（`tests/test_generation_profile.py`）只驗證 NFE 計數、區段標記與統計計算，實際數值須在 GPU 上量測。
