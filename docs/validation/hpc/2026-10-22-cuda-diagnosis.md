# CI 與 L40S 診斷紀錄

CI 修正：PR #17，commit a0045d04f39b7aa5073df814a008847dbcb8ac88。
GitHub Actions run 39809148702 / job 119930338976 成功。
本機 100 passed、8 skipped；雲端 PR merge（含 main bc58eb8）103 passed、8 skipped。兩者 CPU SDPA 6/6 通過。
原因：depth-1 checkout 將 HEAD 當成 root，空白檢查誤報既有檔案。
修正：fetch-depth 2，第一個 parent diff，7 個真實 Git CLI 回歸測試通過。
Standards / Spec 審查均無 findings。

## CUDA 最小重現

node-201，l40s account/partition，1 GPU、1 CPU、2 GiB、2 分鐘上限。
Job 49048：python/3.11.12-gpu module + 專案既有 .venvs/mdgen2026 解譯器，PyTorch 2.11.0+cu128。
2×2 CUDA 矩陣乘法失敗，CUDA unknown error。
Job 49051（run cuda-variants-20261022T155708Z）：baseline、只 unset CUDA_VISIBLE_DEVICES、只清理 LD_LIBRARY_PATH，均相同失敗。診斷腳本完成，不代表 GPU 驗證通過。

- SLURM_JOB_GPUS=0，GPU_DEVICE_ORDINAL=0，CUDA_VISIBLE_DEVICES=0。
- 載入 module 前 nvidia-smi -L 已回報 No devices found。
- 分配到的 /dev/nvidia0：open(O_RDWR) 回傳 EIO / Input/output error。
- 未分配的 /dev/nvidia1：EPERM；不據此推論另一張卡是否故障。
- /dev/nvidiactl 與 nvidia-uvm 可開啟。
- 使用真實 /usr/lib64/libcuda.so.580.95.05，kernel driver 亦為 580.95.05。
- 此證據將故障定位至 PyTorch 以下的 GPU/驅動裝置存取層，尚無法區分驅動狀態與硬體原因。

需 server 管理端檢查 node-201 的分配 GPU 0、kernel NVIDIA/Xid 記錄與裝置健康；未重設 GPU、重載驅動、改系統 module 或動其他使用者作業。
GPU 恢復後，以同一最小 probe 先確認矩陣乘法通過，再執行專案訓練步測試。完整 L40S 模型與精度驗證尚未完成。

## 快照與產物

遠端：/share/home/r15543056/MDgen/codex-runs/cuda-variants-20261022T155708Z
原始程式快照仍位於上述遠端 run；本次保存的 JSON：[baseline](2026-10-22-baseline.json)、[移除可見性設定](2026-10-22-unset-visible.json)、[清理函式庫路徑](2026-10-22-clean-library.json)。
兩個程式檔傳輸前後 SHA-256 相同；快照與輸出分離。前一次 baseline 快照亦保留於 cuda-diag-20261022T155551Z。

Compute Node 重跑命令（須先透過 Slurm 取得 GPU，REPORT 指向可寫的新輸出路徑）：

```bash
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu
REPORT=/path/to/new/report.json /share/home/r15543056/.venvs/mdgen2026/bin/python probe.py
```

上述命令描述當時快照的 probe.py，需於該快照 code 目錄執行。

Probe 在 CUDA 上建立全 1 的 2×2 tensor，相乘並同步，斷言結果每個元素等於 2；失敗時輸出 JSON 並回傳非零。
