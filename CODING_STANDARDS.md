# Coding Standards

Review 時逐條套用。機械性規則改寫成測試或 assert，不放這裡。

## 驗證文件（`docs/validation/`）

- 數值差異的歸因（kernel、TF32、累加順序……）要附量測依據；沒有依據就寫「未解釋」。
- 效能或 parity 結論附上 Slurm job ID 與原始 JSON 路徑。
- README 裡的每個數字（含範圍的兩端）都要能在同目錄的 JSON 或 log 找到，且對應到同一個蛋白、job 與設定；逐一對照，不從其他紀錄或 job_history 抄。
- 結論只陳述量到的範圍（GPU 型號、batch size、精度）；推廣到其他設定要另外量。

## Diagnostics 腳本

- Benchmark 的 A/B 兩邊用同一份輸入與同一張卡；輸入的每個張量都要檢查 batch 維度。
