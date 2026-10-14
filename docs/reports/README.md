# 歷史報告

此目錄保存 mdgen2026 與 sibling mdgen 各輪的實作、驗證與量測報告。報告內容記錄當時的執行方式；檔案已於 #3 從根目錄搬移，報告中的指令與內文路徑仍為當時的位置，Markdown 連結已改為新位置。

| 報告內的舊路徑 | 目前位置 |
| --- | --- |
| 根目錄 `*_REPORT*.md`、`CHANGES_vs_base.md`、`FFN_PEAK_MEMORY_RESEARCH.md`、`MDGEN_LOCAL_INTEGRATION_PLAN.md` | `docs/reports/` |
| `my_new/r2`、`my_new/r2.1`、`my_new/t4-cu126`、`my_new/t4-cu130`、`my_new/t5` | `docs/validation/mdgen2026/` |
| `my_new/environment-mdgen2026-*.yml` | `envs/` |
| `my_new/download_atlas.sh` | `scripts/download_atlas.sh` |
| `test_sdpa.py`、`benchmark_l_scaling.py` | `scripts/diagnostics/`（執行前需 `export PYTHONPATH="$PWD"`） |
| `mdgen/model/mha_legacy.py` | 已移除；內容相同的凍結副本為 `tests/reference/global2026/model/mha_legacy.py` |

`docs/validation/neighbor/preexisting_archive_manifest.json` 是 #1 整合時的雜湊快照，鍵值仍為搬移前的路徑；其餘檔案可在新位置以相同 SHA256 驗證；以下檔案於 #3 有意修改，雜湊不再相符：`docs/reports/FFN_PEAK_MEMORY_RESEARCH.md`（程式碼連結改為新相對路徑）、`job/submit_flrh_*.sh` 四個與 `job/submit_mdgen_l7500_*.sh` 兩個 job 腳本（Ticket 2 更新腳本與輸出路徑，Ticket 5 改用 `job/env.sh` 與環境變數）。原始內容可從 #1 合併時的 git 歷史取得。
