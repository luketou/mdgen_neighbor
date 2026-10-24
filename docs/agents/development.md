# 本機與 CI 開發流程

## CPU 環境

Python 主／次版本以根目錄 `.python-version` 為準，CI 與 bootstrap 都讀同一檔。先安裝該版本的 Python（可用 `uv python install 3.11`），再於 repo 根目錄執行：

```bash
bash scripts/bootstrap_cpu.sh
bash scripts/check.sh
```

Bootstrap 預設建立持久的 `.venv-cpu`，先安裝 PyTorch CPU wheel，完成後才安裝 `requirements-dev.txt` 並執行 `pip check`。再次執行會重用環境。既有目錄不是 venv、或 Python 版本不符時會失敗，保留既有內容。

`MDGEN_PYTHON` 可指定 Python 執行檔；bootstrap 要求與 `.python-version` 相同。檢查指令也接受此變數。`MDGEN_CPU_VENV` 可指定環境目錄；WSL 下可放在 Linux 檔案系統以減少大量小檔案的 I/O，例如：

```bash
export MDGEN_CPU_VENV="$HOME/.local/share/mdgen/venvs/cpu-py311"
bash scripts/bootstrap_cpu.sh
bash scripts/check.sh
```

Bootstrap 會檢查 Git 作者設定；若尚未設定，請用自己的身分執行 `git config --local user.name ...` 與 `git config --local user.email ...`。它不會猜測或改寫作者。`gh` 的可用性另有提示，備援見[工具指引](tools.md)。

## Windows 主機

CPU 環境與 `scripts/check.sh` 都在 WSL 執行：Windows 的 `python` 沒有 torch，CPU venv 位於 WSL 的 `$HOME/.local/share/mdgen/venvs/cpu-py311`。

要在 WSL 跑檢查的 worktree，以 WSL 的 git 建立，例如 `wsl -d Ubuntu -e git -C <repo> worktree add --no-track -b <branch> ../<name> origin/main`，之後的 commit 也用 WSL 的 git。`--no-track` 讓新分支不追蹤 `origin/main`，否則 `.claude/settings.json` 的 pull guard hook 會在每次提示前把它快轉到 `main`，有本機 commit 後則每次回報 pull 失敗；第一次推送時用 `git push -u origin <branch>` 設定 upstream。Windows git 建立的 worktree，`.git` 記錄 `C:/...` 路徑，WSL 的 git（2.43，不支援相對路徑 worktree）無法讀取，`check_repository.py` 與 `check_git_whitespace.sh` 會直接失敗；反之 WSL 建立的 worktree 在 Windows 的 `git worktree list` 會顯示為 `prunable`，屬預期。

## 共用檢查入口

CI 使用相同的 `bash scripts/bootstrap_cpu.sh` 與 `bash scripts/check.sh`。檢查依序執行：

1. Python 版本、tracked text 的 LF 行尾與 [agent 導航](../../AGENTS.md) 中的本地 Markdown 連結。
2. 透過 `scripts/check_git_whitespace.sh` 檢查工作目錄、index、未被忽略的 untracked 檔案及已 commit 變更的新增空白問題：PR merge checkout 以 base parent 為比較起點；本機分支從與 `origin/main`（可用 `MDGEN_CHECK_BASE` 改）的 merge-base 起涵蓋整條分支，與 CI 檢查相同範圍；找不到 base 時只檢查最新 commit。
3. 關閉 CUDA 裝置可見性；GPU preflight 標記 skipped，再以 `pytest tests -m "not gpu and not checkpoint" -q` 執行 CPU 測試，包含 training probe。資源標記定義見 [README 的 Testing](../../README.md#testing)；GPU／checkpoint 測試改由 [Compute Node 測試入口](../../job/README.md#pytest-測試分類) 執行。
4. 獨立 CPU SDPA equivalence check；每次報告寫入 `results/sdpa/check-*/` 的獨立目錄，保留先前結果，支援重複執行。

任何一項失敗即停止，exit code 非零。開始前與修改後皆使用同一入口；臨時只跑個別測試時，收尾仍須完整執行。CPU 檢查不包含 L40S 容量、速度或正式長蛋白驗收。

CI checkout 使用 `fetch-depth: 2`，確保比較所需的 parent 存在。淺層 checkout 若缺 parent，檢查會要求補抓歷史；只有真正的初始 commit 才檢查整份新增內容，避免把既有凍結基準的空白誤認為本次變更。

## 行尾與 dirty diff

`.gitattributes` 要求文字檔以 LF checkout／儲存，二進位檔由 Git 自動辨識。`scripts/check_repository.py` 會拒絕 tracked text 的 CRLF／mixed 行尾；agent 文件使用可檢查的 Markdown 本地連結。

若既有 checkout 出現大量修改，先比較 `git diff --stat` 與 `git diff --ignore-space-at-eol --stat`，再用 `git ls-files --eol` 確認原因。正規化前先對照 index 的原始 bytes；只轉換確認僅有 CRLF 差異的檔案，保留內容變更。行尾整理與功能變更分開提交。

## 審查

固定比較起點與原始需求後，分開執行 Standards／Spec 審查；規範來源為[開發與寫作規範](conventions.md)。檢查結果應標示 Python／裝置、pass／skip 數與尚待 server 驗證的範圍。
