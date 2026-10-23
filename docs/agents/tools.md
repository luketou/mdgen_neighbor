# 工具查找與 GitHub 備援

## 分兩步發現工具

需要查找工具時，先按服務與動作篩選 metadata，只回傳名稱；選定後才讀該工具的完整 schema。一次保留足夠判斷的少量候選，避免將全部工具描述輸出到對話後截斷重查。

在提供 `ALL_TOOLS` 的執行環境，可用：

```javascript
text(ALL_TOOLS.filter(t => /github.*(search_issues|fetch_issue|fetch_issue_comments)$/.test(t.name))
  .map(t => t.name));
```

接著用選定的完整名稱查詢單一工具：

```javascript
text(ALL_TOOLS.find(t => t.name === 'mcp__codex_apps__github_fetch_issue'));
```

依回傳 schema 呼叫；有可用 connector 時，優先使用其能力。其他服務也依相同的名稱 → 單一 schema 流程查找，名稱以本次環境實際回傳為準。

## GitHub 存取

目的 repo 固定為 `luketou/mdgen_neighbor`。Issue 與 comments 必須一起讀，才能辨識已完成但尚未合併的 ticket；並核對本機分支，避免重做遠端已完成工作。

若使用 `gh`，明確傳入 `--repo luketou/mdgen_neighbor`；例如：

```bash
gh issue view 9 --comments --repo luketou/mdgen_neighbor
```

Windows PowerShell 5.1 不會把 here-string 正確 pipe 給外部程式的 stdin；`git commit -F -` 會把整段訊息當成 pathspec。多行的 commit 訊息、PR 或 issue 內文先寫成 UTF-8 檔案，再用 `git commit -F <檔案>`、`gh pr create --body-file <檔案>`、`gh issue comment --body-file <檔案>`。`gh api` 的留言內文同樣寫成 JSON 檔再用 `--input <檔案>`，不在雙引號字串裡組；PowerShell 會轉義其中的反引號與 `$`。

PowerShell 5.1 傳給外部程式的參數會吃掉內嵌的雙引號。`gh api graphql` 的查詢字串不放雙引號，值改用變數傳入，例如 `-f query='query($o:String!){...}' -f o=luketou`。

Windows 的 `gh` 位於 `C:\Program Files\GitHub CLI\gh.exe`；在它安裝前就開的 shell 找不到它，需重新載入 PATH 或直接用完整路徑。

`gh` 不存在、未登入或無法存取時，改用已連線 GitHub connector 的對應 read／search 工具，明確指定同一 repo。寫入操作沿用使用者授權範圍，並遵守[issue 規則](issue-tracker.md)。兩種管道都不可用時，說明缺少的存取能力，保留尚未核對的狀態。
