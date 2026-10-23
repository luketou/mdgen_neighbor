# Sync before each change: fast-forward only, never touches local uncommitted edits.
# Failures (conflict, offline, diverged) are reported, never block the session.
Set-Location (git rev-parse --show-toplevel)
$branch = git rev-parse --abbrev-ref HEAD
git rev-parse --abbrev-ref '@{u}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "git-pull-guard: $branch has no upstream, skipped"
    exit 0
}
$out = (git pull --ff-only 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Output "git-pull-guard: pull FAILED, needs manual handling: $out"
} else {
    Write-Output "git-pull-guard: $branch $out"
}
exit 0
