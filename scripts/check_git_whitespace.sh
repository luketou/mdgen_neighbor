#!/usr/bin/env bash
# Run from the Git working tree to validate its changes and committed branch.
set -euo pipefail

git diff --check
git diff --cached --check
# New files are still untracked when checks run before commit; git diff skips them.
untracked_bad=0
while IFS= read -r -d '' path; do
    report="$(git diff --no-index --check /dev/null "$path" || true)"
    if [[ -n "$report" ]]; then
        printf '%s\n' "$report"
        untracked_bad=1
    fi
done < <(git ls-files --others --exclude-standard -z)
if (( untracked_bad )); then
    exit 2
fi
base_ref="${MDGEN_CHECK_BASE:-origin/main}"
if git rev-parse --verify -q HEAD^2 >/dev/null; then
    # A PR checkout is a merge commit: compare against its first parent (base).
    git diff --check HEAD^1 HEAD
elif base="$(git merge-base "$base_ref" HEAD 2>/dev/null)" && [[ "$base" != "$(git rev-parse HEAD)" ]]; then
    # A local branch: cover every commit since main, as CI does for the PR.
    git diff --check "$base" HEAD
elif git rev-parse --verify -q HEAD^1 >/dev/null; then
    git diff --check HEAD^1 HEAD
elif [[ "$(git rev-parse --is-shallow-repository)" == true ]]; then
    echo 'Whitespace checks require the parent commit. Fetch at least depth 2 before running checks.' >&2
    exit 1
else
    git show --format= --check HEAD
fi
