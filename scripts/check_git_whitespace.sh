#!/usr/bin/env bash
# Run from the Git working tree to validate its changes and latest commit.
set -euo pipefail

git diff --check
git diff --cached --check
if git rev-parse --verify HEAD^1 >/dev/null 2>&1; then
    # A PR checkout is a merge commit: compare against its first parent (base).
    git diff --check HEAD^1 HEAD
elif [[ "$(git rev-parse --is-shallow-repository)" == true ]]; then
    echo 'Whitespace checks require the parent commit. Fetch at least depth 2 before running checks.' >&2
    exit 1
else
    git show --format= --check HEAD
fi
