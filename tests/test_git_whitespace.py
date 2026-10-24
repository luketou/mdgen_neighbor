"""Whitespace validation through the same CLI used by local checks and CI."""
from pathlib import Path
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / 'scripts/check_git_whitespace.sh'


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True,
                          capture_output=True, text=True)


def repository(tmp_path):
    root = tmp_path / 'origin'
    root.mkdir()
    git(root, 'init', '-q', '-b', 'main')
    git(root, 'config', 'user.name', 'Regression fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    (root / 'legacy.txt').write_text('existing trailing space \n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Existing legacy whitespace')
    return root


def check(root):
    return subprocess.run(['bash', str(CHECK)], cwd=root,
                          capture_output=True, text=True)


def test_ci_checkout_ignores_unchanged_legacy_whitespace(tmp_path):
    origin = repository(tmp_path)
    (origin / 'legacy.txt').write_text('existing trailing space \nnew clean line\n')
    git(origin, 'commit', '-qam', 'Add a clean line')
    workflow = yaml.safe_load((ROOT / '.github/workflows/tests.yml').read_text())
    checkout = workflow['jobs']['cpu-tests']['steps'][0]
    depth = checkout.get('with', {}).get('fetch-depth', 1)
    clone = tmp_path / 'checkout'
    git(origin, 'clone', '-q', '--depth', str(depth), origin.as_uri(), str(clone))
    result = check(clone)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('bad', [False, True])
def test_untracked_files_are_checked_before_commit(tmp_path, bad):
    # commit 前跑檢查時，新檔仍是 untracked；git diff --check 看不到它們。
    root = repository(tmp_path)
    (root / 'clean.txt').write_text('clean\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Clean commit')
    (root / 'ignored.log').write_text('ignored space \n')
    (root / '.git' / 'info' / 'exclude').write_text('ignored.log\n')
    (root / 'untracked.log').write_text('trailing space \n' if bad else 'clean\n')
    result = check(root)
    assert (result.returncode != 0) == bad, result.stdout + result.stderr
    assert 'ignored.log' not in result.stdout
    if bad:
        assert 'untracked.log:1: trailing whitespace' in result.stdout


@pytest.mark.parametrize('location', ['worktree', 'index', 'commit'])
def test_new_whitespace_is_rejected_at_each_git_boundary(tmp_path, location):
    root = repository(tmp_path)
    (root / 'new.txt').write_text('clean line\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Add clean tracked file')
    (root / 'new.txt').write_text('new trailing space \n')
    if location in ('index', 'commit'):
        git(root, 'add', 'new.txt')
    if location == 'commit':
        git(root, 'commit', '-qm', 'Introduce whitespace')
    result = check(root)
    assert result.returncode != 0
    assert 'new.txt:1: trailing whitespace' in result.stdout


@pytest.mark.parametrize('bad', [False, True])
def test_pr_merge_checks_changes_against_base_parent(tmp_path, bad):
    root = repository(tmp_path)
    git(root, 'checkout', '-qb', 'feature')
    (root / 'feature.txt').write_text('new line \n' if bad else 'new line\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Feature change')
    git(root, 'checkout', '-q', 'main')
    (root / 'base.txt').write_text('base change\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Base change')
    git(root, 'merge', '-q', '--no-ff', '--no-edit', 'feature')
    clone = tmp_path / 'pr-merge'
    git(root, 'clone', '-q', '--depth', '2', root.as_uri(), str(clone))
    result = check(clone)
    assert (result.returncode != 0) == bad, result.stdout + result.stderr
    if bad:
        assert 'feature.txt:1: trailing whitespace' in result.stdout


@pytest.mark.parametrize('bad', [False, True])
def test_local_branch_checks_every_commit_since_main(tmp_path, bad):
    # CI 比對 PR merge 與 base；本機也要涵蓋整條分支，不只最新 commit。
    origin = repository(tmp_path)
    clone = tmp_path / 'local'
    git(origin, 'clone', '-q', origin.as_uri(), str(clone))
    git(clone, 'config', 'user.name', 'Regression fixture')
    git(clone, 'config', 'user.email', 'fixture@example.invalid')
    git(clone, 'checkout', '-qb', 'feature')
    (clone / 'early.txt').write_text('early line \n' if bad else 'early line\n')
    git(clone, 'add', '.')
    git(clone, 'commit', '-qm', 'Early branch commit')
    (clone / 'late.txt').write_text('clean\n')
    git(clone, 'add', '.')
    git(clone, 'commit', '-qm', 'Latest clean commit')
    result = check(clone)
    assert (result.returncode != 0) == bad, result.stdout + result.stderr
    assert 'legacy.txt' not in result.stdout
    if bad:
        assert 'early.txt:1: trailing whitespace' in result.stdout


def test_depth_one_checkout_reports_missing_parent_without_legacy_noise(tmp_path):
    root = repository(tmp_path)
    (root / 'new.txt').write_text('clean\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Clean change')
    clone = tmp_path / 'too-shallow'
    git(root, 'clone', '-q', '--depth', '1', root.as_uri(), str(clone))
    result = check(clone)
    assert result.returncode != 0
    assert 'Fetch at least depth 2' in result.stderr
    assert 'legacy.txt' not in result.stdout
