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
