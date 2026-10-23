"""Snapshot selection keeps test dependencies while excluding runtime assets."""
import subprocess

from scripts import hpc_run


def test_snapshot_keeps_tracked_reference_sources_without_checkpoints(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    sources = {'mdgen/__init__.py', 'tests/reference/global2026/__init__.py',
               'tests/reference/localbaseline/__init__.py',
               'tests/reference/wrapper_before_ode_endpoints.py'}
    assets = {'ckpt/atlas.ckpt', 'data/trajectory.npy', '.env'}
    for name in sources | assets:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    subprocess.run(['git', 'add', '--force', '--', *sorted(sources | assets)],
                   cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(hpc_run, 'ROOT', tmp_path)

    selected = set(hpc_run.select_files())

    assert sources <= selected
    assert not (assets & selected)
