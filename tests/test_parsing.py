"""Shared CLI helpers used by training and inference entry points."""
import argparse
from types import SimpleNamespace

import pytest

from mdgen.parsing import (LOCAL_DEFAULTS, add_determinism_args, add_local_override_args,
    apply_local_overrides, check_determinism_args, global_mode_override_error)


def parser():
    parser = argparse.ArgumentParser()
    add_local_override_args(parser)
    add_determinism_args(parser)
    return parser


def test_local_overrides_keep_checkpoint_values_unless_given():
    saved = SimpleNamespace(local_spatial=True, core_capacity=64)
    cli = parser().parse_args(['--ghost_capacity', '32', '--no-cell_summaries'])
    apply_local_overrides(saved, cli)
    assert saved.local_spatial is True
    assert saved.core_capacity == 64
    assert saved.ghost_capacity == 32
    assert saved.cell_summaries is False
    assert saved.group_batch_size == LOCAL_DEFAULTS['group_batch_size']


def test_local_override_can_disable_checkpoint_local_spatial():
    saved = SimpleNamespace(local_spatial=True)
    apply_local_overrides(saved, parser().parse_args(['--no-local_spatial']))
    assert saved.local_spatial is False


def test_determinism_and_benchmark_are_mutually_exclusive():
    p = parser()
    check_determinism_args(p, p.parse_args(['--deterministic']))
    with pytest.raises(SystemExit):
        check_determinism_args(p, p.parse_args(['--deterministic', '--benchmark']))


def test_ffn_chunk_size_flag_overrides_checkpoint_value():
    saved = SimpleNamespace(ffn_chunk_size=65536)
    apply_local_overrides(saved, parser().parse_args(['--ffn_chunk_size', '4096']))
    assert saved.ffn_chunk_size == 4096


def test_ffn_chunk_size_unset_keeps_checkpoint_value():
    saved = SimpleNamespace(ffn_chunk_size=12345)
    apply_local_overrides(saved, parser().parse_args([]))
    assert saved.ffn_chunk_size == 12345


@pytest.mark.parametrize('value', ['0', '-8'])
def test_ffn_chunk_size_rejects_non_positive(value):
    with pytest.raises(SystemExit):
        parser().parse_args(['--ffn_chunk_size', value])


@pytest.mark.parametrize('local', [False, True])
def test_ffn_chunk_size_rejected_in_global_mode(local):
    cli = parser().parse_args(['--ffn_chunk_size', '4096'])
    saved = apply_local_overrides(SimpleNamespace(local_spatial=local), cli)
    error = global_mode_override_error(saved, cli)
    assert (error is not None) == (not local)


def test_global_mode_without_ffn_chunk_size_is_allowed():
    cli = parser().parse_args([])
    saved = apply_local_overrides(SimpleNamespace(local_spatial=False), cli)
    assert global_mode_override_error(saved, cli) is None
