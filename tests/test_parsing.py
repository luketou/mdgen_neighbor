"""Shared CLI helpers used by training and inference entry points."""
import argparse
from types import SimpleNamespace

import pytest

from mdgen.parsing import (LOCAL_DEFAULTS, add_determinism_args, add_local_override_args,
    apply_local_overrides, check_determinism_args)


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
