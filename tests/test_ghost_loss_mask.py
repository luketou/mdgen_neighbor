"""Ghost residues enter the model as context but not the loss (#15 Ticket 3)."""
import pytest
import torch

from mdgen.train_step_probe import synthetic_items

TINY = ['--train_split', 'x', '--val_split', 'x', '--data_dir', 'x', '--num_layers', '1',
        '--embed_dim', '32', '--mha_heads', '4', '--ipa_heads', '2', '--ipa_head_dim', '4',
        '--ipa_qk', '2', '--ipa_v', '2', '--num_frames', '4', '--prepend_ipa',
        '--interleave_ipa', '--sim_condition', '--local_spatial', '--spatial_cell_width', '4',
        '--spatial_contact_radius', '8', '--core_capacity', '4', '--ghost_capacity', '20',
        '--group_batch_size', '2']


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    monkeypatch.setenv('MODEL_DIR', str(tmp_path))
    monkeypatch.setattr('sys.argv', ['train.py', *TINY])
    from mdgen.parsing import parse_train_args
    from mdgen.wrapper import NewMDGenWrapper
    args = parse_train_args()
    delattr(args, 'manual_attention')
    torch.manual_seed(0)
    model = NewMDGenWrapper(args)
    with torch.no_grad():  # move the zero-initialised output layer so the loss depends on inputs
        for p in model.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model


def batch(core=None):
    items = synthetic_items(type('A', (), {'num_frames': 4}), 12, 2, seed=0)
    out = torch.utils.data.default_collate(items)
    if core is not None:
        out['core_mask'] = core.expand(2, -1).clone()
    return out


def loss(model, data):
    torch.manual_seed(1)
    model.train()
    return model.general_step(data)


CORE = torch.tensor([1.] * 8 + [0.] * 4)


def test_loss_mask_excludes_ghosts_and_keeps_core(wrapper):
    plain = wrapper.prep_batch(batch())['loss_mask']
    masked = wrapper.prep_batch(batch(CORE))['loss_mask']

    assert torch.equal(masked[:, :, :8], plain[:, :, :8])
    assert masked[:, :, 8:].abs().sum() == 0
    assert plain[:, :, 8:].abs().sum() > 0


def test_all_core_mask_matches_no_core_mask(wrapper):
    assert loss(wrapper, batch(torch.ones(12))) == loss(wrapper, batch())


def test_ghosts_still_provide_context_for_core_loss(wrapper):
    reference = loss(wrapper, batch(CORE))
    moved = batch(CORE)
    moved['trans'][:, :, 9] += 3.0  # residue 9 is a ghost near the core
    assert loss(wrapper, moved) != reference
