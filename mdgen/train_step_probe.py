"""One complete Lightning training step with a JSON evidence report (#9 Seam 1).

Usage:
    python -m mdgen.train_step_probe --output report.json \
        [--configs fp32-reference bf16-train] [--seed 0] \
        [--synthetic_residues L | --dataset_index I] -- <train.py arguments>

Every configuration runs the same batch from the same initial weights and seed;
gradient cosines are reported for each later configuration against the first.
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import pytorch_lightning as pl

from .dataset import collate
from .precision import configure_precision, precision_state

LIGHTNING_PRECISION = {'fp32-reference': '32-true', 'bf16-train': 'bf16-mixed'}


def parameter_group(name):
    """`layers.0.mha_t.w` -> `layers.0.mha_t`; `t_embedder.mlp.0.w` -> `t_embedder`."""
    parts = name.split('.')
    return '.'.join(parts[:3]) if len(parts) > 3 and parts[1].isdigit() else parts[0]


def group_gradients(named_gradients):
    groups = {}
    for name, grad in named_gradients.items():
        groups.setdefault(parameter_group(name), []).append(grad.reshape(-1))
    return {key: torch.cat(value) for key, value in groups.items()}


def gradient_summary(groups):
    return {key: {
        'numel': grad.numel(),
        'norm': float(grad.double().norm()) if grad.isfinite().all() else None,
        'max_abs': float(grad.abs().max()) if grad.isfinite().all() else None,
        'nonfinite': int((~grad.isfinite()).sum()),
    } for key, grad in groups.items()}


def gradient_cosine(reference, candidate):
    """Per-group cosine; None when either side is non-finite, 1 when both are zero."""
    out = {}
    for key, a in reference.items():
        a, b = a.double(), candidate[key].double()
        if not (a.isfinite().all() and b.isfinite().all()):
            out[key] = None
            continue
        na, nb = a.norm(), b.norm()
        if na == 0 or nb == 0:
            out[key] = 1.0 if na == nb else 0.0
        else:
            out[key] = float((a @ b / (na * nb)).clamp(-1, 1))
    return out


def synthetic_items(args, residues, count, seed):
    """Finite random-walk backbones with valid rotations and unit torsion vectors."""
    generator = torch.Generator().manual_seed(seed)
    items = []
    for i in range(count):
        T = args.num_frames
        step = torch.randn(T, residues, 3, generator=generator)
        trans = (3.8 * step / step.norm(dim=-1, keepdim=True)).cumsum(1)
        rots, _ = torch.linalg.qr(torch.randn(T, residues, 3, 3, generator=generator))
        rots = rots * torch.linalg.det(rots)[..., None, None].sign()
        torsions = torch.randn(T, residues, 7, 2, generator=generator)
        items.append({
            'name': f'synthetic_{i}',
            'trans': trans,
            'rots': rots,
            'torsions': torsions / torsions.norm(dim=-1, keepdim=True),
            'torsion_mask': torch.ones(residues, 7),
            'seqres': torch.randint(0, 20, (residues,), generator=generator),
            'mask': torch.ones(residues),
        })
    return items


def synthetic_region_items(args, items, seed):
    """Cut a Training region from each synthetic item's frame-0 full-protein partition."""
    from .full_length.region_batch import assemble_region, sample_region
    from .full_length.partition import PartitionParams, build_spatial_partition
    regions = []
    for i, item in enumerate(items):
        L = len(item['seqres'])
        chain = {'chain_id': torch.zeros(L, dtype=torch.long), 'residue_index': torch.arange(L)}
        ca = item['trans'][0]
        partition = build_spatial_partition(ca, mask=torch.ones(L, dtype=torch.bool), **chain,
                                            **PartitionParams.from_args(args).partition_kwargs())
        selection, region = sample_region(partition, ca, args,
                                          torch.Generator().manual_seed(seed + i))
        fields = {**item, **{key: value.numpy() for key, value in chain.items()}}
        regions.append(assemble_region(fields, selection, region, args.region_residues))
    return regions


def dataset_items(args, index):
    from .dataset import MDGenDataset
    dataset = MDGenDataset(args, split=args.train_split)
    return [dataset[i] for i in range(index, index + args.batch_size)]


def describe_batch(items):
    first = items[0]
    return {
        'names': [item.get('name') for item in items],
        'batch_size': len(items),
        'frame_starts': [int(item['frame_start']) if 'frame_start' in item else None for item in items],
        'shapes': {key: list(value.shape) for key, value in first.items()
                   if hasattr(value, 'shape')},
        **({'regions': [item['region'] for item in items]} if 'region' in first else {}),
    }


def environment():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            cwd=os.path.dirname(__file__), check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cuda = torch.cuda.is_available()
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'pytorch_lightning': pl.__version__,
        'numpy': np.__version__,
        'cuda_available': cuda,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if cuda else None,
        'device_name': torch.cuda.get_device_name() if cuda else None,
        'git_commit': commit,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    }


def batch_shapes(batch, batch_size, prefix=''):
    """Shapes of every tensor in a collated batch; each must lead with batch_size.

    Lists (names, per-item Training region partitions) must hold batch_size
    entries; partitions report their stacked `residue_mask` shape.
    """
    shapes = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            shapes.update(batch_shapes(value, batch_size, f'{prefix}{key}.'))
        elif isinstance(value, list):
            if len(value) != batch_size:
                raise ValueError(f'{prefix}{key} has {len(value)} entries; '
                                 f'expected batch dimension {batch_size}')
            masks = [item.residue_mask for item in value if hasattr(item, 'residue_mask')]
            if masks:
                if len({tuple(mask.shape) for mask in masks}) != 1:
                    raise ValueError(f'{prefix}{key} partitions differ in residue count')
                shapes[f'{prefix}{key}.residue_mask'] = [len(masks), *masks[0].shape]
        elif torch.is_tensor(value):
            if value.dim() == 0 or value.shape[0] != batch_size:
                raise ValueError(f'{prefix}{key} has shape {list(value.shape)}; '
                                 f'expected batch dimension {batch_size}')
            shapes[prefix + key] = list(value.shape)
    return shapes


class StepProbe(pl.Callback):
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.result = {}

    def _sync(self, pl_module):
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize(pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.result['precision_state'] = precision_state(self.name, trainer, pl_module)
        self.result['batch_shapes'] = batch_shapes(batch, self.batch_size)
        self._sync(pl_module)
        if pl_module.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.start = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Raw gradients after backward, before clipping and the optimizer step.
        named = dict(pl_module.model.named_parameters())
        self.result['missing_gradients'] = [n for n, p in named.items()
                                            if p.requires_grad and p.grad is None]
        self.gradients = {n: p.grad.detach().float().cpu().clone()
                          for n, p in named.items() if p.grad is not None}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync(pl_module)
        self.result['step_time_s'] = time.perf_counter() - self.start
        cuda = pl_module.device.type == 'cuda'
        self.result['peak_memory_allocated_bytes'] = (
            torch.cuda.max_memory_allocated(pl_module.device) if cuda else None)
        self.result['peak_memory_reserved_bytes'] = (
            torch.cuda.max_memory_reserved(pl_module.device) if cuda else None)
        loss = outputs['loss'].detach().float()
        self.result['loss'] = float(loss) if loss.isfinite() else None
        self.result['loss_finite'] = bool(loss.isfinite())
        recorded_time = pl_module._log.get('train_time', [])
        self.result['training_time'] = float(np.mean(recorded_time)) if len(recorded_time) else None
        self.result['parameters_nonfinite_after_step'] = sum(
            int((~p.detach().isfinite()).sum()) for p in pl_module.model.parameters())


def run_step(args, items, name, seed, initial_state=None, use_sdpa=True):
    """One fit step of the real `NewMDGenWrapper`; returns (report, grouped gradients, weights)."""
    from .wrapper import NewMDGenWrapper
    args.precision_config, args.precision = name, LIGHTNING_PRECISION[name]
    configure_precision(name)
    pl.seed_everything(seed, verbose=False)
    model = NewMDGenWrapper(args, use_sdpa=use_sdpa)
    if initial_state is None:
        if getattr(args, 'sim_ckpt', None):
            from .full_length.checkpoint import load_compatible_state_dict
            checkpoint = torch.load(args.sim_ckpt, map_location='cpu', weights_only=False)
            loaded = load_compatible_state_dict(model, checkpoint)
            missing = [n for n in loaded['newly_initialized'] if '.cell_summaries.' not in n]
            if missing or loaded['shape_mismatch'] or loaded['unexpected']:
                raise ValueError(f'Checkpoint incompatible: {loaded}')
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        model.load_state_dict(initial_state)
    probe = StepProbe(name, len(items))
    initial_norm = float(torch.stack([p.detach().double().square().sum()
                                     for p in model.parameters()]).sum().sqrt())
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1,
        precision=args.precision, max_steps=1, num_sanity_val_steps=0,
        limit_val_batches=0, gradient_clip_val=args.grad_clip,
        deterministic=args.deterministic, benchmark=args.benchmark,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, callbacks=[probe])
    # Fixed items keep the batch identical; reseeding fixes t, noise and dropout.
    pl.seed_everything(seed, verbose=False)
    trainer.fit(model, torch.utils.data.DataLoader(items, batch_size=len(items),
                                                   collate_fn=collate))
    groups = group_gradients(probe.gradients)
    summary = gradient_summary(groups)
    report = {
        'precision_config': name,
        'initial_parameter_norm': initial_norm,
        **probe.result,
        'grad_checkpointing': bool(args.grad_checkpointing),
        'gradient_nonfinite': sum(g['nonfinite'] for g in summary.values()),
        'gradients': summary,
    }
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, groups, initial_state


def run_probe(args, items, configs, seed, use_sdpa=True, cosine_threshold=0.99):
    runs, grads, initial_state = [], [], None
    for name in configs:
        report, groups, initial_state = run_step(
            args, items, name, seed, initial_state, use_sdpa=use_sdpa)
        runs.append(report)
        grads.append(groups)
    comparisons = []
    for name, report, groups in zip(configs[1:], runs[1:], grads[1:]):
        cosine = gradient_cosine(grads[0], groups)
        finite = {k: v for k, v in cosine.items() if v is not None}
        worst = min(finite, key=finite.get) if finite else None
        comparisons.append({
            'reference': configs[0],
            'candidate': name,
            'gradient_cosine': cosine,
            'max_abs_difference': {k: float((grads[0][k] - groups[k]).abs().max())
                                   if cosine[k] is not None else None for k in cosine},
            'min_gradient_cosine': finite.get(worst),
            'min_gradient_cosine_group': worst,
            'cosine_threshold': cosine_threshold,
            'all_groups_pass': all(v is not None and v >= cosine_threshold
                                   for v in cosine.values()),
            'step_time_speedup': (runs[0]['step_time_s'] / report['step_time_s']
                                  if report['step_time_s'] else None),
        })
    return {
        'probe': 'train-step',
        'seed': seed,
        'configs': list(configs),
        'batch': describe_batch(items),
        'model_args': {k: v for k, v in vars(args).items()
                       if k not in ('precision', 'precision_config')},
        'environment': environment(),
        'runs': runs,
        'comparisons': comparisons,
    }


def parse_probe_args(argv):
    if '--' in argv:
        split = argv.index('--')
        argv, train_argv = argv[:split], argv[split + 1:]
    else:
        train_argv = []
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output', required=True, help='JSON report path')
    parser.add_argument('--configs', nargs='+', choices=tuple(LIGHTNING_PRECISION),
                        default=['fp32-reference'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sim_ckpt', help='initialize model weights, not optimizer resume state')
    parser.add_argument('--time', type=float, default=None, help='fixed transport time for capacity probes')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset_index', type=int, default=0,
                        help='first item of the --train_split dataset (default)')
    source.add_argument('--synthetic_residues', type=int, default=None,
                        help='use a random-walk batch with this many residues')
    parser.add_argument('--cosine_threshold', type=float, default=0.99)
    args = parser.parse_args(argv)
    if args.time is not None and not 0 < args.time < 1:
        parser.error('--time must lie in (0, 1)')
    return args, train_argv


def main(argv=None):
    probe_args, train_argv = parse_probe_args(sys.argv[1:] if argv is None else argv)
    from .parsing import parse_train_args
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], *train_argv]
    try:
        args = parse_train_args()
    finally:
        sys.argv = saved_argv
    use_sdpa = not args.manual_attention
    delattr(args, 'manual_attention')
    args.sim_ckpt = probe_args.sim_ckpt
    args.probe_time = probe_args.time
    if args.sim_ckpt:
        saved = torch.load(args.sim_ckpt, map_location='cpu', weights_only=False)
        trained_args = saved['hyper_parameters']['args']
        # Restore architecture and conditioning; retain explicit data, capacity and runtime args.
        keys = ('num_layers', 'embed_dim', 'mha_heads', 'ipa_heads', 'ipa_head_dim',
                'ipa_qk', 'ipa_v', 'prepend_ipa', 'interleave_ipa', 'no_rope',
                'hyena', 'abs_pos_emb', 'abs_time_emb', 'no_offsets', 'no_frames',
                'no_torsion', 'no_design_torsion', 'supervise_no_torsions',
                'supervise_all_torsions', 'no_aa_emb', 'scale_factor', 'time_multiplier',
                'sim_condition', 'design', 'inpainting', 'dynamic_mpnn', 'mpnn',
                'tps_condition', 'path_type', 'prediction', 'alpha_max',
                'discrete_loss_weight', 'cond_interval')
        for key in keys:
            if hasattr(trained_args, key):
                setattr(args, key, getattr(trained_args, key))
    pl.seed_everything(probe_args.seed, verbose=False)
    if probe_args.synthetic_residues is not None:
        items = synthetic_items(args, probe_args.synthetic_residues, args.batch_size,
                                probe_args.seed)
        if args.training_region:
            items = synthetic_region_items(args, items, probe_args.seed)
    else:
        items = dataset_items(args, probe_args.dataset_index)
    report = run_probe(args, items, probe_args.configs, probe_args.seed,
                       use_sdpa=use_sdpa, cosine_threshold=probe_args.cosine_threshold)
    report['batch']['source'] = ('synthetic' if probe_args.synthetic_residues is not None
                                 else f'{args.train_split}[{probe_args.dataset_index}]')
    report['model_args']['use_sdpa'] = use_sdpa
    if args.sim_ckpt:
        with open(args.sim_ckpt, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['checkpoint'] = {'path': args.sim_ckpt, 'sha256': digest,
                                'optimizer_state': 'fresh'}
    os.makedirs(os.path.dirname(os.path.abspath(probe_args.output)), exist_ok=True)
    with open(probe_args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str, allow_nan=False)
    return report


if __name__ == '__main__':
    main()
