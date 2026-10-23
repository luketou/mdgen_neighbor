"""Zero-truncation controls through the real training-step probe, without acceptance thresholds."""
import copy
import hashlib

import numpy as np
import torch

from .dataset import MDGenDataset, collate
from .train_step_probe import (dataset_items, describe_batch, environment, run_step,
                               synthetic_region_items)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def vector_comparison(reference, candidate):
    """Raw scales for calibration; exact zero direction is undefined, no near-zero cutoff yet."""
    if reference.keys() != candidate.keys():
        raise ValueError('control parameter groups differ')
    report = {}
    for key in reference:
        a, b = reference[key].double(), candidate[key].double()
        if a.shape != b.shape:
            raise ValueError(f'control parameter group shape differs: {key}')
        finite = bool(a.isfinite().all() and b.isfinite().all())
        na, nb, nd = (float(v.norm()) for v in (a, b, b - a))
        report[key] = {
            'numel': a.numel(), 'nonfinite': int((~a.isfinite()).sum() + (~b.isfinite()).sum()),
            'reference_norm': na if finite else None, 'candidate_norm': nb if finite else None,
            'difference_norm': nd if finite else None,
            'reference_rms': na / np.sqrt(a.numel()) if finite else None,
            'candidate_rms': nb / np.sqrt(a.numel()) if finite else None,
            'difference_rms': nd / np.sqrt(a.numel()) if finite else None,
            'relative_difference': nd / na if finite and na else None,
            'norm_ratio': nb / na if finite and na else None,
            'cosine': float((a @ b / (na * nb)).clamp(-1, 1)) if finite and na and nb else None,
            'reference_zero': na == 0 if finite else None,
            'candidate_zero': nb == 0 if finite else None}
    return report


def run_coverage_control(args, full_items, seed, *, synthetic, dataset_index,
                         dataset_rng_state, use_sdpa):
    from .wrapper import NewMDGenWrapper
    if len(full_items) != 1 or not args.local_spatial:
        raise ValueError('coverage control requires batch size 1 and local_spatial')
    if any(getattr(args, flag, False) for flag in ('hyena', 'design', 'abs_pos_emb')):
        raise ValueError('coverage control requires a supported Training region configuration')
    region_args = copy.deepcopy(args)
    region_args.training_region = True
    if synthetic:
        region_items = synthetic_region_items(region_args, full_items, seed)
    else:
        after_sample = np.random.get_state()
        try:
            np.random.set_state(dataset_rng_state)
            region_items = dataset_items(region_args, dataset_index)
        finally:
            np.random.set_state(after_sample)
        full = full_items[0]
        arr = np.load(f"{args.data_dir}/{full['name']}{args.suffix}.npy", mmap_mode='r')
        partition, _, _ = MDGenDataset(args, args.train_split).full_partition(full['name'], arr)
        # Both views retain the stored frame-0 partition even for a later window.
        full_items = [{**full, 'partition': partition}]
    full, region = full_items[0], region_items[0]
    length = len(full['seqres'])
    if region['region']['residues'] != length or region['region']['loss_residues'] != length:
        raise ValueError('coverage control requires every full-protein residue to be a loss residue')
    if full.get('name') != region.get('name') or full.get('frame_start') != region.get('frame_start'):
        raise ValueError('coverage control requires identical replica and frame window')
    wrapper = NewMDGenWrapper(args, use_sdpa=use_sdpa)
    with torch.no_grad():
        full_prep, region_prep = (wrapper.prep_batch(collate([item])) for item in (full, region))
    del wrapper
    full_mask = full_prep['loss_mask'][0] * (
        full_prep['model_kwargs']['x_cond_mask'][0] == 0).unsqueeze(-1)
    region_mask = region_prep['loss_mask'][0] * (
        region_prep['model_kwargs']['x_cond_mask'][0] == 0).unsqueeze(-1)
    if not torch.equal(full_mask, region_mask[:, :length]) or region_mask[:, length:].count_nonzero():
        raise ValueError('coverage control supervision differs')
    if not torch.equal(full_prep['latents'][0], region_prep['latents'][0, :, :length]):
        raise ValueError('coverage control latent inputs differ')
    noise = torch.randn(full_prep['latents'][0].shape,
                        generator=torch.Generator().manual_seed(seed))
    region_noise = torch.zeros_like(region_prep['latents'][0])
    region_noise[:, :length] = noise
    full = {**full, 'probe_noise': noise, 'probe_loss_mask': full_mask}
    region = {**region, 'probe_noise': region_noise, 'probe_loss_mask': region_mask}
    alignment = {
        'full_noise_sha256': tensor_hash(noise),
        'region_noise_sha256': tensor_hash(region_noise[:, :length]),
        'full_mask_sha256': tensor_hash(full_mask),
        'region_mask_sha256': tensor_hash(region_mask[:, :length]),
        'full_denominator': float(full_mask.double().sum()),
        'region_denominator': float(region_mask.double().sum()),
        'residues': length, 'region_width': region_noise.shape[1],
        'frame_start': full.get('frame_start'), 'name': full.get('name')}
    if alignment['full_denominator'] <= 0:
        raise ValueError('coverage control has no unconditioned supervision')
    runs, gradients, updates, initial_state = [], [], [], None
    for config, item in ((args, full), (args, full), (region_args, region)):
        result, grad, initial_state, delta = run_step(config, [item], 'fp32-reference', seed,
            initial_state, use_sdpa=use_sdpa, capture_updates=True)
        runs.append(result)
        gradients.append(grad)
        updates.append(delta)
    comparisons = []
    for i, kind in ((1, 'same_path_repeat'), (2, 'full_coverage')):
        a, b = runs[0]['loss'], runs[i]['loss']
        comparisons.append({'kind': kind,
            'loss': {'reference': a, 'candidate': b,
                     'absolute_difference': abs(b - a) if a is not None and b is not None else None},
            'gradients': vector_comparison(gradients[0], gradients[i]),
            'updates': vector_comparison(updates[0], updates[i])})
    return {'probe': 'full-coverage-control', 'thresholds_applied': False,
            'seed': seed, 'alignment': alignment, 'batch': describe_batch([full]),
            'model_args': vars(args).copy(), 'runs': runs, 'comparisons': comparisons,
            'environment': environment()}
