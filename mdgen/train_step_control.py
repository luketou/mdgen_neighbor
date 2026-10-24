"""同一 supervision 的成對訓練與完整涵蓋控制，沿用官方 training-step probe。"""
import copy

import numpy as np
import torch

from .dataset import MDGenDataset, collate
from .train_step_probe import (dataset_items, describe_batch, environment, run_step,
                               synthetic_region_items, tensor_hash)


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


def paired_selection(partition, ca, budget, halo, seed_group):
    """共同可行時以兩圈選 loss groups；否則只保留一圈自己的結果。"""
    from .full_length.group_region import sample_region_groups, with_halo
    try:
        selected = sample_region_groups(partition, ca, region_residues=budget,
                                       halo_rings=2, seed_group=seed_group)
        return with_halo(partition, selected.loss_groups, halo), True
    except ValueError as error:
        if 'cannot hold seed group' not in str(error) or halo != 1:
            raise
        return sample_region_groups(partition, ca, region_residues=budget,
                                    halo_rings=1, seed_group=seed_group), False


def check_denominators(full_mask, region_mask):
    """零分母不是失敗；不相等或非有限分母拒絕比較。"""
    a, b = (float(mask.double().sum()) for mask in (full_mask, region_mask))
    if not np.isfinite(a) or not np.isfinite(b) or a != b or a < 0:
        raise ValueError(f'paired supervision denominators differ or invalid: {a}, {b}')
    return {'full_denominator': a, 'region_denominator': b,
            'status': 'no_supervision' if a == 0 else 'ready'}


def prepare_paired_items(args, full, seed, *, synthetic, use_sdpa):
    """從同一完整輸入選取 region，保留幾何、noise 與有效監督分量。"""
    from .wrapper import NewMDGenWrapper
    from .full_length.partition import PartitionParams, build_spatial_partition
    from .full_length.group_region import region_subpartition
    from .full_length.region_batch import assemble_region
    if synthetic:
        length = len(full['seqres'])
        chain = {'chain_id': torch.zeros(length, dtype=torch.long),
                 'residue_index': torch.arange(length)}
        ca = full['trans'][0]
        partition = build_spatial_partition(ca, mask=torch.as_tensor(full['mask']).bool(),
            **chain, **PartitionParams.from_args(args).partition_kwargs())
        full = {**full, **chain}
    else:
        arr = np.load(f"{args.data_dir}/{full['name']}{args.suffix}.npy", mmap_mode='r')
        partition, ca, _ = MDGenDataset(args, args.train_split).full_partition(full['name'], arr)
    selection, common = paired_selection(partition, ca, args.region_residues,
                                        args.halo_rings, args.region_seed_group)
    region_partition = region_subpartition(partition, selection)
    full = {**full, 'partition': partition}
    region = assemble_region(full, selection, region_partition, args.region_residues)
    wrapper = NewMDGenWrapper(args, use_sdpa=use_sdpa)
    with torch.no_grad():
        fp, rp = (wrapper.prep_batch(collate([item])) for item in (full, region))
    del wrapper
    indices = selection.residues
    n = len(indices)
    full_mask = fp['loss_mask'][0] * (fp['model_kwargs']['x_cond_mask'][0] == 0).unsqueeze(-1)
    loss_residues = torch.zeros(len(full['seqres']), dtype=torch.bool)
    loss_residues[indices] = region_partition.loss_core_mask
    full_mask = full_mask * loss_residues[None, :, None]
    region_mask = rp['loss_mask'][0] * (rp['model_kwargs']['x_cond_mask'][0] == 0).unsqueeze(-1)
    if not torch.equal(full_mask[:, indices], region_mask[:, :n]) or region_mask[:, n:].count_nonzero():
        raise ValueError('paired effective masks differ')
    denominators = check_denominators(full_mask, region_mask)
    if denominators['status'] != 'no_supervision' and not torch.equal(
            fp['latents'][0, :, indices], rp['latents'][0, :, :n]):
        raise ValueError('paired latent inputs differ')
    noise = torch.randn(fp['latents'][0].shape, generator=torch.Generator().manual_seed(seed))
    region_noise = torch.zeros_like(rp['latents'][0])
    region_noise[:, :n] = noise[:, indices]
    alignment = {**denominators,
        'full_noise_sha256': tensor_hash(noise[:, indices]),
        'region_noise_sha256': tensor_hash(region_noise[:, :n]),
        'full_mask_sha256': tensor_hash(full_mask[:, indices]),
        'region_mask_sha256': tensor_hash(region_mask[:, :n]),
        'loss_residue_ids': loss_residues.nonzero().flatten().tolist(),
        'loss_group_ids': list(selection.loss_groups),
        'region_residue_ids': indices.tolist(), 'full_length': len(full['seqres']),
        'paired_common_subset': common, 't': args.probe_time,
        'full_coverage_or_truncated': 'full_coverage' if n == len(full['seqres']) else 'truncated'}
    return ({**full, 'probe_noise': noise, 'probe_loss_mask': full_mask},
            {**region, 'probe_noise': region_noise, 'probe_loss_mask': region_mask}, alignment)


def compare_paired_runs(a, b, ga, gb, ua, ub, *, control):
    """選項 B 只判定控制；截斷差異只描述，非有限值仍失敗。"""
    gradients, updates = vector_comparison(ga, gb), vector_comparison(ua, ub)
    finite = all(run['loss_finite'] and not run['gradient_nonfinite']
                 and not run['parameters_nonfinite_after_step']
                 and not any(run.get('nonfinite_counts', {}).values()) for run in (a, b))
    missing = [set(run['missing_gradients']) for run in (a, b)]
    unexpected_missing = sorted({n for names in missing for n in names if '.cell_summaries.' not in n})
    finite = finite and not unexpected_missing and missing[0] == missing[1]
    finite = finite and all(not g['nonfinite'] for groups in (gradients, updates) for g in groups.values())
    aligned = (a['initial_state_sha256'] == b['initial_state_sha256']
               and a['optimizer_initial_state'] == b['optimizer_initial_state']
               and a['optimizer'] == b['optimizer']
               and a.get('clipping_rule') == b.get('clipping_rule'))
    delta = b['loss'] - a['loss'] if a['loss'] is not None and b['loss'] is not None else None
    passes = finite and aligned and delta is not None and abs(delta) <= 2e-6
    for groups, cutoff, relative in ((gradients, 1e-7, 5e-4), (updates, 2e-6, 2e-2)):
        for group in groups.values():
            near = bool(group['reference_rms'] is not None and group['reference_rms'] <= cutoff)
            group['near_zero_reference'] = near
            group['one_sided_zero'] = group['reference_zero'] != group['candidate_zero']
            if near:
                for key in ('relative_difference', 'norm_ratio', 'cosine'):
                    group[key] = None
            passed = (not group['nonfinite'] and group['difference_rms'] <= cutoff
                      and (near or group['relative_difference'] <= relative))
            if control:
                group['control_passed'] = bool(passed)
            passes = passes and passed
    return {'status': ('failed' if not finite or not aligned else
                       ('passed' if passes else 'failed') if control else 'descriptive'),
            'control_passed': bool(passes) if control else None,
            'loss': {'reference': a['loss'], 'candidate': b['loss'],
                     'signed_difference': delta, 'absolute_difference': abs(delta) if delta is not None else None,
                     'relative_difference': abs(delta / a['loss']) if delta is not None and a['loss'] else None},
            'unexpected_missing_gradients': unexpected_missing,
            'one_sided_missing_gradients': sorted(missing[0] ^ missing[1]),
            'inactive_gradients': sorted(missing[0] & missing[1]),
            'gradients': gradients, 'updates': updates}


def run_paired_region(args, items, seed, *, synthetic, warmup=1, repeats=3,
                      control_type='paired', artifact_dir=None, use_sdpa=True, raw_tensors='all'):
    """沿用 Lightning、Adam、clipping；每次皆由相同權重與空 optimizer 起步。"""
    from pathlib import Path
    if raw_tensors not in ('all', 'none'):
        raise ValueError('raw_tensors must be all or none')
    if artifact_dir is not None:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    if (len(items) != 1 or not args.local_spatial or args.training_region or args.dropout != 0
            or args.probe_time is None or args.region_seed_group is None or warmup < 0 or repeats < 1):
        raise ValueError('paired probe requires full local batch 1, dropout 0, fixed t and valid repeats')
    if args.prediction != 'velocity' or any(getattr(args, flag, False) for flag in ('hyena', 'design', 'abs_pos_emb', 'no_frames', 'ema')):
        raise ValueError('unsupported paired configuration')
    from .precision import configure_precision
    configure_precision('fp32-reference')
    full, region, alignment = prepare_paired_items(args, items[0], seed,
                                                 synthetic=synthetic, use_sdpa=use_sdpa)
    report = {'probe': 'paired-training-region', 'schema_version': 1, 'contract_version': '1.1',
        'seed': seed, 'alignment': alignment, 'warmup': warmup, 'measured_repeats': repeats,
        'raw_tensors': raw_tensors, 'control_type': control_type, 'batch': describe_batch(items), 'model_args': vars(args).copy(),
        'environment': environment(), 'measurements': []}
    if alignment['status'] == 'no_supervision':
        return {**report, 'status': 'no_supervision'}
    region_args = copy.deepcopy(args)
    region_args.training_region = True
    initial = None
    control = control_type == 'same_path_repeat' or alignment['full_coverage_or_truncated'] == 'full_coverage'
    for iteration in range(warmup + repeats):
        outputs = []
        for side, config, item in (('full', args, full),
                ('candidate', args if control_type == 'same_path_repeat' else region_args,
                 full if control_type == 'same_path_repeat' else region)):
            path = (Path(artifact_dir) / f'{iteration}-{side}.pt') if artifact_dir is not None and raw_tensors == 'all' else None
            try:
                result, grad, initial, update = run_step(config, [item], 'fp32-reference', seed,
                    initial, use_sdpa=use_sdpa, capture_updates=True, artifact_path=path)
            except Exception as error:
                error.probe_evidence = {**getattr(error, 'probe_evidence', {}),
                                        'side': side, 'iteration': iteration, 'alignment': alignment}
                raise
            if result['training_time'] != args.probe_time:
                raise ValueError('actual transport time differs from paired case')
            if result.get('loss_components') is not None:
                expected = alignment['full_denominator' if side == 'full' else 'region_denominator']
                if result['loss_components']['denominator'] != expected:
                    raise ValueError('actual loss denominator differs from paired supervision')
            if artifact_dir is not None:
                import json
                # candidate 失敗時，已完成 full 側的資源與數值證據仍保留。
                with (Path(artifact_dir) / f'{iteration}-{side}.json').open('x') as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
            outputs.append((result, grad, update))
        a, b = outputs
        comparison = compare_paired_runs(a[0], b[0], a[1], b[1], a[2], b[2], control=control)
        report['measurements'].append({'iteration': iteration, 'warmup': iteration < warmup,
            'full': a[0], 'candidate': b[0], 'comparison': comparison})
    statuses = [m['comparison']['status'] for m in report['measurements']]
    report['status'] = 'failed' if 'failed' in statuses else 'passed' if control else 'descriptive'
    return report
