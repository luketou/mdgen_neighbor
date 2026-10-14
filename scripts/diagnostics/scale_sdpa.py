"""Matched L40S engineering benchmark; execute each case in a fresh process."""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import pandas as pd
import torch
from mdgen.wrapper import NewMDGenWrapper
from mdgen.rigid_utils import Rigid
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get('MDGEN_RESOURCE_ROOT', str(ROOT.parent/'mdgen')))


def move(value):
    if isinstance(value, (torch.Tensor, Rigid)):
        return value.cuda()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name: move(getattr(value, f.name)) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k: move(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move(v) for v in value)
    if isinstance(value, list):
        return [move(v) for v in value]
    return value


def fixture(length):
    """Repeat a real protein on a 3-D lattice, truncate to L; separate chains.

    This is a resource benchmark, not a claim about a real large protein.
    Both modes receive byte-identical coordinates, sequence and conditioning.
    """
    source = RESOURCE_ROOT / 'data/atlas_data/5e3e_A_R1_i40.npy'
    first = np.array(np.load(source, mmap_mode='r')[0], dtype=np.float32)
    sequence = pd.read_csv(RESOURCE_ROOT / 'splits/mini_atlas_all.csv').set_index('name').loc['5e3e_A', 'seqres']
    size = len(sequence)
    copies = (length + size - 1) // size
    side = int(np.ceil(copies ** (1/3)))
    spacing = float(np.ptp(first[:, 1], axis=0).max() + 40)
    shifts = np.array([(i % side, (i//side) % side, i//(side*side)) for i in range(copies)], dtype=np.float32) * spacing
    array = (first[None] + shifts[:, None, None, :]).reshape(-1,14,3)[:length][None]
    aa = torch.tensor([restype_order[c] for c in (sequence*copies)[:length]])
    frames = atom14_to_frames(torch.from_numpy(array))
    atom37 = torch.from_numpy(atom14_to_atom37(array, aa[None])).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, aa[None])
    batch = dict(torsions=torsions[None].expand(1,250,-1,-1,-1).contiguous(),
                 torsion_mask=torsion_mask[:1],
                 trans=frames.get_trans()[None].expand(1,250,-1,-1).contiguous(),
                 rots=frames.get_rots().get_rot_mats()[None].expand(1,250,-1,-1,-1).contiguous(),
                 seqres=aa[None], mask=torch.ones(1,length),
                 chain_id=(torch.arange(length)//size)[None],
                 residue_index=(torch.arange(length)%size)[None])
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        digest.update(key.encode()); digest.update(value.numpy().tobytes())
    return batch, dict(kind='repeated_5e3e_A_on_3D_lattice', source=str(source),
                      input_sha256=digest.hexdigest(), spacing_angstrom=spacing, chains=copies)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['global','local'], required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--operation', choices=['forward','generation'], required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--validation-only', action='store_true', help='Separate output-sanity pass; not a timing benchmark')
    parser.add_argument('--preprocessing', choices=['native','shared_chunked'], default='native')
    parser.add_argument('--ffn-policy', choices=['full','chunked'], default='full')
    parser.add_argument('--ffn-chunk-size', type=int, default=65536)
    parser.add_argument('--ode-save-points', type=int, choices=[2,50],
                        help='Override saved ODE points for the paired storage benchmark')
    args = parser.parse_args()
    if args.ffn_chunk_size <= 0:
        parser.error('--ffn-chunk-size must be positive')
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f'{args.mode}_{args.length}_{args.operation}.json'
    report = dict(mode=args.mode, residues=args.length, frames=250, batch_size=1,
                  operation=args.operation, precision='float32', seed=137, status='running',
                  summaries=False, ffn_policy=args.ffn_policy, sampling_method='dopri5', atol=1e-6, rtol=1e-3,
                  job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
                  model_source=__import__('mdgen.wrapper', fromlist=['']).__file__)
    report['model_source_sha256'] = hashlib.sha256(Path(report['model_source']).read_bytes()).hexdigest()
    report['source_hashes'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('mdgen/wrapper.py', 'mdgen/model/latent_model.py',
                     'mdgen/model/ipa.py', 'mdgen/model/mha.py',
                     'mdgen/full_length/model.py', 'mdgen/full_length/attention.py',
                     'mdgen/full_length/partition.py', 'mdgen/rigid_utils.py')}
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(path)
    def stage(name):
        report['stage'] = name; save()
    save()
    report['validation_only'] = args.validation_only
    report['preprocessing'] = args.preprocessing
    if args.preprocessing == 'shared_chunked':
        # Both repos use exactly the existing local rot_to_quat implementation;
        # this is an explicit benchmark policy, not a production code change.
        source = ROOT/'mdgen/rigid_utils.py'
        report['preprocessing_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location('benchmark_shared_rigid', source)
        shared = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shared)
        import mdgen.rigid_utils as rigid_module
        rigid_module.rot_to_quat = shared.rot_to_quat
    started = time.perf_counter()
    operation_started = None
    try:
        stage('cuda_initialization')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prop = torch.cuda.get_device_properties(0)
        report.update(gpu=prop.name, gpu_uuid=str(prop.uuid), total_vram_bytes=prop.total_memory,
                      torch=torch.__version__, matmul_precision=torch.get_float32_matmul_precision(),
                      tf32=torch.backends.cuda.matmul.allow_tf32)
        stage('checkpoint_loading')
        ckpt_path = RESOURCE_ROOT/'ckpt/atlas.ckpt'
        report['checkpoint_sha256'] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        config = copy.deepcopy(checkpoint['hyper_parameters']['args'])
        config.local_spatial = args.mode == 'local'
        config.temporal_sdpa = True
        config.cell_summaries = False
        config.group_batch_size = 4
        config.num_frames = 250
        config.ffn_chunk_size = args.ffn_chunk_size if args.ffn_policy == 'chunked' else 250 * args.length + 1
        report['ffn_chunk_size'] = config.ffn_chunk_size
        config.ode_atol, config.ode_rtol = 1e-6, 1e-3
        config.sampling_method = 'dopri5'
        torch.manual_seed(137)
        model = NewMDGenWrapper(config).float().cuda().eval()
        # Explicit policy supports the original full-FFN comparison and the
        # inference-only chunking experiment without changing old defaults.
        for layer in model.model.layers:
            if hasattr(layer, 'ffn_chunk_size'):
                layer.ffn_chunk_size = config.ffn_chunk_size
        if args.mode == 'global':
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            from mdgen.full_length.checkpoint import load_compatible_state_dict
            loading = load_compatible_state_dict(model, checkpoint)
            assert not loading['shape_mismatch'] and not loading['unexpected']
            assert all('.cell_summaries.' in k for k in loading['newly_initialized'])
            report['checkpoint_loading'] = loading
        del checkpoint
        stage('fixture_creation_cpu')
        batch, report['fixture'] = fixture(args.length)
        report['input_all_finite'] = all(bool(torch.isfinite(value).all()) for value in batch.values() if value.is_floating_point())
        assert report['input_all_finite'], 'fixture contains nonfinite input'
        with torch.no_grad():
            if args.operation == 'forward':
                # Prepare identical conditioning on CPU: isolated forward must
                # remain measurable even when the real GPU preprocessing OOMs.
                stage('preprocessing_cpu')
                prepared = model.prep_batch(batch)
                if args.mode == 'local':
                    report['partition'] = prepared['model_kwargs']['spatial_partitions'][0].diagnostics()
                stage('input_transfer')
                kwargs = move(prepared['model_kwargs'])
                shape = prepared['latents'].shape
                del prepared, batch
                torch.manual_seed(137)
                inputs = torch.randn(shape, device='cuda', dtype=torch.float32)
                timestep = torch.tensor([.5], device='cuda')
                def forward():
                    return model.model.forward_inference(inputs, timestep, **kwargs)
                report['baseline_allocated_bytes'] = torch.cuda.memory_allocated()
                stage('forward_warmup')
                operation_started = time.perf_counter()
                for _ in range(0 if args.validation_only else 2):
                    forward()
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                report['baseline_allocated_bytes'] = baseline
                stage('forward')
                operation_started = time.perf_counter()
                timings=[]
                repeats = 1 if args.validation_only else 5
                for repeat in range(repeats):
                    torch.cuda.synchronize(); begin=time.perf_counter()
                    output=forward(); torch.cuda.synchronize()
                    timings.append(time.perf_counter()-begin)
                    if repeat == repeats-1:
                        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
                        report['output_all_finite'] = bool(output.isfinite().all())
                        assert report['output_all_finite'], 'Non-finite forward output'
                    del output
                report.update(seconds=timings, median_seconds=statistics.median(timings),
                              baseline_allocated_bytes=baseline)
            else:
                stage('input_transfer')
                batch = move(batch)
                original_prep = model.prep_batch
                def prep(value):
                    stage('preprocessing_gpu')
                    result=original_prep(value)
                    if args.mode == 'local':
                        report['partition']=result['model_kwargs']['spatial_partitions'][0].diagnostics()
                    stage('ode_solver')
                    return result
                model.prep_batch=prep
                original_forward = model.model.forward_inference
                evaluations=[0]
                def counted(*values, **kwargs):
                    # Stage metadata is kept in memory per NFE to avoid filesystem
                    # timing overhead inside the solver; catch writes it on OOM.
                    report['stage']='model_forward'; evaluations[0]+=1
                    report['model_evaluations']=evaluations[0]
                    result=original_forward(*values, **kwargs)
                    report['stage']='ode_solver'
                    if evaluations[0] % 5 == 0:
                        print(f'PROGRESS {args.mode} L={args.length} NFE={evaluations[0]} elapsed={time.perf_counter()-begin:.1f}s', flush=True)
                    return result
                model.model.forward_inference=counted
                original_sampler=model.transport_sampler.sample_ode
                def sampler(*values, **kwargs):
                    if args.ode_save_points is not None:
                        kwargs['num_steps'] = args.ode_save_points
                    report['ode_saved_points'] = kwargs.get('num_steps', 50)
                    report['actual_solver'] = dict(method=kwargs.get('sampling_method', 'dopri5'),
                                                   atol=kwargs.get('atol', 1e-6), rtol=kwargs.get('rtol', 1e-3))
                    sample_fn=original_sampler(*values, **kwargs)
                    def sample(*values, **kwargs):
                        result=sample_fn(*values, **kwargs)
                        stage('structure_reconstruction')
                        return result
                    return sample
                model.transport_sampler.sample_ode=sampler
                baseline=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                report['baseline_allocated_bytes'] = baseline
                report['model_evaluations'] = 0
                torch.manual_seed(137); torch.cuda.manual_seed_all(137)
                torch.cuda.synchronize(); begin=time.perf_counter()
                operation_started = begin
                output=model.inference(batch); torch.cuda.synchronize()
                report.update(seconds=time.perf_counter()-begin, model_evaluations=evaluations[0],
                              baseline_allocated_bytes=baseline, output_shape=list(output[0].shape))
                report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
                report['output_all_finite'] = all(bool(v.isfinite().all()) for v in output if isinstance(v,torch.Tensor))
                assert report['output_all_finite'], 'Non-finite generated structure'
                # After timing and memory capture, fingerprint final structures
                # to check exact equality across the storage-policy pair.
                digest = hashlib.sha256()
                for value in output:
                    if isinstance(value, torch.Tensor):
                        array = value.detach().cpu().contiguous().numpy()
                        digest.update(str((array.shape, array.dtype)).encode())
                        digest.update(array.tobytes())
                report['output_sha256'] = digest.hexdigest()
                del output
            report['status']='success'
        report.setdefault('peak_allocated_bytes',torch.cuda.max_memory_allocated())
        report.setdefault('peak_reserved_bytes',torch.cuda.max_memory_reserved())
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='oom', error=str(error), traceback=traceback.format_exc(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save()
        raise
    finally:
        if operation_started is not None:
            report['operation_elapsed_seconds'] = time.perf_counter()-operation_started
        report['case_wall_seconds']=time.perf_counter()-started
        save()
        print(json.dumps({k:v for k,v in report.items() if k not in ['partition','checkpoint_loading','traceback']}), flush=True)


if __name__ == '__main__':
    main()
