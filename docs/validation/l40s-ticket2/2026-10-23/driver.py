"""Finite L40S validation through the published training-step CLI; no model edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--require-gpu', action='store_true')
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
out = args.output.parent
out.mkdir(parents=True, exist_ok=True)
summary = {'passed': False, 'status': 'failed', 'scope': 'Ticket 2 small synthetic batch',
           'cases': {}, 'job_id': os.environ.get('SLURM_JOB_ID')}
env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '137', 'PYTHONPATH': str(Path.cwd()),
       'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
       'WANDB_MODE': 'disabled', 'MODEL_DIR': str(out / 'model-log'),
       'TRITON_CACHE_DIR': str(out / 'triton')}
tiny = ['--train_split', 'unused', '--val_split', 'unused', '--data_dir', 'unused',
        '--num_layers', '2', '--embed_dim', '32', '--mha_heads', '4', '--ipa_heads', '2',
        '--ipa_head_dim', '4', '--ipa_qk', '2', '--ipa_v', '2', '--num_frames', '4',
        '--prepend_ipa', '--interleave_ipa', '--sim_condition', '--batch_size', '2',
        '--deterministic', '--model_dir', str(out / 'models')]
local = ['--local_spatial', '--spatial_cell_width', '4', '--spatial_contact_radius', '8',
         '--core_capacity', '4', '--ghost_capacity', '20', '--group_batch_size', '2']

def execute(command, log, timeout=100):
    with (out / log).open('w') as handle:
        result = subprocess.run(command, env=env, cwd=out if log == 'pytest.log' else None,
                                stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{log}: command failed with exit {result.returncode}')

try:
    (out / 'triton').mkdir(exist_ok=True)
    execute([sys.executable, '.scratch/l40s-validation/minimal_mha.py', '--require-gpu',
             '--output', str(out / 'minimal-mha.json')], 'minimal-mha.log')
    summary['minimal_mha'] = json.loads((out / 'minimal-mha.json').read_text())
    execute([sys.executable, 'scripts/diagnostics/gpu_preflight.py', '--require-gpu',
             '--output', str(out / 'gpu.json')], 'gpu.log')
    summary['gpu'] = json.loads((out / 'gpu.json').read_text())
    for mode in ['global', 'local']:
        reports = []
        for repetition in [1, 2]:
            name = f'{mode}-{repetition}'
            command = [sys.executable, '-m', 'mdgen.train_step_probe', '--output', str(out / (name + '.json')),
                       '--configs', 'fp32-reference', 'fp32-reference', 'bf16-train',
                       '--synthetic_residues', '12', '--seed', '137', '--', *tiny,
                       *(local if mode == 'local' else [])]
            summary.setdefault('commands', []).append(command)
            execute(command, name + '.log')
            report = json.loads((out / (name + '.json')).read_text())
            reports.append(report)
            summary['cases'][name] = report
            assert report['environment']['cuda_available']
            assert report['environment']['device_name'] == 'NVIDIA L40S'
            for run in report['runs']:
                assert run['loss_finite'] and run['loss'] > 0
                assert run['gradient_nonfinite'] == 0
                assert run['parameters_nonfinite_after_step'] == 0
                assert run['step_time_s'] > 0
                assert run['peak_memory_allocated_bytes'] > 0
                assert run['peak_memory_reserved_bytes'] >= run['peak_memory_allocated_bytes']
                assert run['precision_state']['parameter_dtypes'] == ['torch.float32']
                assert all('.cell_summaries.' in n for n in run['missing_gradients'])
            repeated = report['comparisons'][0]
            assert repeated['all_groups_pass']
            assert all(abs(v - 1) < 1e-12 for v in repeated['gradient_cosine'].values())
            assert all(v == 0 for v in repeated['max_abs_difference'].values())
            assert report['runs'][0]['precision_state']['float32_matmul_precision'] == 'highest'
            assert report['runs'][2]['precision_state']['autocast_dtype'] == 'torch.bfloat16'
        # Separate CLI processes must reproduce the numerical evidence exactly.
        for left, right in zip(reports[0]['runs'], reports[1]['runs']):
            for field in ['loss', 'gradients', 'gradient_nonfinite', 'parameters_nonfinite_after_step']:
                assert left[field] == right[field], f'{mode} cross-process mismatch: {field}'
        summary.setdefault('cross_process_reproducible', {})[mode] = True
    execute([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
             '--basetemp', str(out / 'pytest-temp'),
             str(Path.cwd() / 'tests/test_train_step_probe.py') + '::test_nonfinite_batch_is_counted',
             str(Path.cwd() / 'tests/test_train_step_probe.py') + '::test_gradient_cosine_cases',
             str(Path.cwd() / 'tests/test_train_step_probe.py') + '::test_cuda_reports_peak_memory',
             str(Path.cwd() / 'tests/test_train_step_probe.py') + '::test_cuda_bf16_probe_keeps_fp32_weights_and_finite_gradients'], 'pytest.log')
    summary['pytest'] = (out / 'pytest.log').read_text()
    nonfinite_code = """
import json,sys,torch
from mdgen.parsing import parse_train_args
from mdgen.train_step_probe import synthetic_items,run_probe
train=json.loads(sys.argv[1]);sys.argv=['train.py',*train];a=parse_train_args();delattr(a,'manual_attention')
reports={}
for kind in ['nan','inf']:
 items=synthetic_items(a,12,2,137);items[0]['torsions'][0,3,0,0]=float(kind)
 r=run_probe(a,items,['fp32-reference','fp32-reference'],137)
 assert r['runs'][0]['loss_finite'] is False
 assert r['runs'][0]['gradient_nonfinite']>0
 assert r['comparisons'][0]['all_groups_pass'] is False
 reports[kind]=r
open(OUTPUT,'w').write(json.dumps(reports,allow_nan=False))
""".replace('OUTPUT', repr(str(out / 'nonfinite.json')))
    execute([sys.executable, '-c', nonfinite_code, json.dumps(tiny)], 'nonfinite.log')
    summary['nonfinite_injections'] = json.loads((out / 'nonfinite.json').read_text())
    summary['passed'] = True
    summary['status'] = 'passed'
except Exception as error:
    summary['error'] = str(error)
    summary['traceback'] = traceback.format_exc()
finally:
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps({'passed': summary['passed'], 'status': summary['status'], 'error': summary.get('error')}))
raise SystemExit(0 if summary['passed'] else 1)
