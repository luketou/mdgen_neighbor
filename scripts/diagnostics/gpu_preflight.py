"""Check the selected interpreter can execute CUDA, without loading a model."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    report = {'status': 'failed', 'passed': False, 'python': sys.executable,
              'env': {key: os.environ.get(key) for key in
                      ('CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'SLURM_JOB_GPUS')}}
    try:
        import torch
        report['torch'] = torch.__version__
        required = args.require_gpu or bool(os.environ.get('SLURM_JOB_GPUS'))
        if not torch.cuda.is_available() and not required:
            report['status'] = 'skipped'
        else:
            x = torch.ones((2, 2), device='cuda')
            y = x @ x
            torch.cuda.synchronize()
            if not torch.equal(y, torch.full_like(y, 2)):
                raise RuntimeError('CUDA matmul returned an incorrect result')
            report.update(status='passed', passed=True, gpu=torch.cuda.get_device_name(0))
    except Exception as error:
        report['error'] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
