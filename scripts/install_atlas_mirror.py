"""Install pinned ATLAS mirror files and validate all trajectory frames on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, entry):
    size = path.stat().st_size
    if size != entry['bytes']:
        raise ValueError(f"Size mismatch: {path.name}: {size} != {entry['bytes']}")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            blob.update(chunk)
    if entry.get('sha256') and digest.hexdigest() != entry['sha256']:
        raise ValueError(f"Mirror SHA256 mismatch: {path.name}")
    if entry.get('git_blob_sha1') and blob.hexdigest() != entry['git_blob_sha1']:
        raise ValueError(f"Mirror Git blob mismatch: {path.name}")
    return {'path': path.name, 'bytes': size, 'sha256': digest.hexdigest()}


def install(protein, manifest, destination):
    import mdtraj as md
    import numpy as np

    name = protein['name']
    final = destination / name
    if final.exists():
        raise RuntimeError(f'Refusing to overwrite existing directory: {final}')
    stage = destination / f".{name}-mirror-{os.environ['SLURM_JOB_ID']}.partial"
    stage.mkdir()
    records = []
    for entry in protein['files']:
        filename = entry['name']
        if Path(filename).name != filename or filename in ('.', '..'):
            raise ValueError(f'Unsafe manifest filename: {filename}')
        path = stage / filename
        print(f"Downloading {filename}: {entry['bytes']} bytes", flush=True)
        subprocess.run([
            'curl', '--fail', '--location', '--silent', '--show-error',
            '--connect-timeout', '15', '--max-time', '900', '--retry', '2',
            '--retry-delay', '5', '--speed-limit', '1024', '--speed-time', '60',
            '--output', str(path), entry['url'],
        ], check=True)
        record = verify(path, entry)
        record['source_url'] = entry['url']
        records.append(record)

    pdb = stage / f'{name}.pdb'
    topology = md.load_pdb(str(pdb)).topology
    sequence = ''.join(topology.to_fasta())
    if sequence != protein['expected_sequence']:
        raise ValueError(f'{name}: PDB sequence does not match the manifest')
    trajectories = []
    for replica in (1, 2, 3):
        path = stage / f'{name}_R{replica}.xtc'
        frames = 0
        first = previous = None
        for chunk in md.iterload(str(path), top=str(pdb), chunk=25):
            if chunk.n_atoms != topology.n_atoms or not np.isfinite(chunk.xyz).all():
                raise ValueError(f'{path.name}: atom count mismatch or nonfinite coordinates')
            times = chunk.time
            if first is None:
                first = float(times[0])
            differences = np.diff(times)
            if previous is not None:
                differences = np.append(differences, times[0] - previous)
            if not np.allclose(differences, 100.0, rtol=0, atol=0.01):
                raise ValueError(f'{path.name}: frame spacing is not 100 ps')
            previous = float(times[-1])
            frames += chunk.n_frames
        if frames < 1000 or previous - first < 99000:
            raise ValueError(f'{path.name}: incomplete 100 ns reference trajectory')
        trajectories.append({'replica': replica, 'frames': frames, 'atoms': topology.n_atoms,
                             'start_ps': first, 'end_ps': previous, 'interval_ps': 100.0,
                             'coordinates_finite': True, 'full_stream_decode': True})
        print(f'Validated {path.name}: {frames} frames, {first}..{previous} ps', flush=True)

    record = {'name': name, 'status': 'complete', 'destination': str(final),
              'repository': manifest['repository'], 'revision': manifest['revision'],
              'source_type': manifest['source_type'], 'official_origin': manifest['official_origin'],
              'sampling_note': '100 ps ATLAS analysis mirror; not the dense protein ZIP',
              'sequence_matches_manifest': True, 'residues': len(sequence),
              'training_independence': 'not yet verified', 'files': records,
              'trajectories': trajectories}
    (stage / 'installation.json').write_text(json.dumps(record, indent=2) + '\n')
    os.rename(stage, final)
    print(f'Installed and validated {name}: {final}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run installation inside a Slurm CPU job')
    import mdtraj
    print(f'mdtraj={mdtraj.__version__}', flush=True)
    manifest = json.loads(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    required = sum(f['bytes'] for p in manifest['proteins'] for f in p['files'])
    if shutil.disk_usage(args.destination).free < required * 2:
        raise RuntimeError('Insufficient space for mirror installation')
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'status': 'running',
              'started_unix': time.time(), 'proteins': [], 'expected_payload_bytes': required,
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    report_path = args.run_dir / 'results' / 'installation_summary.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    for protein in manifest['proteins']:
        try:
            item = install(protein, manifest, args.destination)
        except Exception as error:
            item = {'name': protein['name'], 'status': 'failed', 'error': str(error)}
            print(f"FAILED {protein['name']}: {error}", flush=True)
        report['proteins'].append(item)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'complete' if all(p['status'] == 'complete' for p in report['proteins']) else 'failed'
    report['finished_unix'] = time.time()
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Final status: {report['status']}; summary={report_path}", flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
