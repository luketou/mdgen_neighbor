"""Issue #43 Ticket 1: acquire, audit and prepare the fixed four-protein dataset."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mdtraj as md
import numpy as np
import torch

from install_atlas_mirror import install, sha256, verify
from mdgen import residue_constants as rc
from mdgen.dataset import MDGenDataset
from mdgen.full_length.partition import PartitionParams
from mdgen.full_length.partition_store import load_partition
from mdgen.parsing import LOCAL_DEFAULTS
from mdgen.topology import chain_numbering


def read_checkpoint(path, report_dir):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint['hyper_parameters']
    saved = saved.get('args', saved)
    saved = vars(saved) if hasattr(saved, '__dict__') else saved
    keys = ('train_split', 'val_split', 'suffix', 'num_frames', 'frame_interval', 'atlas', 'crop')
    metadata = {key: saved.get(key) for key in keys}
    metadata.update(path=str(path), sha256=sha256(path))
    # Download record written next to the checkpoint (Hugging Face API response, pinned revision).
    record = json.loads(path.with_name('atlas.source.json').read_text())
    if record['file']['lfs']['sha256'] != metadata['sha256']:
        raise ValueError('Checkpoint differs from its recorded download')
    metadata['download_record'] = record
    # Physical spacing also needs the original source README; suffix alone is insufficient.
    original_readme = Path('/share/home/r15543056/MDgen/mdgen/data/atlas_sims/6irx_A/README.txt')
    text = original_readme.read_text()
    metadata['original_source_readme_sha256'] = sha256(original_readme)
    metadata['original_source_readme'] = str(original_readme)
    recipe = json.loads(Path('scripts/atlas_validation_split_audit.json').read_text())['upstream_recipe']
    if 'every 10 ps' not in text or recipe['atlas_preprocess_stride'] != 40:
        raise ValueError('Cannot establish the upstream preprocessing recipe')
    if saved.get('frame_interval') not in (None, 1) or saved.get('num_frames') != 250:
        raise ValueError('Checkpoint has an unreviewed temporal configuration')
    metadata['selected_input_interval_ps'] = 400
    metadata['checkpoint_training_interval_ps'] = None
    metadata['derivation'] = 'validation input follows pinned upstream ATLAS recipe: 10 ps x stride 40; actual checkpoint training interval is not embedded'
    metadata['upstream_recipe'] = recipe
    metadata['training_membership'] = 'upstream split comparison is evidence; actual checkpoint training membership remains unverified'
    params = {}
    for key in ('spatial_cell_width', 'spatial_contact_radius', 'core_capacity', 'ghost_capacity'):
        params[key] = saved.get(key, LOCAL_DEFAULTS[key])
    metadata['partition_settings'] = params
    metadata['partition_settings_origin'] = {
        k: 'checkpoint' if k in saved else 'current explicit local default' for k in params}
    (report_dir / 'checkpoint_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata, params


def topology_audit(pdb, sequence):
    structure = md.load_pdb(str(pdb))
    top = structure.topology
    if ''.join(top.to_fasta()) != sequence:
        raise ValueError(f'{pdb}: sequence differs from the pinned manifest')
    chain_id, residue_index = chain_numbering(top)
    residues = list(top.residues)
    for residue in residues:
        allowed = set(rc.restype_name_to_atom14_names.get(residue.name, ())) - {''}
        actual = {a.name for a in residue.atoms if a.element.symbol != 'H'}
        if not allowed or allowed - actual or actual - allowed - {'OXT'}:
            raise ValueError(f'{pdb}: unsupported or incomplete heavy atoms at {residue}')
    breaks = []
    for previous, following in zip(residues, residues[1:]):
        if previous.chain.index != following.chain.index:
            continue
        c = next(a.index for a in previous.atoms if a.name == 'C')
        n = next(a.index for a in following.atoms if a.name == 'N')
        distance = float(np.linalg.norm(structure.xyz[0, c] - structure.xyz[0, n]) * 10)
        if distance > 2.0:
            breaks.append({'previous': previous.index, 'next': following.index, 'distance_A': distance})
            if following.resSeq == previous.resSeq + 1:
                raise ValueError(f'{pdb}: geometric chain break absent from topology numbering')
    return {'residues': len(sequence), 'atoms': top.n_atoms, 'chains': top.n_chains,
            'numbering_gaps': int(np.sum((np.diff(residue_index) != 1) & (np.diff(chain_id) == 0))),
            'geometric_breaks': breaks, 'standard_heavy_atoms_complete': True}


def overlap_audit(proteins, train_split):
    from Bio.Align import PairwiseAligner
    with open(train_split) as stream:
        train = list(csv.DictReader(stream))
    aligner = PairwiseAligner(mode='local', match_score=2, mismatch_score=-1,
                              open_gap_score=-10, extend_gap_score=-0.5)
    result = {}
    for protein in proteins:
        query = protein['expected_sequence']
        ranked = sorted(((aligner.score(query, r['seqres']), r) for r in train),
                        key=lambda pair: pair[0], reverse=True)
        closest = []
        for score, row in ranked[:3]:
            alignment = aligner.align(query, row['seqres'])[0]
            blocks = list(zip(*alignment.aligned))
            count = sum(a[1] - a[0] for a, _ in blocks)
            matches = sum(sum(x == y for x, y in zip(query[a[0]:a[1]], row['seqres'][b[0]:b[1]]))
                          for a, b in blocks)
            closest.append({'name': row['name'], 'local_alignment_score': float(score),
                            'aligned_identity': matches / count if count else None,
                            'aligned_query_fraction': count / len(query),
                            'aligned_reference_fraction': count / len(row['seqres'])})
        result[protein['name']] = {
            'exact_training_names': [r['name'] for r in train if r['name'] == protein['name']],
            'exact_training_sequences': [r['name'] for r in train if r['seqres'] == query],
            'highest_local_alignment_scores': closest,
            'method': 'all training sequences scored; top 3 inspected using local alignment 2/-1/-10/-0.5',
            'independence': 'not certified: local alignment is a screen, not an exhaustive domain/family audit',
            'unassigned_reason': 'unknown; no verified exclusion reason' if protein['name'] == '6sup_A' else 'local test split',
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source_dir', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train_split', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Execute acquisition and data processing in a Slurm CPU job')
    args.out_dir.mkdir()  # Never reuse/overwrite a prior prepared dataset.
    report = {'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
              'source_dir': str(args.source_dir), 'out_dir': str(args.out_dir), 'proteins': []}
    report_path = args.out_dir / 'summary.json'
    try:
        checkpoint, settings = read_checkpoint(args.checkpoint, args.out_dir)
        manifest = json.loads(args.manifest.read_text())
        report['manifest_sha256'] = sha256(args.manifest)
        report['checkpoint'] = checkpoint
        report['overlap'] = overlap_audit(manifest['proteins'], args.train_split)
        split = args.out_dir / 'validation.csv'
        with split.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['name', 'seqres'])
            writer.writerows((p['name'], p['expected_sequence']) for p in manifest['proteins'])
        arrays = args.out_dir / 'arrays'
        partitions = args.out_dir / 'partitions'
        arrays.mkdir()
        # Match the upstream recipe once; do not claim absent checkpoint provenance.
        stride = checkpoint['selected_input_interval_ps'] // 100
        for protein in manifest['proteins']:
            name = protein['name']
            folder = args.source_dir / name
            if not folder.exists():
                install(protein, manifest, args.source_dir)
            source_files = [verify(folder / entry['name'], entry) for entry in protein['files']]
            topology = topology_audit(folder / f'{name}.pdb', protein['expected_sequence'])
            report['proteins'].append({'name': name, 'topology': topology, 'source_files': source_files})
        subprocess.run([
            sys.executable, 'scripts/prep_sims.py', '--split', str(split),
            '--atlas_dir', str(args.source_dir), '--outdir', str(arrays),
            '--atlas', '--atlas_layout', 'analysis', '--expected_interval_ps', '100',
            '--stride', str(stride), '--dtype', 'float32', '--suffix', '_i400ps',
        ], check=True)
        command = [sys.executable, '-m', 'mdgen.full_length.partition_store', '--split', str(split),
                   '--data_dir', str(arrays), '--out_dir', str(partitions), '--atlas',
                   '--suffix', '_i400ps', '--drift_stride', '250']
        for key, value in settings.items():
            command.extend([f'--{key}', str(value)])
        subprocess.run(command, check=True)
        params = PartitionParams.from_args(SimpleNamespace(**settings))
        for protein, entry in zip(manifest['proteins'], report['proteins']):
            name, length = protein['name'], len(protein['expected_sequence'])
            entry['outputs'] = []
            dataset_args = SimpleNamespace(
                data_dir=str(arrays), suffix='_i400ps', frame_interval=None, num_frames=4,
                atlas=True, crop=length, training_region=False, overfit=False,
                overfit_peptide=None, overfit_frame=True, copy_frames=False, no_frames=False)
            for replica in (1, 2, 3):
                stem = f'{name}_R{replica}_i400ps'
                arr = np.load(arrays / f'{stem}.npy', mmap_mode='r')
                if arr.shape != (251, length, 14, 3) or arr.dtype != np.float32 or not np.isfinite(arr).all():
                    raise ValueError(f'{stem}: invalid prepared shape/dtype/coordinates')
                with np.load(arrays / f'{stem}_chain.npz') as chain:
                    partition = load_partition(partitions / f'{stem}_partition.npz', params,
                                               chain_id=chain['chain_id'], residue_index=chain['residue_index'],
                                               ca=arr[0, :, 1])
                index = next(i for i, p in enumerate(manifest['proteins']) if p['name'] == name)
                item = MDGenDataset(dataset_args, split, replicates=[replica])[index]
                if not all(torch.isfinite(v).all() for v in item.values() if torch.is_tensor(v)):
                    raise ValueError(f'{stem}: nonfinite official Dataset item')
                # Units (nm -> A) and frame mapping at source frame 0; repeats prep_sims' alignment,
                # so it does not check the alignment itself.
                source = md.load(str(args.source_dir / name / f'{name}_R{replica}.xtc'),
                                 top=str(args.source_dir / name / f'{name}.pdb'), stride=4)
                source.atom_slice(source.topology.select('not element H'), inplace=True)
                source.superpose(source, frame=0)
                error = 0.0
                for residue in (0, length // 2, length - 1):
                    atom = next(a.index for a in source.topology.residue(residue).atoms if a.name == 'CA')
                    error = max(error, float(np.max(np.abs(arr[0, residue, 1] - source.xyz[0, atom] * 10))))
                if error > 0.01:
                    raise ValueError(f'{stem}: coordinate units/alignment round-trip mismatch: {error}')
                paths = [arrays / f'{stem}.npy', arrays / f'{stem}_chain.npz',
                         partitions / f'{stem}_partition.npz']
                entry['outputs'].append({'stem': stem, 'shape': list(arr.shape), 'dtype': str(arr.dtype),
                                         'effective_interval_ps': 400, 'groups': len(partition.groups),
                                         'dataset_frames_checked': 4, 'coordinate_error_A': error,
                                         'files': [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                                                   for p in paths]})
                del source, item, arr
            print(f'ACCEPTED {name}: all 3 replicas, arrays, sidecars, partitions and Dataset input', flush=True)
        report['status'] = 'complete'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'COMPLETE: {report_path}', flush=True)


if __name__ == '__main__':
    main()
