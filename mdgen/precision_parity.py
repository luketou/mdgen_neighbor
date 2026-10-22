"""Precision parity for coordinates in Å; atom slots start with N, CA, C."""
import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_THRESHOLDS = dict(
    bad_cn_ratio_difference=0.01, ca_mean_difference=0.02,
    ca_std_difference=0.02, md_rmsf_pearson_difference=0.02,
    first_rmsd_median=0.3, first_rmsd_max=1.0, later_rmsf_correlation=0.95,
)


def _align(mobile, reference):
    centered = mobile - mobile.mean(axis=-2, keepdims=True)
    target = reference - reference.mean(axis=-2, keepdims=True)
    u, _, vh = np.linalg.svd(np.swapaxes(centered, -1, -2) @ target)
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    correction[..., 2, 2] = np.linalg.det(u @ vh)
    return centered @ (u @ correction @ vh) + reference.mean(axis=-2, keepdims=True)


def _rmsf(ca, reference):
    aligned = _align(ca, reference)
    return np.sqrt(np.mean(np.sum((aligned - aligned.mean(axis=0)) ** 2, axis=-1), axis=0))


def _pearson(a, b):
    # Identical constant profiles have parity; other constant profiles are undefined.
    if np.allclose(a, b, rtol=0, atol=1e-12):
        return 1.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _distribution(values):
    return dict(count=int(values.size), min=float(values.min()),
                p05=float(np.quantile(values, .05)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(values.max()),
                mean=float(values.mean()), std=float(values.std()))


def compare_trajectories(reference, candidate, *, md=None, residue_mask=None,
                         chain_id=None, residue_index=None, thresholds=None):
    """Compare matching [rollout, frame, residue, atom, xyz] arrays.

    MD has shape [frame, residue, atom, xyz]. All RMSF uses the same first
    reference CA frame for rigid alignment. Thresholds are absolute differences,
    except later_rmsf_correlation which is a lower bound.
    """
    reference, candidate = (np.asarray(x, dtype=float) for x in (reference, candidate))
    if (reference.shape != candidate.shape or reference.ndim != 5
            or reference.shape[-1] != 3 or reference.shape[-2] < 3
            or min(reference.shape[:2]) < 1 or reference.shape[1] < 2):
        raise ValueError('Expected matching [rollout, frame>=2, residue, atom>=3, 3] arrays')
    if not all(np.isfinite(x).all() for x in (reference, candidate)):
        raise ValueError('Coordinates must be finite')
    n = reference.shape[2]
    mask = np.ones(n, dtype=bool) if residue_mask is None else np.asarray(residue_mask, dtype=bool)
    chains = np.zeros(n) if chain_id is None else np.asarray(chain_id)
    indices = np.arange(n) if residue_index is None else np.asarray(residue_index)
    if any(x.shape != (n,) for x in (mask, chains, indices)) or mask.sum() < 3:
        raise ValueError('Metadata must match residues; at least three CA atoms are required')
    pairs = mask[:-1] & mask[1:] & (chains[:-1] == chains[1:]) & (np.diff(indices) == 1)
    if not pairs.any():
        raise ValueError('No consecutive same-chain residue pairs')
    limits = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        if thresholds.keys() - limits.keys():
            raise ValueError('Unknown threshold')
        limits.update(thresholds)
    if any(not np.isfinite(v) or v < 0 for v in limits.values()) or limits['later_rmsf_correlation'] > 1:
        raise ValueError('Invalid threshold')
    geometry = []
    for x in (reference, candidate):
        cn = np.linalg.norm(x[..., :-1, 2, :] - x[..., 1:, 0, :], axis=-1)[..., pairs]
        ca = np.linalg.norm(x[..., :-1, 1, :] - x[..., 1:, 1, :], axis=-1)[..., pairs]
        geometry.append(dict(bad_cn_ratio=float(((cn < 1.1) | (cn > 1.6)).mean()),
                             bad_ca_ratio=float(((ca < 3.4) | (ca > 4.2)).mean()),
                             cn_distribution=_distribution(cn), ca_distribution=_distribution(ca)))
    checks = {}

    def check(name, value, minimum=False):
        checks[name] = dict(value=value, threshold=limits[name],
                            operator='>=' if minimum else '<=',
                            passed=value is not None and bool(value >= limits[name] if minimum else value <= limits[name]))

    check('bad_cn_ratio_difference', abs(geometry[0]['bad_cn_ratio'] - geometry[1]['bad_cn_ratio']))
    for stat in ('mean', 'std'):
        check(f'ca_{stat}_difference', abs(geometry[0]['ca_distribution'][stat] - geometry[1]['ca_distribution'][stat]))
    a, b = reference[..., mask, 1, :], candidate[..., mask, 1, :]
    rmsd = np.sqrt(np.mean(np.sum((_align(b[0], a[0]) - a[0]) ** 2, axis=-1), axis=-1))
    check('first_rmsd_median', float(np.median(rmsd)))
    check('first_rmsd_max', float(rmsd.max()))
    later = []
    for i in range(1, len(a)):
        later.append(_pearson(_rmsf(a[i], a[0, 0]), _rmsf(b[i], a[0, 0])))
    if later:
        check('later_rmsf_correlation', None if None in later else min(later), minimum=True)
    md_correlations = None
    if md is not None:
        md = np.asarray(md, dtype=float)
        if (md.ndim != 4 or md.shape[1:] != reference.shape[2:]
                or md.shape[0] < 2 or not np.isfinite(md).all()):
            raise ValueError('MD must have matching residue/atom dimensions and finite frames>=2')
        md_profile = _rmsf(md[:, mask, 1], a[0, 0])
        md_correlations = [_pearson(_rmsf(x.reshape(-1, mask.sum(), 3), a[0, 0]), md_profile) for x in (a, b)]
        check('md_rmsf_pearson_difference', None if None in md_correlations else abs(md_correlations[0] - md_correlations[1]))
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks,
                geometry=dict(reference=geometry[0], candidate=geometry[1]),
                first_rollout_ca_rmsd_angstrom=rmsd.tolist(), later_rollout_rmsf_correlations=later,
                md_rmsf_pearson=md_correlations,
                not_evaluated=(["md_rmsf_pearson_difference"] if md is None else []) +
                (["later_rmsf_correlation"] if not later else []))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Precision parity comparator (Å, N/CA/C atom order)')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    results = []
    seen = set()
    for entry in manifest['targets']:
        identity = (entry['protein'], entry['seed'])
        if identity in seen:
            raise ValueError('Duplicate protein/seed')
        seen.add(identity)
        def load(key):
            return np.load(args.manifest.parent / entry[key], allow_pickle=False)
        report = compare_trajectories(load('reference'), load('candidate'),
                                      md=load('md') if entry.get('md') else None,
                                      **{k: entry[k] for k in ('residue_mask', 'chain_id', 'residue_index') if k in entry},
                                      thresholds=manifest.get('thresholds'))
        results.append(dict(protein=identity[0], seed=identity[1], **report))
    if not results:
        raise ValueError('At least one target is required')
    output = dict(kind='Precision parity', units='angstrom', passed=all(r['passed'] for r in results),
                  targets=results, manifest=manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    lines = ['# Precision parity', '', 'PASS' if output['passed'] else 'FAIL']
    for result in results:
        lines += ['', f"{result['protein']} / seed {result['seed']}"]
        lines += [f"- {name}: {'PASS' if c['passed'] else 'FAIL'}; {c['value']} {c['operator']} {c['threshold']}" for name, c in result['checks'].items()]
        lines += [f'- {name}: 未評估' for name in result['not_evaluated']]
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return 0 if output['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
