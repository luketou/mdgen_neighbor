import numpy as np

from mdgen.precision_parity import compare_trajectories


def fixture():
    rng = np.random.default_rng(9)
    return rng.normal(size=(2, 8, 6, 3, 3))


def test_identical_trajectories_pass():
    x = fixture()
    report = compare_trajectories(x, x, md=x[0])
    assert report['passed']
    assert report['checks']['first_rmsd_max']['value'] < 1e-12
    assert report['checks']['md_rmsf_pearson_difference']['value'] == 0


def test_rigid_motion_is_removed_by_alignment():
    x = fixture()
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    report = compare_trajectories(x, x @ rotation + [8, -3, 2])
    assert report['passed']
    assert max(report['first_rollout_ca_rmsd_angstrom']) < 1e-12


def test_known_aligned_rmsd_and_configurable_threshold():
    x = fixture()
    # Scaling centered coordinates cannot be removed by a rigid alignment.
    x[:, :, :, 1] = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                             [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    y = x.copy()
    y[0, :, :, 1] *= 3
    report = compare_trajectories(x, y)
    np.testing.assert_allclose(report['first_rollout_ca_rmsd_angstrom'], 2)
    assert not report['checks']['first_rmsd_max']['passed']
    assert compare_trajectories(x, y, thresholds={'first_rmsd_max': 3})['checks']['first_rmsd_max']['passed']


def test_later_rollout_and_md_detect_different_fluctuations():
    x = fixture()
    y = x.copy()
    y[1, :, 0, 1] *= 100
    report = compare_trajectories(x, y, md=x.reshape(-1, 6, 3, 3))
    assert not report['passed']
    assert not report['checks']['later_rmsf_correlation']['passed']
    assert not report['checks']['md_rmsf_pearson_difference']['passed']


def test_geometry_screening_and_chain_gaps():
    x = np.zeros((1, 3, 4, 3, 3))
    x[..., 0, 0] = np.arange(4) * 3.8
    x[..., 1, 0] = np.arange(4) * 3.8
    x[..., 2, 0] = np.arange(4) * 3.8 + 2.5
    y = x.copy()
    y[..., 0, 2, 0] += 10
    report = compare_trajectories(x, y)
    assert report['geometry']['reference']['bad_cn_ratio'] == 0
    assert report['checks']['bad_cn_ratio_difference']['value'] == 1 / 3
    assert not report['passed']
    excluded = compare_trajectories(x, y, chain_id=[0, 1, 1, 1])
    assert excluded['passed']
    assert excluded['geometry']['reference']['ca_distribution']['count'] == 6
    assert compare_trajectories(x, y, residue_index=[0, 2, 3, 4])['passed']


def test_invalid_coordinates_rejected():
    import pytest
    x = fixture()
    x[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        compare_trajectories(x, x)


def test_cli_writes_strict_json_and_readable_report(tmp_path):
    import json
    from mdgen.precision_parity import main
    x = fixture()
    np.save(tmp_path / 'x.npy', x)
    manifest = {'targets': [{'protein': 'synthetic', 'seed': 9,
                             'reference': 'x.npy', 'candidate': 'x.npy'}]}
    (tmp_path / 'input.json').write_text(json.dumps(manifest))
    assert main([str(tmp_path / 'input.json'), '--output', str(tmp_path / 'report.json')]) == 0
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['passed']
    assert report['targets'][0]['not_evaluated'] == ['md_rmsf_pearson_difference']
    assert 'PASS' in (tmp_path / 'report.md').read_text()
    x[0, :, 0, 1] *= 100
    np.save(tmp_path / 'y.npy', x)
    manifest['targets'][0]['candidate'] = 'y.npy'
    (tmp_path / 'input.json').write_text(json.dumps(manifest))
    assert main([str(tmp_path / 'input.json'), '--output', str(tmp_path / 'report.json')]) == 1
