"""Training region sizes on two synthetic 7,500-residue frame-0 structures (CPU only).

    python docs/validation/12-region-budget/synthetic_regions.py <repo root> > synthetic_regions.json

random_walk is the frame 0 of `train_step_probe --synthetic_residues 7500 --num_frames 250 --seed 0`;
compact is a serpentine walk through a 5.2 A cubic lattice (~140 A^3 per residue, protein-interior
density). For each halo depth the report gives the seed-region size of every group, and for each
budget the loss / halo / context residues of the regions grown from seeds 0-7.
"""
import json
import sys

import torch

sys.path.insert(0, sys.argv[1])
from mdgen.full_length.group_region import _halo, _residues, region_subpartition, sample_region_groups  # noqa: E402
from mdgen.full_length.partition import PartitionParams, build_spatial_partition  # noqa: E402

L = 7500


def random_walk():
    g = torch.Generator().manual_seed(0)
    step = torch.randn(250, L, 3, generator=g)
    return (3.8 * step / step.norm(dim=-1, keepdim=True)).cumsum(1)[0]


def compact(spacing=5.2):
    n = 20
    pts = []
    for z in range(n):
        for y in (range(n) if z % 2 == 0 else reversed(range(n))):
            xs = range(n) if (y + z) % 2 == 0 else reversed(range(n))
            pts.extend((x, y, z) for x in xs)
    return torch.tensor(pts[:L], dtype=torch.float32) * spacing


params = PartitionParams(cell_width=16., contact_radius=16., core_capacity=128, ghost_capacity=128)
report = {}
for name, ca in [('random_walk', random_walk()), ('compact', compact())]:
    partition = build_spatial_partition(ca, mask=torch.ones(L, dtype=torch.bool),
                                        chain_id=torch.zeros(L, dtype=torch.long),
                                        residue_index=torch.arange(L), **params.partition_kwargs())
    seed_regions = {rings: [len(_residues(partition, [g, *_halo(partition, [g], rings)]))
                            for g in range(len(partition.groups))] for rings in (0, 1, 2)}
    rows = []
    for budget in (512, 768, 1024):
        for rings in (1, 2):
            for seed in range(8):
                try:
                    sel = sample_region_groups(partition, ca, region_residues=budget, halo_rings=rings,
                                               generator=torch.Generator().manual_seed(seed))
                except ValueError as e:
                    rows.append(dict(budget=budget, rings=rings, seed=seed, error=str(e)[:80]))
                    continue
                region = region_subpartition(partition, sel)
                loss = int(region.loss_core_mask.sum())
                owned = int(region.partition.residue_mask.sum())
                rows.append(dict(budget=budget, rings=rings, seed=seed, loss_groups=len(sel.loss_groups),
                                 halo_groups=len(sel.halo_groups), residues=len(sel.residues), loss=loss,
                                 halo=owned - loss, context=len(sel.residues) - owned))
    report[name] = dict(groups=len(partition.groups), seed_regions=seed_regions, rows=rows)
json.dump(report, sys.stdout)
