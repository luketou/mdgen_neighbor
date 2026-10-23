"""Seed-region sizes and loss residues on a real structure (C-alpha from a PDB file)."""
import sys

import torch

repo, pdb = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from mdgen.full_length.group_region import _halo, _residues, sample_region_groups, region_subpartition  # noqa
from mdgen.full_length.partition import PartitionParams, build_spatial_partition  # noqa

ca, chains, resid, seen = [], [], [], set()
with open(pdb) as f:
    for line in f:
        if line.startswith('ENDMDL'):
            break
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in ' A':
            key = (line[21], int(line[22:26]), line[26])
            if key in seen:
                continue
            seen.add(key)
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chains.append(ord(line[21]))
            resid.append(int(line[22:26]))
ca = torch.tensor(ca)
L = len(ca)
_, chain_id = torch.unique(torch.tensor(chains), return_inverse=True)
params = PartitionParams(cell_width=16., contact_radius=16., core_capacity=128, ghost_capacity=128)
p = build_spatial_partition(ca, mask=torch.ones(L, dtype=torch.bool), chain_id=chain_id,
                            residue_index=torch.tensor(resid), **params.partition_kwargs())
print(f'{pdb.rsplit("/", 1)[-1]}: {L} residues, {int(chain_id.max()) + 1} chains, {len(p.groups)} groups')
for rings in (0, 1, 2):
    sizes = torch.tensor([len(_residues(p, [g, *_halo(p, [g], rings)])) for g in range(len(p.groups))],
                         dtype=torch.float)
    q = torch.quantile(sizes, torch.tensor([0.5, 0.9, 1.0])).tolist()
    fits = {b: f'{(sizes <= b).float().mean().item():.0%}' for b in (768, 1024, 1536, 2048)}
    print(f'  rings={rings} seed-region median/p90/max = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}  seeds fitting {fits}')
for budget in (768, 1024):
    for rings in (1, 2):
        loss, fail = [], 0
        for seed in range(16):
            try:
                sel = sample_region_groups(p, ca, region_residues=budget, halo_rings=rings,
                                           generator=torch.Generator().manual_seed(seed))
            except ValueError:
                fail += 1
                continue
            loss.append(int(region_subpartition(p, sel).loss_core_mask.sum()))
        print(f'  budget={budget} rings={rings}: unfit {fail}/16, loss residues {sorted(loss)}')
