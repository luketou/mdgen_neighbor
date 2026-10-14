#!/bin/bash
#SBATCH --job-name=mdgen-l7500-chunked
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mdgen_l7500_chunked_%j.out
#SBATCH --error=logs/mdgen_l7500_chunked_%j.err

set -eo pipefail
source job/env.sh
: "${MDGEN_ATLAS_CKPT:?set MDGEN_ATLAS_CKPT to the ATLAS checkpoint path}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python - <<'PY'
import os

import torch

from mdgen.model.ipa import InvariantPointAttention
from mdgen.model.layers import gelu
from mdgen.rigid_utils import Rigid
from mdgen.wrapper import NewMDGenWrapper

checkpoint = os.environ["MDGEN_ATLAS_CKPT"]
model = NewMDGenWrapper.load_from_checkpoint(
    checkpoint, use_sdpa=True, weights_only=False
)
model.eval().float().to("cuda")
ipa = model.model.ipa_layers[0].ipa
length = 64
single = torch.randn(1, length, ipa.c_s, device="cuda")
frames = Rigid.identity(
    (1, length), dtype=torch.float32, device="cuda", requires_grad=False, fmt="rot_mat"
)
mask = torch.ones((1, length), dtype=torch.float32, device="cuda")

with torch.inference_mode():
    ipa.query_chunk_size = length + 1
    reference = ipa(single, frames, frame_mask=mask)
    ipa.query_chunk_size = 16
    chunked = ipa(single, frames, frame_mask=mask)

torch.testing.assert_close(chunked, reference, rtol=1e-5, atol=1e-5)
print(
    "IPA chunk equivalence passed:",
    f"max_abs_error={(chunked - reference).abs().max().item():.3e}",
)

layer = model.model.layers[0]
ffn_input = torch.randn(1, 4, 64, layer.embed_dim, device="cuda")
with torch.inference_mode():
    ffn_reference = layer.fc2(gelu(layer.fc1(ffn_input)))
    layer.ffn_chunk_size = 32
    ffn_chunked = layer._ffn(ffn_input.clone())

torch.testing.assert_close(ffn_chunked, ffn_reference, rtol=1e-5, atol=1e-5)
print(
    "FFN chunk equivalence passed:",
    f"max_abs_error={(ffn_chunked - ffn_reference).abs().max().item():.3e}",
)
PY

python scripts/diagnostics/benchmark_l_scaling.py \
  --sim_ckpt "$MDGEN_ATLAS_CKPT" \
  --output "docs/validation/mdgen2026/r2.1/C49_E2-cu128-l40s-chunked.json" \
  --lengths 7500 \
  --seed 137 \
  --deterministic
