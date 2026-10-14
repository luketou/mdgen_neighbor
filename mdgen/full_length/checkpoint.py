"""Shape-safe reuse of weights across dense and full-length model versions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def load_compatible_state_dict(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> dict[str, object]:
    """Load exact name/shape matches and report all migration decisions."""
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise ValueError("checkpoint must be a state dict or contain 'state_dict'")
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    unexpected: list[str] = []

    for original_name, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            unexpected.append(str(original_name))
            continue
        candidates = [str(original_name)]
        if str(original_name).startswith("model."):
            candidates.append(str(original_name)[len("model.") :])
        name = next((candidate for candidate in candidates if candidate in target), None)
        if name is None:
            unexpected.append(str(original_name))
        elif target[name].shape != value.shape:
            shape_mismatch.append(name)
        else:
            compatible[name] = value

    model.load_state_dict(compatible, strict=False)
    newly_initialized = sorted(set(target) - set(compatible))
    return {
        "loaded": sorted(compatible),
        "loaded_count": len(compatible),
        "shape_mismatch": sorted(shape_mismatch),
        "unexpected": sorted(unexpected),
        "newly_initialized": newly_initialized,
        "newly_initialized_count": len(newly_initialized),
        "newly_initialized_summary_parameters": [
            name for name in newly_initialized if "cell_summaries." in name
        ],
    }
