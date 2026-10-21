"""Explicit training arithmetic and reports from Lightning's active context."""
import json

import torch
from pytorch_lightning import Callback


def configure_precision(name):
    if name not in ('fp32-reference', 'bf16-train'):
        raise ValueError(f'Unknown precision configuration: {name}')
    allow_tf32 = name == 'bf16-train'
    torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


class PrecisionReport(Callback):
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger

    def _report(self, trainer, pl_module):
        device_type = pl_module.device.type
        with trainer.precision_plugin.forward_context():
            enabled = torch.is_autocast_enabled(device_type)
            report = {
                'precision_config': self.name,
                'lightning_precision': trainer.precision,
                'device_type': device_type,
                'autocast_enabled': enabled,
                'autocast_dtype': str(torch.get_autocast_dtype(device_type)) if enabled else None,
                'parameter_dtypes': sorted({str(p.dtype) for p in pl_module.parameters()}),
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            }
        self.logger.info('Precision configuration: %s', json.dumps(report, sort_keys=True))

    def on_fit_start(self, trainer, pl_module):
        self._report(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        self._report(trainer, pl_module)
