import json
import logging

import pytest
import torch
import pytorch_lightning as pl

from mdgen.precision import configure_precision, PrecisionReport


@pytest.mark.parametrize('name, precision, dtype, matmul, tf32', [
    ('fp32-reference', '32-true', torch.float32, 'highest', False),
    ('bf16-train', 'bf16-mixed', torch.bfloat16, 'high', True),
])
def test_precision_training(tmp_path, name, precision, dtype, matmul, tf32):
    previous = (torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    class TinyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)

        def training_step(self, batch, batch_idx):
            assert torch.is_autocast_enabled('cpu') == tf32
            assert torch.get_float32_matmul_precision() == matmul
            assert torch.backends.cuda.matmul.allow_tf32 == tf32
            assert torch.backends.cudnn.allow_tf32 == tf32
            assert self.layer.weight.dtype == torch.float32
            output = self.layer(batch[0])
            assert output.dtype == dtype
            return output.float().square().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    log_path = tmp_path / 'precision.log'
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        configure_precision(name)
        trainer = pl.Trainer(accelerator='cpu', precision=precision, max_steps=1,
            logger=False, enable_checkpointing=False, enable_model_summary=False,
            enable_progress_bar=False, callbacks=[PrecisionReport(name, logger)])
        trainer.fit(TinyModel(), torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.ones(2, 2)), batch_size=2))
        report = json.loads(log_path.read_text().split('Precision configuration: ')[1])
        assert report['autocast_enabled'] == tf32
        assert report['autocast_dtype'] == ('torch.bfloat16' if tf32 else None)
        assert report['float32_matmul_precision'] == matmul
        assert report['cuda_matmul_allow_tf32'] == tf32
        assert report['cudnn_allow_tf32'] == tf32
    finally:
        logger.removeHandler(handler)
        handler.close()
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
