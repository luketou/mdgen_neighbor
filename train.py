from mdgen.parsing import parse_train_args
args = parse_train_args()  # runtime-args-separation
# sdpa-route:start
use_sdpa = not args.manual_attention
delattr(args, 'manual_attention')
# sdpa-route:end
from mdgen.logger import get_logger
logger = get_logger(__name__)

import torch, os, wandb
from mdgen.dataset import MDGenDataset
from mdgen.wrapper import NewMDGenWrapper
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl


# seed-initialization:start
if args.train_seed is not None:
    pl.seed_everything(args.train_seed, workers=True)
# seed-initialization:end

from mdgen.precision import configure_precision, PrecisionReport
configure_precision(args.precision_config)

if args.wandb:
    wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        settings=wandb.Settings(start_method="fork"),
        project="mdgen",
        name=args.run_name,
        config=args,
    )


trainset = MDGenDataset(args, split=args.train_split)

if args.overfit:
    valset = trainset
else:
    valset = MDGenDataset(args, split=args.val_split, repeat=args.val_repeat, region_seed=0)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
model = NewMDGenWrapper(args, use_sdpa=use_sdpa)  # sdpa-route

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    deterministic=args.deterministic,  # deterministic-execution
    benchmark=args.benchmark,  # deterministic-benchmark-guard
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    precision=args.precision,
    enable_progress_bar=not args.wandb,
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"],
    callbacks=[
        PrecisionReport(args.precision_config, logger),
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            save_top_k=-1,
            every_n_epochs=args.ckpt_freq,
        ),
        ModelSummary(max_depth=2),
    ],
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    check_val_every_n_epoch=args.val_epoch_freq,
    logger=False
)

if args.validate:
    trainer.validate(
        model,
        val_loader,
        ckpt_path=args.ckpt,
        weights_only=False,  # runtime-compatibility
    )
else:
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.ckpt,
        weights_only=False,  # runtime-compatibility
    )
