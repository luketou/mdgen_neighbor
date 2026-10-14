from mdgen.parsing import parse_train_args
args = parse_train_args(include_runtime_args=True)  # runtime-args-separation
# sdpa-route:start
use_sdpa = not args.manual_attention
# sdpa-route:end
print_sdpa_backend = args.print_sdpa_backend  # sdpa-diagnostics
peak_memory_path = args.peak_memory  # peak_memory
execution_time_path = args.execution_time  # execution_time
# sdpa-route:start
delattr(args, 'manual_attention')
# sdpa-route:end
delattr(args, 'print_sdpa_backend')  # sdpa-diagnostics
delattr(args, 'peak_memory')  # peak_memory
delattr(args, 'execution_time')  # execution_time
from mdgen.logger import get_logger
logger = get_logger(__name__)

import json  # peak_memory
import torch, os, wandb
from mdgen.dataset import MDGenDataset
from mdgen.wrapper import NewMDGenWrapper
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl


# sdpa-diagnostics
def get_sdpa_backend(operator_name):
    backend_operators = {
        '_scaled_dot_product_flash_attention': 'FLASH_ATTENTION',
        '_scaled_dot_product_efficient_attention': 'EFFICIENT_ATTENTION',
        '_scaled_dot_product_cudnn_attention': 'CUDNN_ATTENTION',
        '_scaled_dot_product_attention_math': 'MATH',
    }
    for operator, backend in backend_operators.items():
        if operator in operator_name:
            return backend
    return None


# sdpa-diagnostics
def get_attention_module_settings(model):
    settings = {}
    for name, module in model.named_modules():
        if not hasattr(module, 'last_attention_backend'):
            continue
        settings[name] = {
            'last_attention_backend': module.last_attention_backend,
            'last_sdpa_fallback_reason': module.last_sdpa_fallback_reason,
        }
    return settings


# sdpa-diagnostics
class SDPABackendReportCallback(pl.Callback):
    def __init__(self, output_path, use_sdpa):
        super().__init__()
        self.output_path = os.path.abspath(output_path)
        self.use_sdpa = use_sdpa
        self.profiler = None
        self.reported = False
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.reported:
            return
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        )
        self.profiler.start()

    def on_before_backward(self, trainer, pl_module, loss):
        if self.reported or self.profiler is None:
            return
        self.profiler.stop()
        self._write_report(pl_module)
        self.profiler = None
        self.reported = True
        print(f'SDPA backend report saved to: {self.output_path}')

    def _write_report(self, model):
        lines = [
            'SDPA backend report',
            'mode: training forward',
            f'requested_attention_path: {"sdpa" if self.use_sdpa else "manual"}',
            f'torch_version: {torch.__version__}',
            f'cuda_wheel: {torch.version.cuda}',
            f'gpu: {torch.cuda.get_device_name(0)}',
            f'lightning_precision: {args.precision}',
            '',
            'Enabled PyTorch SDPA backends:',
        ]
        enabled_checks = (
            ('FLASH_ATTENTION', 'flash_sdp_enabled'),
            ('EFFICIENT_ATTENTION', 'mem_efficient_sdp_enabled'),
            ('CUDNN_ATTENTION', 'cudnn_sdp_enabled'),
            ('MATH', 'math_sdp_enabled'),
        )
        for backend, check_name in enabled_checks:
            check = getattr(torch.backends.cuda, check_name, None)
            lines.append(f'{backend}: {check() if check is not None else "unknown"}')

        lines.extend(['', 'MDGen attention modules:'])
        found_module = False
        for name, module in model.named_modules():
            if not hasattr(module, 'last_attention_backend'):
                continue
            found_module = True
            backend = module.last_attention_backend or 'not_executed'
            fallback = module.last_sdpa_fallback_reason
            message = f'{name}: path={backend}'
            if fallback is not None:
                message += f', fallback={fallback}'
            lines.append(message)
        if not found_module:
            lines.append('No attention modules expose backend information.')

        lines.extend(['', 'PyTorch SDPA operators observed:'])
        observed_backends = set()
        observed_operators = []
        events = self.profiler.key_averages(group_by_input_shape=True)
        for event in events:
            if 'scaled_dot_product' not in event.key:
                continue
            backend = get_sdpa_backend(event.key)
            if backend is not None:
                observed_backends.add(backend)
            observed_operators.append((event, backend))

        if observed_backends:
            lines.append(f'selected_backends: {", ".join(sorted(observed_backends))}')
        else:
            lines.append('selected_backends: none detected')

        if not observed_operators:
            lines.append('No scaled-dot-product attention operators were observed.')
        for event, backend in observed_operators:
            self_device_time = getattr(
                event,
                'self_device_time_total',
                getattr(event, 'self_cuda_time_total', 0.0),
            )
            device_time = getattr(
                event,
                'device_time_total',
                getattr(event, 'cuda_time_total', 0.0),
            )
            lines.extend([
                f'operator: {event.key}',
                f'  backend: {backend or "dispatcher/unknown"}',
                f'  calls: {event.count}',
                f'  input_shapes: {event.input_shapes}',
                f'  self_cpu_time_total_us: {event.self_cpu_time_total}',
                f'  cpu_time_total_us: {event.cpu_time_total}',
                f'  self_device_time_total_us: {self_device_time}',
                f'  device_time_total_us: {device_time}',
            ])

        with open(self.output_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')


# peak_memory
def write_peak_memory_report(
    output_path, trainer, args, use_sdpa, oom=False, error=None
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    attention_path = 'sdpa' if use_sdpa else 'manual'
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    report = {
        'attention_path': attention_path,
        'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'precision': args.precision,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic_algorithms_enabled': (
            torch.are_deterministic_algorithms_enabled()
        ),
        'attention_modules': get_attention_module_settings(
            trainer.lightning_module
        ),
        'batch_size': args.batch_size,
        'num_frames': args.num_frames,
        'crop': args.crop,
        'gradient_checkpointing': args.grad_checkpointing,
        'completed_steps': trainer.global_step,
        'oom': oom,
        'peak_memory_allocated_bytes': peak_allocated,
        'peak_memory_allocated_gb': peak_allocated / 1024 ** 3,
        'peak_memory_reserved_bytes': peak_reserved,
        'peak_memory_reserved_gb': peak_reserved / 1024 ** 3,
    }
    if error is not None:
        report['error'] = str(error)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f'Peak memory report saved to: {output_path}')


# execution_time
class ExecutionTimeCallback(pl.Callback):
    warmup_steps = 5

    def __init__(self, output_path, args, use_sdpa):
        super().__init__()
        self.output_path = os.path.abspath(output_path)
        self.args = args
        self.use_sdpa = use_sdpa
        self.train_batches_seen = 0
        self.current_start_event = None
        self.event_pairs = []
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.train_batches_seen < self.warmup_steps:
            return
        self.current_start_event = torch.cuda.Event(enable_timing=True)
        self.current_start_event.record()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.current_start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.event_pairs.append((self.current_start_event, end_event))
            self.current_start_event = None
        self.train_batches_seen += 1

    def on_fit_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        step_times = [
            start.elapsed_time(end) / 1000.0
            for start, end in self.event_pairs
        ]
        report = {
            'attention_path': 'sdpa' if self.use_sdpa else 'manual',
            'gpu': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'precision': self.args.precision,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms_enabled': (
                torch.are_deterministic_algorithms_enabled()
            ),
            'attention_modules': get_attention_module_settings(pl_module),
            'batch_size': self.args.batch_size,
            'num_frames': self.args.num_frames,
            'crop': self.args.crop,
            'gradient_checkpointing': self.args.grad_checkpointing,
            'completed_steps': trainer.global_step,
            'warmup_steps': min(self.warmup_steps, self.train_batches_seen),
            'measured_steps': len(step_times),
            'mean_seconds_per_step': (
                sum(step_times) / len(step_times) if step_times else None
            ),
            'median_seconds_per_step': (
                sorted(step_times)[len(step_times) // 2]
                if len(step_times) % 2 == 1
                else (
                    sum(sorted(step_times)[len(step_times) // 2 - 1:len(step_times) // 2 + 1]) / 2
                    if step_times else None
                )
            ),
            'min_seconds_per_step': min(step_times) if step_times else None,
            'max_seconds_per_step': max(step_times) if step_times else None,
        }
        with open(self.output_path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print(f'Execution time report saved to: {self.output_path}')


# seed-initialization:start
if args.train_seed is not None:
    pl.seed_everything(args.train_seed, workers=True)
# seed-initialization:end

torch.set_float32_matmul_precision('medium')  # fp32-matmul-precision

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
    valset = MDGenDataset(args, split=args.val_split, repeat=args.val_repeat)

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
# sdpa-diagnostics:start
callbacks = [
    ModelCheckpoint(
        dirpath=os.environ["MODEL_DIR"],
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
    ),
    ModelSummary(max_depth=2),
]
if print_sdpa_backend is not None:
    callbacks.append(SDPABackendReportCallback(print_sdpa_backend, use_sdpa))
# sdpa-diagnostics:end
# execution_time:start
if execution_time_path is not None:
    if not torch.cuda.is_available():
        raise RuntimeError('--execution_time requires a CUDA device')
    callbacks.append(ExecutionTimeCallback(execution_time_path, args, use_sdpa))
# execution_time:end

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
    callbacks=callbacks, # sdpa-diagnostics
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
    # peak_memory:start
    if peak_memory_path is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('--peak_memory requires a CUDA device')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # peak_memory:end
    # peak_memory:start
    try:
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=args.ckpt,
            weights_only=False,  # runtime-compatibility
        )
    except torch.cuda.OutOfMemoryError as error:
        if peak_memory_path is not None:
            write_peak_memory_report(
                peak_memory_path,
                trainer,
                args,
                use_sdpa,
                oom=True,
                error=error,
            )
        raise
    else:
        if peak_memory_path is not None:
            torch.cuda.synchronize()
            write_peak_memory_report(
                peak_memory_path, trainer, args, use_sdpa, oom=False
            )
    # peak_memory:end
