import argparse  # seed-deterministic-args
from argparse import ArgumentParser
import os


# Single source for local options; older checkpoint args predate these keys.
LOCAL_DEFAULTS = dict(
    local_spatial=False,
    cell_summaries=False,
    spatial_cell_width=16.,
    spatial_contact_radius=16.,
    core_capacity=128,
    ghost_capacity=128,
    group_batch_size=4,
    ffn_chunk_size=65536,
    spatial_attention_backend='sdpa',
)


def fill_local_defaults(args):
    for key, value in LOCAL_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'must be a positive integer, got {value}')
    return value


# Options an inference run may override on top of the checkpoint's args.
LOCAL_OVERRIDE_TYPES = dict(
    local_spatial=argparse.BooleanOptionalAction,
    cell_summaries=argparse.BooleanOptionalAction,
    spatial_cell_width=float,
    spatial_contact_radius=float,
    core_capacity=int,
    ghost_capacity=int,
    group_batch_size=int,
    ffn_chunk_size=positive_int,
)


def add_local_override_args(parser):
    """None means: keep the value stored in the checkpoint."""
    for key, kind in LOCAL_OVERRIDE_TYPES.items():
        if kind is argparse.BooleanOptionalAction:
            parser.add_argument(f'--{key}', action=kind, default=None)
        else:
            parser.add_argument(f'--{key}', type=kind, default=None)


def apply_local_overrides(model_args, cli_args):
    for key in LOCAL_OVERRIDE_TYPES:
        if getattr(cli_args, key) is not None:
            setattr(model_args, key, getattr(cli_args, key))
    return fill_local_defaults(model_args)


def global_mode_override_error(model_args, cli_args):
    """Message for local-only overrides that would be silently ignored."""
    if not model_args.local_spatial and cli_args.ffn_chunk_size is not None:
        return '--ffn_chunk_size requires local_spatial'
    return None


def add_determinism_args(parser):
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic algorithms when available.",
    )
    parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the cuDNN benchmark autotuner.",
    )


def check_determinism_args(parser, args):
    if args.deterministic and args.benchmark:
        parser.error("--deterministic and --benchmark cannot both be enabled")


def parse_train_args(include_runtime_args=False):  # runtime-args-separation
    parser = ArgumentParser()

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_seed", type=int, default=None)  # seed-deterministic-args
    parser.add_argument("--model_dir",type=str,default=None,help="Directory used as MODEL_DIR",)
    add_determinism_args(parser)  # seed-deterministic-args
    
    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--overfit_peptide", type=str, default=None)
    group.add_argument("--overfit_frame", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--val_repeat", type=int, default=1)
    group.add_argument("--inference_batches", type=int, default=0)
    group.add_argument("--batch_size", type=int, default=8)
    group.add_argument("--val_freq", type=int, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--no_validate", action='store_true')
    group.add_argument("--designability_freq", type=int, default=1)

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=1)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument(
        '--precision',
        type=str,
        choices=('fp32-reference', 'bf16-train', '32-true', 'bf16-mixed'),
        default='fp32-reference',
        help='Named training precision configuration (Lightning aliases also accepted).',
    )
    
    ## Training data 
    group = parser.add_argument_group("Training data settings")
    group.add_argument('--train_split', type=str, default=None, required=True)
    group.add_argument('--val_split', type=str, default=None, required=True)
    group.add_argument('--data_dir', type=str, default=None, required=True)
    group.add_argument('--num_frames', type=int, default=50)
    group.add_argument('--crop', type=int, default=256)
    group.add_argument('--suffix', type=str, default='')
    group.add_argument('--atlas', action='store_true')
    group.add_argument('--replicates', type=int, nargs='+', default=[1, 2, 3],
                       help='--atlas replicate numbers to sample from')
    group.add_argument('--val_replicates', type=int, nargs='+', default=None,
                       help='--atlas replicates for validation (default: --replicates)')
    group.add_argument('--copy_frames', action='store_true')
    group.add_argument('--no_pad', action='store_true')
    group.add_argument('--short_md', action='store_true')
    group.add_argument('--training_region', action='store_true',
                       help='Train on groups of the frame-0 full-protein partition instead of --crop.')
    group.add_argument('--region_residues', type=positive_int, default=896,
                       help='Training region size, counting loss, halo and context residues.')
    group.add_argument('--halo_rings', type=int, default=1,
                       help='Rings of halo groups around the loss groups (no loss).')
    group.add_argument('--partition_dir', type=str, default=None,
                       help='<npy stem>_partition.npz from mdgen.full_length.partition_store; '
                            'without it each item partitions frame 0 itself.')

    ### Masking settings
    group = parser.add_argument_group("Masking settings")
    group.add_argument('--design_key_frames', action='store_true')
    group.add_argument('--no_aa_emb', action='store_true')
    group.add_argument("--no_torsion", action='store_true')
    group.add_argument("--no_design_torsion", action='store_true')
    group.add_argument("--supervise_no_torsions", action='store_true')
    group.add_argument("--supervise_all_torsions", action='store_true')

    ## Ablations settings
    group = parser.add_argument_group("Ablations settings")
    group.add_argument('--no_offsets', action='store_true')
    group.add_argument('--no_frames', action='store_true')
    
    
    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--hyena', action='store_true')
    group.add_argument('--no_rope', action='store_true')
    group.add_argument('--manual_attention', action='store_true')  # sdpa-route
    group.add_argument('--local_spatial', action='store_true')
    group.add_argument('--cell_summaries', action=argparse.BooleanOptionalAction, default=LOCAL_DEFAULTS['cell_summaries'])
    group.add_argument('--spatial_cell_width', type=float, default=LOCAL_DEFAULTS['spatial_cell_width'])
    group.add_argument('--spatial_contact_radius', type=float, default=LOCAL_DEFAULTS['spatial_contact_radius'])
    group.add_argument('--core_capacity', type=int, default=LOCAL_DEFAULTS['core_capacity'])
    group.add_argument('--ghost_capacity', type=int, default=LOCAL_DEFAULTS['ghost_capacity'])
    group.add_argument('--group_batch_size', type=int, default=LOCAL_DEFAULTS['group_batch_size'])

    # runtime-args-separation:start
    if include_runtime_args:
        # sdpa-diagnostics:start
        group.add_argument(
            '--print_sdpa_backend',
            dest='print_sdpa_backend',
            type=str,
            default=None,
            metavar='PATH',
        )
        # sdpa-diagnostics:end
        # peak_memory:start
        group.add_argument(
            '--peak_memory',
            type=str,
            default=None,
            metavar='FILE_PATH',
            help='Write peak CUDA training memory usage to this JSON file.',
        )
        # peak_memory:end
        # execution_time:start
        group.add_argument(
            '--execution_time',
            type=str,
            default=None,
            metavar='FILE_PATH',
            help='Write CUDA training step timing statistics to this JSON file.',
        )
        # execution_time:end
    # runtime-args-separation:end
    group.add_argument('--dropout', type=float, default=0.0)
    group.add_argument('--scale_factor', type=float, default=1.0)
    group.add_argument('--interleave_ipa', action='store_true')
    group.add_argument('--prepend_ipa', action='store_true')
    group.add_argument('--oracle', action='store_true')
    group.add_argument('--num_layers', type=int, default=5)
    group.add_argument('--embed_dim', type=int, default=384)
    group.add_argument('--mha_heads', type=int, default=16)
    group.add_argument('--ipa_heads', type=int, default=4)
    # group.add_argument('--ipa_layers', type=int, default=None)
    group.add_argument('--ipa_head_dim', type=int, default=32)
    group.add_argument('--ipa_qk', type=int, default=8)
    group.add_argument('--ipa_v', type=int, default=8)

    group.add_argument('--time_multiplier', type=float, default=100.)
    group.add_argument('--abs_pos_emb', action='store_true')
    group.add_argument('--abs_time_emb', action='store_true')

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="GVP", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--sampling_method", type=str, default="dopri5", choices=["dopri5", "euler"])
    group.add_argument('--alpha_max', type=float, default=8)
    group.add_argument('--discrete_loss_weight', type=float, default=0.5)
    group.add_argument("--dirichlet_flow_temp", type=float, default=1.0)
    group.add_argument('--allow_nan_cfactor', action='store_true')
    # group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    

    ## video settings
    group = parser.add_argument_group("Video settings")
    group.add_argument('--tps_condition', action='store_true')
    group.add_argument('--design', action='store_true')
    group.add_argument('--design_from_traj', action='store_true')
    group.add_argument('--sim_condition', action='store_true')
    group.add_argument('--inpainting', action='store_true')
    group.add_argument('--dynamic_mpnn', action='store_true')
    group.add_argument('--mpnn', action='store_true')
    group.add_argument('--frame_interval', type=int, default=None)
    group.add_argument('--cond_interval', type=int, default=None) # for superresolution
    
    args = parser.parse_args()
    args.precision_config = {
        '32-true': 'fp32-reference', 'bf16-mixed': 'bf16-train',
    }.get(args.precision, args.precision)
    args.precision = {
        'fp32-reference': '32-true', 'bf16-train': 'bf16-mixed',
    }[args.precision_config]
    if args.local_spatial and (args.manual_attention or args.no_frames):
        parser.error('local_spatial requires SDPA and residue frames')
    if args.training_region:
        if not args.local_spatial or args.hyena or args.design or args.abs_pos_emb:
            parser.error('training_region requires --local_spatial and does not support '
                         '--hyena, --design or --abs_pos_emb')
        if args.halo_rings < 0:
            parser.error('--halo_rings must be non-negative')
    check_determinism_args(parser, args)  # deterministic-benchmark-guard
    if args.model_dir is not None:
        os.environ["MODEL_DIR"] = os.path.join(args.model_dir, args.run_name)
    else:
        os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    
    return args
