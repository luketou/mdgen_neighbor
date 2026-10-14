#!/usr/bin/env python3
"""Compare MDGen manual attention and SDPA forward/backward numerics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mdgen.model.mha import MultiheadAttention
from mdgen.model.mha_legacy import MultiheadAttention as LegacyMultiheadAttention  # sdpa-upstream-reference
from mdgen.model.latent_model import AttentionWithRoPE  # sdpa-equivalence-test


OUTPUT_RELATIVE_ERROR_LIMIT = 1e-5
GRADIENT_RELATIVE_ERROR_LIMIT = 1e-4
PADDING_INVARIANCE_ERROR_LIMIT = 1e-6  # sdpa-equivalence-test


# sdpa-equivalence-test
class SDPAEquivalenceTest:
    def __init__(self, output_dir: Path, device: torch.device, seed: int):
        self.output_dir = output_dir
        self.device = device
        self.seed = seed
        self.embed_dim = 384
        self.num_heads = 16

    @staticmethod
    def _relative_max_error(reference: torch.Tensor, candidate: torch.Tensor):
        max_absolute_error = (reference - candidate).abs().max().item()
        reference_scale = reference.abs().max().item()
        if reference_scale == 0.0:
            relative_error = 0.0 if max_absolute_error == 0.0 else float("inf")
        else:
            relative_error = max_absolute_error / reference_scale
        return max_absolute_error, relative_error

    @staticmethod
    def _to_numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def _build_modules(self):
        common_args = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": 0.0,
            "add_bias_kv": True,
            "use_rotary_embeddings": True,
        }
        manual = LegacyMultiheadAttention(**common_args)  # sdpa-upstream-reference
        sdpa = MultiheadAttention(**common_args, use_sdpa=True)
        sdpa.load_state_dict(manual.state_dict())
        manual.eval().to(device=self.device, dtype=torch.float32)
        sdpa.eval().to(device=self.device, dtype=torch.float32)
        return manual, sdpa

    # sdpa-equivalence-test
    def _build_current_modules(self, use_rotary_embeddings=True):
        common_args = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": 0.0,
            "add_bias_kv": True,
            "use_rotary_embeddings": use_rotary_embeddings,
        }
        manual = MultiheadAttention(**common_args, use_sdpa=False)
        sdpa = MultiheadAttention(**common_args, use_sdpa=True)
        sdpa.load_state_dict(manual.state_dict())
        manual.eval().to(device=self.device, dtype=torch.float32)
        sdpa.eval().to(device=self.device, dtype=torch.float32)
        return manual, sdpa

    # sdpa-equivalence-test
    def _build_no_rope_modules(self):
        common_args = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": 0.0,
            "add_bias_kv": True,
            "use_rotary_embeddings": False,
        }
        manual = AttentionWithRoPE(**common_args, use_sdpa=False)
        sdpa = AttentionWithRoPE(**common_args, use_sdpa=True)
        sdpa.load_state_dict(manual.state_dict())
        manual.eval().to(device=self.device, dtype=torch.float32)
        sdpa.eval().to(device=self.device, dtype=torch.float32)
        return manual, sdpa

    def _make_mask(self, batch_size, sequence_length, padded, full_padding_row=False):  # sdpa-time-full-row-padding
        mask = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=self.device,
        )
        if padded:
            mask[0, -3:] = True
            if batch_size > 1:
                mask[1, -1:] = True
        # sdpa-time-full-row-padding:start
        if full_padding_row:
            mask[0, :] = True
        # sdpa-time-full-row-padding:end
        return mask

    def _save_parameter_gradients(self, path: Path, gradients):
        arrays = {
            name.replace(".", "__"): self._to_numpy(gradient)
            for name, gradient in gradients.items()
        }
        np.savez(path, **arrays)

    # sdpa-equivalence-test
    def _test_position_ids_rejected(self):
        sequence_length = 8
        batch_size = 2
        model_input = torch.randn(
            sequence_length,
            batch_size,
            self.embed_dim,
            dtype=torch.float32,
            device=self.device,
        )
        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=self.device,
        ).expand(batch_size, -1)
        rejected = {}
        for rope_name, use_rotary_embeddings in (("rope", True), ("no_rope", False)):
            modules = dict(
                zip(
                    ("manual", "sdpa"),
                    self._build_current_modules(
                        use_rotary_embeddings=use_rotary_embeddings
                    ),
                )
            )
            for backend_name, module in modules.items():
                result_name = f"{rope_name}_{backend_name}"
                try:
                    module(
                        query=model_input,
                        key=model_input,
                        value=model_input,
                        need_weights=False,
                        position_ids=position_ids,
                    )
                except NotImplementedError:
                    rejected[result_name] = True
                else:
                    rejected[result_name] = False
        return {
            "manual_raises": rejected["rope_manual"],
            "sdpa_raises": rejected["rope_sdpa"],
            "no_rope_manual_raises": rejected["no_rope_manual"],
            "no_rope_sdpa_raises": rejected["no_rope_sdpa"],
            "passed": all(rejected.values()),
        }

    # sdpa-equivalence-test
    def _test_no_rope_padding(self):
        sequence_length = 32
        batch_size = 4
        manual, sdpa = self._build_no_rope_modules()
        base_input = torch.randn(
            batch_size,
            sequence_length,
            self.embed_dim,
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.float32,
            device=self.device,
        )
        mask[0, -3:] = 0.0
        mask[1, -1:] = 0.0
        grad_output = torch.randn_like(base_input)

        manual_input = base_input.detach().clone().requires_grad_(True)
        sdpa_input = base_input.detach().clone().requires_grad_(True)
        manual_output = manual(manual_input, mask=mask, position_ids=None)
        sdpa_output = sdpa(sdpa_input, mask=mask, position_ids=None)

        torch.autograd.backward(manual_output, grad_tensors=grad_output)
        torch.autograd.backward(sdpa_output, grad_tensors=grad_output)
        if manual_input.grad is None or sdpa_input.grad is None:
            raise RuntimeError("no_rope_padding: input gradient was not produced")

        manual_parameter_gradients = {
            name: parameter.grad
            for name, parameter in manual.named_parameters()
            if parameter.grad is not None
        }
        sdpa_parameter_gradients = {
            name: parameter.grad
            for name, parameter in sdpa.named_parameters()
            if parameter.grad is not None
        }
        if manual_parameter_gradients.keys() != sdpa_parameter_gradients.keys():
            raise RuntimeError(
                "no_rope_padding: manual and SDPA parameter gradients differ"
            )

        output_max_absolute_error, output_relative_error = self._relative_max_error(
            manual_output, sdpa_output
        )
        input_grad_max_absolute_error, input_grad_relative_error = (
            self._relative_max_error(manual_input.grad, sdpa_input.grad)
        )
        parameter_gradient_relative_error = 0.0
        for parameter_name in manual_parameter_gradients:
            _, relative_error = self._relative_max_error(
                manual_parameter_gradients[parameter_name],
                sdpa_parameter_gradients[parameter_name],
            )
            parameter_gradient_relative_error = max(
                parameter_gradient_relative_error, relative_error
            )
        gradient_relative_error = max(
            input_grad_relative_error, parameter_gradient_relative_error
        )

        padding_positions = ~mask.to(dtype=torch.bool)
        perturbed_input = base_input.detach().clone()
        perturbed_input[padding_positions] += 100.0
        with torch.no_grad():
            manual_perturbed_output = manual(
                perturbed_input, mask=mask, position_ids=None
            )
            sdpa_perturbed_output = sdpa(
                perturbed_input, mask=mask, position_ids=None
            )
        valid_positions = mask.to(dtype=torch.bool)
        manual_padding_invariance_error = (
            manual_output.detach()[valid_positions]
            - manual_perturbed_output[valid_positions]
        ).abs().max().item()
        sdpa_padding_invariance_error = (
            sdpa_output.detach()[valid_positions]
            - sdpa_perturbed_output[valid_positions]
        ).abs().max().item()

        manual_path = manual.attn.last_attention_backend
        sdpa_path = sdpa.attn.last_attention_backend
        passed = (
            manual_path == "torch_mha"
            and sdpa_path == "sdpa"
            and output_relative_error < OUTPUT_RELATIVE_ERROR_LIMIT
            and gradient_relative_error < GRADIENT_RELATIVE_ERROR_LIMIT
            and manual_padding_invariance_error < PADDING_INVARIANCE_ERROR_LIMIT
            and sdpa_padding_invariance_error < PADDING_INVARIANCE_ERROR_LIMIT
        )
        return {
            "sequence_length": sequence_length,
            "effective_batch_size": batch_size,
            "padding": True,
            "use_rotary_embeddings": False,
            "manual_path": manual_path,
            "sdpa_path": sdpa_path,
            "output_max_absolute_error": output_max_absolute_error,
            "output_relative_error": output_relative_error,
            "input_grad_max_absolute_error": input_grad_max_absolute_error,
            "input_grad_relative_error": input_grad_relative_error,
            "parameter_grad_relative_error": parameter_gradient_relative_error,
            "gradient_relative_error": gradient_relative_error,
            "manual_padding_invariance_error": manual_padding_invariance_error,
            "sdpa_padding_invariance_error": sdpa_padding_invariance_error,
            "padding_invariance_error_limit": PADDING_INVARIANCE_ERROR_LIMIT,
            "passed": passed,
        }

    def _run_case(self, name, sequence_length, batch_size, padded, full_padding_row=False):  # sdpa-time-full-row-padding
        case_dir = self.output_dir / name
        case_dir.mkdir()

        manual, sdpa = self._build_modules()
        base_input = torch.randn(
            sequence_length,
            batch_size,
            self.embed_dim,
            dtype=torch.float32,
            device=self.device,
        )
        grad_output = torch.randn_like(base_input)
        key_padding_mask = self._make_mask(
            batch_size, sequence_length, padded, full_padding_row
        )  # sdpa-time-full-row-padding

        manual_input = base_input.detach().clone().requires_grad_(True)
        sdpa_input = base_input.detach().clone().requires_grad_(True)

        manual_output, _ = manual(
            query=manual_input,
            key=manual_input,
            value=manual_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        sdpa_output, _ = sdpa(
            query=sdpa_input,
            key=sdpa_input,
            value=sdpa_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            position_ids=None,
        )

        torch.autograd.backward(manual_output, grad_tensors=grad_output)
        torch.autograd.backward(sdpa_output, grad_tensors=grad_output)

        if manual_input.grad is None or sdpa_input.grad is None:
            raise RuntimeError(f"{name}: input gradient was not produced")

        manual_parameter_gradients = {
            parameter_name: parameter.grad
            for parameter_name, parameter in manual.named_parameters()
            if parameter.grad is not None
        }
        sdpa_parameter_gradients = {
            parameter_name: parameter.grad
            for parameter_name, parameter in sdpa.named_parameters()
            if parameter.grad is not None
        }
        if manual_parameter_gradients.keys() != sdpa_parameter_gradients.keys():
            raise RuntimeError(f"{name}: manual and SDPA parameter gradients differ")

        output_max_absolute_error, output_relative_error = self._relative_max_error(
            manual_output, sdpa_output
        )
        input_grad_max_absolute_error, input_grad_relative_error = (
            self._relative_max_error(manual_input.grad, sdpa_input.grad)
        )

        parameter_gradient_errors = {}
        parameter_gradient_relative_error = 0.0
        for parameter_name in manual_parameter_gradients:
            max_absolute_error, relative_error = self._relative_max_error(
                manual_parameter_gradients[parameter_name],
                sdpa_parameter_gradients[parameter_name],
            )
            parameter_gradient_errors[parameter_name] = {
                "max_absolute_error": max_absolute_error,
                "relative_error": relative_error,
            }
            parameter_gradient_relative_error = max(
                parameter_gradient_relative_error, relative_error
            )

        np.save(case_dir / "manual_output.npy", self._to_numpy(manual_output))
        np.save(case_dir / "sdpa_output.npy", self._to_numpy(sdpa_output))
        np.save(
            case_dir / "manual_input_grad.npy", self._to_numpy(manual_input.grad)
        )
        np.save(case_dir / "sdpa_input_grad.npy", self._to_numpy(sdpa_input.grad))
        self._save_parameter_gradients(
            case_dir / "manual_parameter_grads.npz", manual_parameter_gradients
        )
        self._save_parameter_gradients(
            case_dir / "sdpa_parameter_grads.npz", sdpa_parameter_gradients
        )

        gradient_relative_error = max(
            input_grad_relative_error, parameter_gradient_relative_error
        )
        # sdpa-time-full-row-padding:start
        outputs_finite = bool(
            torch.isfinite(manual_output).all().item()
            and torch.isfinite(sdpa_output).all().item()
        )
        gradients_finite = bool(
            torch.isfinite(manual_input.grad).all().item()
            and torch.isfinite(sdpa_input.grad).all().item()
            and all(
                torch.isfinite(gradient).all().item()
                for gradient in manual_parameter_gradients.values()
            )
            and all(
                torch.isfinite(gradient).all().item()
                for gradient in sdpa_parameter_gradients.values()
            )
        )
        # sdpa-time-full-row-padding:end
        passed = (
            sdpa.last_attention_backend == "sdpa"
            and outputs_finite  # sdpa-time-full-row-padding
            and gradients_finite  # sdpa-time-full-row-padding
            and output_relative_error < OUTPUT_RELATIVE_ERROR_LIMIT
            and gradient_relative_error < GRADIENT_RELATIVE_ERROR_LIMIT
        )
        result = {
            "sequence_length": sequence_length,
            "effective_batch_size": batch_size,
            "padding": padded,
            "full_padding_row": full_padding_row,  # sdpa-time-full-row-padding
            "position_ids": None,
            "manual_path": "upstream_manual",  # sdpa-upstream-reference
            "sdpa_path": sdpa.last_attention_backend,
            "sdpa_fallback_reason": sdpa.last_sdpa_fallback_reason,
            "output_max_absolute_error": output_max_absolute_error,
            "output_relative_error": output_relative_error,
            "input_grad_max_absolute_error": input_grad_max_absolute_error,
            "input_grad_relative_error": input_grad_relative_error,
            "parameter_grad_relative_error": parameter_gradient_relative_error,
            "gradient_relative_error": gradient_relative_error,
            "outputs_finite": outputs_finite,  # sdpa-time-full-row-padding
            "gradients_finite": gradients_finite,  # sdpa-time-full-row-padding
            "parameter_gradient_errors": parameter_gradient_errors,
            "passed": passed,
        }
        with (case_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return result

    def run(self):
        position_ids_non_none = self._test_position_ids_rejected()  # sdpa-equivalence-test
        # sdpa-time-full-row-padding:start
        cases = (
            ("residue_no_padding", 32, 4, False, False),
            ("residue_padding", 32, 4, True, False),
            ("time_no_padding", 16, 8, False, False),
            ("time_padding", 16, 8, True, False),
            ("time_full_row_padding", 16, 8, True, True),
        )
        # sdpa-time-full-row-padding:end
        results = {}
        for case_index, case in enumerate(cases):
            torch.manual_seed(self.seed + case_index)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed + case_index)
            results[case[0]] = self._run_case(*case)

        # sdpa-equivalence-test:start
        no_rope_seed = self.seed + len(cases)
        torch.manual_seed(no_rope_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(no_rope_seed)
        no_rope_padding = self._test_no_rope_padding()
        # sdpa-equivalence-test:end

        summary = {
            "seed": self.seed,
            "device": str(self.device),
            "dtype": "float32",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": 0.0,
            "add_bias_kv": True,
            "use_rotary_embeddings": True,
            "position_ids": None,
            # sdpa-equivalence-test:start
            "position_ids_non_none": position_ids_non_none,
            "no_rope_padding": no_rope_padding,
            # sdpa-equivalence-test:end
            "tf32_cuda_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "output_relative_error_limit": OUTPUT_RELATIVE_ERROR_LIMIT,
            "gradient_relative_error_limit": GRADIENT_RELATIVE_ERROR_LIMIT,
            "cases": results,
            # sdpa-equivalence-test:start
            "passed": (
                position_ids_non_none["passed"]
                and no_rope_padding["passed"]
                and all(result["passed"] for result in results.values())
            ),
            # sdpa-equivalence-test:end
        }
        with (self.output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary


# sdpa-equivalence-test
def main():
    parser = argparse.ArgumentParser(
        description="Compare MDGen manual attention and SDPA numerics."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"--output_dir must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu only for a smoke test")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    test = SDPAEquivalenceTest(args.output_dir, device, args.seed)
    summary = test.run()
    for case_name, result in summary["cases"].items():
        print(
            f"{case_name}: output={result['output_relative_error']:.3e}, "
            f"gradient={result['gradient_relative_error']:.3e}, "
            f"path={result['sdpa_path']}, pass={result['passed']}"
        )
    # sdpa-equivalence-test:start
    no_rope_padding = summary["no_rope_padding"]
    print(
        "no_rope_padding: "
        f"output={no_rope_padding['output_relative_error']:.3e}, "
        f"gradient={no_rope_padding['gradient_relative_error']:.3e}, "
        f"manual_padding={no_rope_padding['manual_padding_invariance_error']:.3e}, "
        f"sdpa_padding={no_rope_padding['sdpa_padding_invariance_error']:.3e}, "
        f"pass={no_rope_padding['passed']}"
    )
    # sdpa-equivalence-test:end
    print(f"summary: {args.output_dir / 'summary.json'}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
