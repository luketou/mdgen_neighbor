"""Final structures and work preservation at the agreed inference seam."""
import copy
from pathlib import Path
import sys

import pytest
import torch

from mdgen.wrapper import NewMDGenWrapper
sys.path.insert(0, str(Path(__file__).parent / 'reference'))
from wrapper_before_ode_endpoints import FiftyPointWrapper
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/diagnostics'))
from scale_sdpa import fixture, move

ROOT = Path(__file__).parents[1]


def small_wrapper():
    checkpoint = torch.load(Path(__import__('os').environ.get('MDGEN_RESOURCE_ROOT', str(ROOT.parent/'mdgen')))/'ckpt/atlas.ckpt', map_location='cpu', weights_only=False)
    config = copy.deepcopy(checkpoint['hyper_parameters']['args'])
    del checkpoint
    config.embed_dim, config.num_layers = 32, 1
    config.prepend_ipa, config.interleave_ipa = False, False
    config.local_spatial, config.temporal_sdpa, config.cell_summaries = True, True, False
    config.num_frames, config.ffn_chunk_size = 250, 65536
    config.sampling_method, config.dropout = 'dopri5', 0.
    torch.manual_seed(151)
    wrapper = NewMDGenWrapper(config).cuda().float().eval()
    with torch.no_grad():
        for name, parameter in wrapper.model.named_parameters():
            if 'cell_summaries' not in name:
                parameter.normal_(0, .025)
    return wrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Inference memory budget requires CUDA')
def test_inference_preserves_final_structure_and_work_with_less_saved_ode_workspace():
    wrapper = small_wrapper()
    batch, _ = fixture(16)
    batch = move(batch)
    evaluations = [0]
    forward = wrapper.model.forward_inference
    def counted(*args, **kwargs):
        evaluations[0] += 1
        return forward(*args, **kwargs)
    # Measurement only: the real model and solver execute without stubs.
    wrapper.model.forward_inference = counted
    sampler = wrapper.transport_sampler.sample_ode
    baseline = [0]
    def measured_sampler(*args, **kwargs):
        sample = sampler(*args, **kwargs)
        def measured(*args, **kwargs):
            # Isolate ODE/reconstruction workspace from the small fixture's
            # dominating CUDA quaternion preprocessing workspace.
            baseline[0] = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            return sample(*args, **kwargs)
        return measured
    wrapper.transport_sampler.sample_ode = measured_sampler
    def run(inference):
        torch.manual_seed(137); torch.cuda.manual_seed_all(137)
        evaluations[0] = 0
        output = inference(wrapper, batch)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline[0]
        return output, evaluations[0], peak
    with torch.no_grad():
        # Warm both paths before measuring; retain no warmup results.
        for inference in (FiftyPointWrapper.inference, NewMDGenWrapper.inference):
            result, _, _ = run(inference)
            del result
        expected, old_nfe, old_peak = run(FiftyPointWrapper.inference)
        expected = tuple(v.cpu() for v in expected)
        actual, new_nfe, new_peak = run(NewMDGenWrapper.inference)
        assert actual[0].shape == (1, 250, 16, 14, 3)
        assert torch.isfinite(actual[0]).all()
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(actual[1].cpu(), expected[1], atol=0, rtol=0)
        assert new_nfe == old_nfe and old_nfe > 2
    print(f'ODE_STORAGE_PEAK old={old_peak} endpoint={new_peak} NFE={old_nfe}/{new_nfe}', flush=True)
    # A 50-point solution allocates 48 more state tensors than endpoints.
    # Require a meaningful workspace decrease, allowing other peak allocations.
    state_bytes = 1 * 250 * 16 * wrapper.latent_dim * 4
    assert old_peak - new_peak > 30 * state_bytes, (old_peak, new_peak)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_dopri5_saved_points_preserve_final_solution_and_model_evaluations(dtype):
    from types import SimpleNamespace
    from mdgen.transport.transport import Sampler, create_transport
    transport = create_transport(SimpleNamespace(design=False), 'Linear', 'velocity', None)
    initial = torch.linspace(-1, 1, 42, dtype=dtype).reshape(1, 2, 3, 7)
    results, evaluations = [], []
    for points in (50, 2):
        count = [0]
        def velocity(value, time):
            count[0] += 1
            return -.3 * value
        result = Sampler(transport).sample_ode(
            sampling_method='dopri5', num_steps=points, atol=1e-6, rtol=1e-3)(initial, velocity)
        assert len(result) == points
        torch.testing.assert_close(result[0], initial, atol=0, rtol=0)
        results.append(result[-1]); evaluations.append(count[0])
    # Independent closed-form solution of dx/dt = -.3x.
    import math
    expected = initial * math.exp(-.3)
    torch.testing.assert_close(results[1], expected, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(results[1], results[0], atol=2e-5, rtol=1e-4)
    assert evaluations[0] == evaluations[1] and evaluations[0] > 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Wrapper comparison requires CUDA')
@pytest.mark.parametrize('training,grad_enabled,method', [
    (True,False,'dopri5'), (False,True,'dopri5'), (False,False,'rk4')],
    ids=['training','grad-enabled','fixed-step'])
def test_other_inference_modes_retain_fifty_point_solver_behavior(training, grad_enabled, method):
    wrapper = small_wrapper()
    wrapper.train(training)
    wrapper.args.sampling_method = method
    batch, _ = fixture(16)
    batch = move({k: (v[:,:4].contiguous() if k in ('trans','rots','torsions') else v)
                  for k,v in batch.items()})
    sampler = wrapper.transport_sampler.sample_ode
    points, evaluations = [], [0]
    def observed_sampler(*args, **kwargs):
        sample = sampler(*args, **kwargs)
        def observed(*args, **kwargs):
            result = sample(*args, **kwargs)
            points.append(len(result))
            return result
        return observed
    wrapper.transport_sampler.sample_ode = observed_sampler
    forward = wrapper.model.forward_inference
    def counted(*args, **kwargs):
        evaluations[0] += 1
        return forward(*args, **kwargs)
    wrapper.model.forward_inference = counted
    outputs, counts = [], []
    with torch.set_grad_enabled(grad_enabled):
        for inference in (FiftyPointWrapper.inference, NewMDGenWrapper.inference):
            torch.manual_seed(137); torch.cuda.manual_seed_all(137)
            evaluations[0] = 0
            result = inference(wrapper, batch)
            assert result[0].requires_grad == grad_enabled
            outputs.append(tuple(v.detach().cpu() for v in result));counts.append(evaluations[0])
            del result
    assert points == [50,50]
    assert counts[0] == counts[1]
    torch.testing.assert_close(outputs[1][0], outputs[0][0], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(outputs[1][1], outputs[0][1], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Large CUDA conditioning requires GPU')
def test_large_conditioning_batch_generates_finite_frames():
    wrapper = small_wrapper()
    batch, _ = fixture(128)
    with torch.no_grad():
        output, sequence = wrapper.inference(move(batch))
    assert output.shape == (1,250,128,14,3)
    assert torch.isfinite(output).all()
    assert sequence.shape == (1,250,128)
