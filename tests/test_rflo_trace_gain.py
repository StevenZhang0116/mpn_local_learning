"""Capped RFLO recurrence: scalar oracles, gates, legacy behavior, and persistence."""
import contextlib
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import _bootstrap
import mpn
import train_mpn
import train_common
from test_input_modulation_trace import make_net, data
from validate_local_learning import build_net as make_single_net, diag_rflo_reference

sys.path.insert(0, str(_bootstrap.ROOT / "notebooks"))
import diagnose_gradients as diagnostic


class TestRFLOTraceGain(unittest.TestCase):
    def build(self, rho, **kwargs):
        net, cfg = make_net(input_mode="match", **kwargs)
        cfg["ml_params"]["rflo_trace_rho"] = rho
        with contextlib.redirect_stdout(io.StringIO()):
            capped = mpn.DeepMultiPlasticNet(cfg, verbose=False).double()
        capped.load_state_dict(net.state_dict())
        return capped, cfg

    @torch.no_grad()
    def test_disabled_matches_original_diagonal_reference(self):
        # Independent reference advances the full P tensor, zeroing off-diagonals.
        net = make_single_net(2, 3, 2, "tanh", True, True, "scalar", "scalar",
                              learning_rule="local_diag_rflo")
        self.assertIsNone(net.mp_layer.rflo_trace_rho)
        x, y, mask, _ = data()
        reference = diag_rflo_reference(net, x, y, mask)
        actual = net.local_diag_rflo_gradients(x, y, mask)
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=1e-10, atol=1e-12)

    @torch.no_grad()
    def test_cap_inactive_agrees_with_uncapped_rule(self):
        capped, _ = self.build(.99)
        for layer in capped.mp_layers:
            layer.eta.fill_(.001)
            layer.lam.fill_(.5)
        baseline = deepcopy(capped)
        for layer in baseline.mp_layers:
            layer.rflo_trace_rho = None
        batch = data()[:3]
        actual = diagnostic.diagnose(capped, batch)
        expected = diagnostic.diagnose(baseline, batch)
        self.assertTrue(all(r["trace_gain_clipped_fraction"] == 0 for r in actual["trace_rows"]))
        for key, value in expected["gradients"]["local_diag_rflo"].items():
            torch.testing.assert_close(actual["gradients"]["local_diag_rflo"][key], value,
                                       rtol=1e-10, atol=1e-12)

    @torch.no_grad()
    def test_constant_drive_is_bounded_instead_of_exponential(self):
        # With fixed M=0, x=phi'=eta=W=1 and lambda=.99:
        # uncapped A_t=1.99*A_(t-1)+1; capped A_t=.95*A_(t-1)+1.
        net, _ = self.build(.95, embed=1, widths=(1,))
        layer = net.mp_layers[0]
        layer.W.fill_(1)
        layer.M_init.zero_()
        layer.reset_state(B=1)
        layer.reset_diag_rflo_state(B=1)
        baseline = deepcopy(layer)
        baseline.rflo_trace_rho = None
        x = phi = ell = torch.ones(1, 1, dtype=torch.double)
        eta = torch.ones(1, 1, dtype=torch.double)
        lam = torch.full_like(eta, .99)
        for _ in range(137):
            for current in (layer, baseline):
                current._local_step_diag(x, phi, ell, eta, lam)
                # Zero postsynaptic drive keeps M fixed, isolating trace recurrence.
                current.update_M_matrix_local_fast(x, torch.zeros_like(phi), eta=eta, lam=lam)
        self.assertAlmostEqual(layer.A.item(), (1 - .95**137) / (1 - .95), places=11)
        self.assertLessEqual(layer.A.item(), 20)
        self.assertGreater(baseline.A.item(), 1e35)

    @torch.no_grad()
    def test_fused_explicit_and_scalar_oracle_with_masks_and_write_gates(self):
        for write_mode in ("linear", "hard", "smooth"):
            for bias in ("direct", "exact"):
                with self.subTest(write_mode=write_mode, bias=bias):
                    net, _ = self.build(.4, embed=2, widths=(2,), write_mode=write_mode)
                    layer = net.mp_layers[0]
                    layer.local_bias_mode = bias
                    layer.W.copy_(torch.tensor([[2., -3.], [.1, -.2]]))
                    layer.M_init.zero_()
                    layer.set_plasticity_freeze(torch.tensor([1]), torch.tensor([1]))
                    layer.reset_state(B=3)
                    layer.reset_diag_rflo_state(B=3)
                    layer.A.fill_(.25)
                    layer.A[:, 1, 1] = 0
                    # Force a zero eligibility factor to catch unsafe E/factor division.
                    layer.A[0, 0, 1] = 1 / 3
                    explicit = deepcopy(layer)
                    recorder = diagnostic.TraceRecorder(net)
                    x = torch.tensor([[0., 1.], [1., -2.], [.5, 2.]], dtype=torch.double)
                    phi = torch.tensor([[1., .2], [.6, 1.], [.8, .5]], dtype=torch.double)
                    post = torch.tensor([[.8, -.5], [.4, .9], [1., -.7]], dtype=torch.double)
                    ell = torch.tensor([[.1, .2], [-.4, .3], [.6, -.1]], dtype=torch.double)
                    eta = torch.tensor([[1., .7], [.4, 1.2]], dtype=torch.double)
                    lam = torch.tensor([[.8, .7], [.6, .5]], dtype=torch.double)
                    active = torch.tensor([0., .25, 1.], dtype=torch.double)
                    expected_A = torch.zeros_like(layer.A)
                    expected_M = torch.zeros_like(layer.M)
                    expected_g = torch.zeros_like(layer.W)
                    clipped_gains = 0
                    # Scalar reference applies the mathematical recurrence and gates
                    # independently of production broadcasting and finalization helpers.
                    for b in range(3):
                        r = active[b].item()
                        for i in range(2):
                            for j in range(2):
                                w, a, m = layer.W[i, j].item(), layer.A[b, i, j].item(), layer.M[b, i, j].item()
                                xx, pp = x[b, j].item(), phi[b, i].item()
                                e, l = eta[i, j].item(), lam[i, j].item()
                                lower, upper = ((layer.M_bounds[1, i, j].item(), layer.M_bounds[0, i, j].item())
                                                if write_mode == "hard" else (-math.inf, math.inf))
                                expected_g[i, j] += ell[b, i].item() * pp * xx * (1 + m + w*a)
                                k = e * pp * xx**2
                                clipped_gains += abs(l + k*w) > .4
                                candidate = max(-.4, min(.4, l + k*w)) * a + k*(1+m)
                                raw = l*m + e*post[b, i].item()*xx
                                if write_mode == "smooth":
                                    expected_A[b, i, j] = r*(1-math.tanh(raw/.4)**2)*candidate + (1-r)*a
                                    expected_M[b, i, j] = r*.4*math.tanh(raw/.4) + (1-r)*m
                                else:
                                    raw = r*raw + (1-r)*m
                                    gate = float(lower <= raw <= upper) if write_mode == "hard" else 1.
                                    expected_A[b, i, j] = gate*(r*candidate + (1-r)*a)
                                    expected_M[b, i, j] = max(lower, min(upper, raw)) if write_mode == "hard" else raw
                    expected_A[:, 1, 1] = expected_M[:, 1, 1] = 0
                    grad_w, grad_b = layer._local_step_diag(x, phi, ell, eta, lam, active)
                    E, R = explicit.compute_diag_rflo_eligibility(x, phi)
                    explicit.update_diag_rflo_traces(x, E, R, active, (eta, lam))
                    for current in (layer, explicit):
                        current.update_M_matrix_local_fast(x, post, eta=eta, lam=lam, update_mask=active)
                        torch.testing.assert_close(current.A, expected_A, rtol=1e-12, atol=1e-12)
                        torch.testing.assert_close(current.M, expected_M, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(grad_w, expected_g, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(grad_w, torch.einsum("Bi,BiI->iI", ell, E))
                    torch.testing.assert_close(grad_b, (ell*R).sum(0))
                    self.assertEqual(recorder.rows[0]["trace_gain_clipped_fraction"],
                                     clipped_gains / expected_A.numel())
                    if bias == "exact":
                        torch.testing.assert_close(layer.Q, explicit.Q)

    def test_only_rflo_mp_weight_gradients_change(self):
        capped, _ = self.build(.2)
        baseline = deepcopy(capped)
        for layer in baseline.mp_layers:
            layer.rflo_trace_rho = None
        batch = data()[:3]
        for rule in ("bptt", "local_direct", "local_exact_rowlocal", "local_diag_rflo"):
            with self.subTest(rule=rule):
                capped.learning_rule = baseline.learning_rule = rule
                actual = capped.sequence_gradients(*batch)
                expected = baseline.sequence_gradients(*batch)
                torch.testing.assert_close(actual["outputs"], expected["outputs"], rtol=0, atol=0)
                self.assertEqual(float(actual["loss"]), float(expected["loss"]))
                keys = list(capped._trainable_params())
                unchanged = keys if rule != "local_diag_rflo" else [k for k in keys if k not in ("W", "W1")]
                for key in unchanged:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                if rule == "local_diag_rflo":
                    self.assertFalse(torch.allclose(actual["W"], expected["W"]))
        capped.cross_layer_steps = 1  # exercise the explicit path inside a deep unroll
        result = capped.sequence_gradients(*batch)
        self.assertTrue(all(torch.isfinite(result[k]).all() for k in capped._trainable_params()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device agreement")
    def test_cuda_agreement_and_optimizer_step(self):
        for write_mode in ("hard", "smooth"):
            for cross in (0, 1):
                with self.subTest(write_mode=write_mode, cross=cross):
                    cpu, _ = self.build(.4, write_mode=write_mode)
                    cpu.cross_layer_steps = cross
                    gpu = deepcopy(cpu).cuda()
                    x, y, mask, active = data()
                    expected = cpu.sequence_gradients(x, y, mask, update_masks=active)
                    actual = gpu.sequence_gradients(x.cuda(), y.cuda(), mask.cuda(), update_masks=active.cuda())
                    for key in [*cpu._trainable_params(), "outputs"]:
                        torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=1e-8, atol=1e-10)
                    before = gpu.mp_layers[0].W.detach().clone()
                    optimizer = torch.optim.Adam(gpu.parameters(), lr=1e-3)
                    optimizer.step()
                    self.assertFalse(torch.equal(before, gpu.mp_layers[0].W))
                    self.assertTrue(all(torch.isfinite(p).all() for p in gpu.parameters()))

    @torch.no_grad()
    def test_zero_plasticity_and_pre_only_remain_direct(self):
        for kind in ("hebb_assoc", "hebb_pre"):
            net, _ = self.build(.4, kind=kind)
            if kind == "hebb_assoc":
                for layer in net.mp_layers:
                    layer.eta.zero_()
            batch = data()[:3]
            expected = net.local_direct_gradients(*batch)
            actual = net.local_diag_rflo_gradients(*batch)
            for key in net._trainable_params():
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)

    def test_cli_validation_metadata_and_checkpoint_reload(self):
        for value in (0, 1, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "rflo_trace_rho"):
                    self.build(value)
                with patch.object(sys, "argv", ["train_mpn.py", "--rflo-trace-rho", str(value)]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        train_mpn._parse_args()
        with patch.object(sys, "argv", ["train_mpn.py", "--rflo-trace-rho", ".95"]):
            self.assertEqual(train_mpn._parse_args().rflo_trace_rho, .95)
        with patch.object(sys, "argv", ["train_mpn.py"]):
            self.assertIsNone(train_mpn._parse_args().rflo_trace_rho)
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(train_mpn, RFLO_TRACE_RHO=.95, N_HIDDEN=[2, 2], RULESET="delaygo"):
                cfg = train_mpn._cfg()
                config_path = Path(directory) / "config.json"
                train_common.save_config(cfg, str(config_path))
            saved = json.loads(config_path.read_text())
            self.assertEqual(saved["net_params"]["ml_params"]["rflo_trace_rho"], .95)
            net, params = self.build(.95)
            path = Path(directory) / "checkpoint.pt"
            torch.save(dict(net_params=params, state_dict=net.state_dict(), learning_rule="local_diag_rflo"), path)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = train_mpn.load_net(path, device=torch.device("cpu"), dtype=torch.double)
            self.assertTrue(all(layer.rflo_trace_rho == .95 for layer in loaded.mp_layers))
            expected = net.sequence_gradients(*data()[:3])
            actual = loaded.sequence_gradients(*data()[:3])
            for key in net._trainable_params():
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            checkpoint = torch.load(path, weights_only=False)
            rebuilt, settings = diagnostic.build_model(checkpoint, torch.device("cpu"), torch.double)
            self.assertEqual(settings["rflo_trace_rho"], .95)
            report = diagnostic.diagnose(rebuilt, data()[:3])
            self.assertGreater(max(r["trace_gain_clipped_fraction"] for r in report["trace_rows"]), 0)
            del checkpoint["net_params"]["ml_params"]["rflo_trace_rho"]
            legacy, settings = diagnostic.build_model(checkpoint, torch.device("cpu"), torch.double)
            self.assertIsNone(settings["rflo_trace_rho"])
            self.assertTrue(all(layer.rflo_trace_rho is None for layer in legacy.mp_layers))


if __name__ == "__main__":
    unittest.main()
