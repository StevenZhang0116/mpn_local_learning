"""Validate scaling statistics and non-invasive sample/time gradient recording."""
from copy import deepcopy
import sys
import unittest

import torch

import _bootstrap
from test_input_modulation_trace import make_net, data

sys.path.insert(0, str(_bootstrap.ROOT / "notebooks"))
import diagnose_gradients as dg
import verify_rflo_scaling as scaling


class TestRFLOScaling(unittest.TestCase):
    def test_scalar_fit_and_cancellation_counterexample(self):
        d = torch.tensor([1., -2., 3.], dtype=torch.double)
        fit = scaling.scaling_fit(d, 2*d)
        self.assertAlmostEqual(fit["gain"], 2.)
        self.assertAlmostEqual(fit["residual"], 0.)
        self.assertAlmostEqual(fit["parallel_correction_energy_fraction"], 1.)
        orthogonal = scaling.scaling_fit(torch.tensor([1., 0.]), torch.tensor([1., .1]))
        self.assertGreater(orthogonal["cosine"], .99)
        self.assertAlmostEqual(orthogonal["parallel_correction_energy_fraction"], 0.)
        fit = scaling.scaling_fit(d, -d)
        self.assertAlmostEqual(fit["gain"], -1.)
        self.assertAlmostEqual(fit["residual"], 0.)
        self.assertIsNone(scaling.scaling_fit(torch.zeros(3), d)["gain"])
        u = torch.tensor([1., -.99], dtype=torch.double)
        f = torch.tensor([.98, 1.02], dtype=torch.double)
        self.assertGreater(float(u.sum()), 0.)
        self.assertAlmostEqual(float((u*f).sum()), -.0298)

    def test_diagonal_generalizes_only_when_stable(self):
        torch.manual_seed(6)
        d = torch.randn(8, 6, dtype=torch.double)
        gains = torch.tensor([.5, 1., 2., 3., 4., 6.], dtype=torch.double)
        r = d*gains
        fit = scaling.heldout_scaling(d, r)
        self.assertLess(fit["positive_diagonal_test_error"], 1e-14)
        self.assertGreater(fit["scalar_test_error"], .1)
        # Identical fit set, incompatible held-out gains: cannot hide this by refitting.
        r[4:] *= -1
        changed = scaling.heldout_scaling(d, r)
        self.assertAlmostEqual(changed["positive_diagonal_test_error"], 2.)
        self.assertEqual(changed["scalar_gain"], fit["scalar_gain"])
        d[:, 0] = 0
        r[:, 0] = 1
        fit = scaling.heldout_scaling(d, r)
        self.assertGreater(fit["positive_diagonal_test_error"], 0)
        self.assertAlmostEqual(fit["unsupported_diagonal_fraction"], 1/6)

    def test_virtual_adam_matches_torch_and_fixed_positive_scaling(self):
        torch.manual_seed(4)
        grads = torch.randn(8, 7, dtype=torch.double)
        parameter = torch.nn.Parameter(torch.zeros(7, dtype=torch.double))
        optimizer = torch.optim.Adam([parameter], lr=.001)
        expected = []
        for grad in grads:
            parameter.grad = grad.clone()
            before = parameter.detach().clone()
            optimizer.step()
            expected.append((before-parameter.detach())/.001)
        torch.testing.assert_close(scaling.adam_directions(grads), torch.stack(expected),
                                   rtol=1e-12, atol=1e-12)
        positive = torch.arange(1., 8., dtype=torch.double)
        torch.testing.assert_close(scaling.adam_directions(grads, eps=0),
                                   scaling.adam_directions(grads*positive, eps=0),
                                   rtol=1e-12, atol=1e-12)

    def test_recorder_reconstructs_gradients_and_preserves_production(self):
        for mode in ("linear", "hard", "smooth"):
            with self.subTest(mode=mode):
                net, _ = make_net(widths=(3,), input_mode="match", write_mode=mode)
                batch = data()[:3]
                plain = dg.diagnose(net, batch)
                observers = []
                def observe(model):
                    recorder = scaling.ScalingRecorder(model)
                    observers.append(recorder)
                    return recorder
                observed = dg.diagnose(net, batch, recorder_factory=observe)
                for rule in dg.RULES:
                    for key in plain["gradients"][rule]:
                        torch.testing.assert_close(observed["gradients"][rule][key],
                                                   plain["gradients"][rule][key], rtol=0, atol=0)
                metrics, dt, rt = observers[0].finish(observed["gradients"])
                self.assertLess(metrics["reconstruction_direct_error"], 1e-12)
                self.assertLess(metrics["reconstruction_rflo_error"], 1e-12)
                torch.testing.assert_close(dt[0], rt[0], rtol=0, atol=0)
                # At one MP layer, full row-local MP-weight eligibility is exact.
                exact = deepcopy(net).local_gradients(*batch)
                torch.testing.assert_close(exact["W"], observed["gradients"]["bptt"]["W"],
                                           rtol=1e-10, atol=1e-12)

    def test_zero_base_nonzero_rflo_contribution_is_retained(self):
        net, _ = make_net(widths=(3,), input_mode="match")
        layer = net.mp_layers[0]
        layer.reset_diag_rflo_state(B=2)
        layer.reset_state(B=2)
        layer.M.fill_(-1.)
        layer.A.fill_(1.)
        recorder = scaling.ScalingRecorder(net)
        x = torch.ones(2, 3, dtype=torch.double)
        phi = torch.ones(2, 3, dtype=torch.double)
        eta, lam = layer._eta_lam_full()
        grad_w, _ = layer._local_step_diag(x, phi, phi, eta, lam)
        zero = torch.zeros_like(grad_w)
        metrics, _, _ = recorder.finish({"local_direct": {"W": zero}, "local_diag_rflo": {"W": grad_w}})
        self.assertAlmostEqual(metrics["zero_base_rflo_energy_fraction"], 1.)
        self.assertAlmostEqual(metrics["contribution_diagonal_residual"], 1.)


if __name__ == "__main__":
    unittest.main()
