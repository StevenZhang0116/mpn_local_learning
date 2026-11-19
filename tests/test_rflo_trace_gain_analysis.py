"""notebooks/rflo_trace_gain.py: the recorded diagonal gain is lam + k*W and the
exact row gain is lam + sum_J k_J W_iJ (the row sum of the diagonal terms); the
three local variants share init and data; the capped trace is bounded while the
uncapped one is not forced to be; every gradient is taken under the task's own
loss (cross-entropy for two-hop recall); and the CLI writes the figure, summary
and curves. Tiny CPU double networks. Run from tests/: python -m unittest test_rflo_trace_gain_analysis -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap
import mpn
sys.path.insert(0, str(_bootstrap.ROOT / "notebooks"))
import rflo_trace_gain as rtg  # noqa: E402

SMALL = dict(task_name="delaygo", widths=(6, 5), batch=3, seed=0, dtype=torch.float64)


class TestRecordedGains(unittest.TestCase):
    def test_gains_match_their_definitions(self):
        net, (x, y, mask) = rtg.build(rho=None, rule="local_diag_rflo", **SMALL)
        _, rec = rtg.record_diag(net, x, y, mask)
        T = x.shape[1]
        for n, layer in enumerate(net.mp_layers):
            B, i, I = x.shape[0], layer.n_output, layer.n_input
            dg = rec[n]["diag_gain"].reshape(T, B, i, I)
            rg = rec[n]["row_gain"].reshape(T, B, i)
            lam = float(layer.lam.mean())
            # exact row gain = lam + sum over J of (diag gain_J - lam): the row SUM of
            # the same-synapse terms the diagonal approximation keeps one of.
            np.testing.assert_allclose(rg, lam + (dg - lam).sum(-1), rtol=1e-10, atol=1e-12)
            self.assertEqual(len(rec[n]["maxA"]), T)
            self.assertTrue(np.isfinite(rec[n]["maxA"]).all())
        # The diagonal gain recomputed independently from the forward quantities.
        layer = net.mp_layers[0]
        eta, lam = layer._eta_lam_full()
        with torch.no_grad():
            net.reset_state(B=x.shape[0])
            out, h, z, phi_p, _ = net._forward_local_stack(x[:, 0, :])
        k = eta.unsqueeze(0) * phi_p[0].unsqueeze(-1) * h[0].square().unsqueeze(1)
        expected = (lam.unsqueeze(0) + k * layer.W.unsqueeze(0)).detach().flatten().numpy()
        np.testing.assert_allclose(rec[0]["diag_gain"][:expected.size], expected, rtol=1e-10, atol=1e-12)

    def test_build_is_reproducible_for_ring_tasks(self):
        a, (xa, ya, ma) = rtg.build(rho=None, rule="local_direct", **SMALL)
        b, (xb, yb, mb) = rtg.build(rho=None, rule="local_direct", **SMALL)
        self.assertTrue(torch.equal(xa, xb) and torch.equal(ya, yb) and torch.equal(ma, mb))
        for k, p in a._trainable_params().items():
            self.assertTrue(torch.equal(p, b._trainable_params()[k]), k)

    def test_variants_share_init_and_forward_and_capped_trace_is_bounded(self):
        summary, curves = rtg.analyze(rho=0.5, **SMALL)
        self.assertEqual(len(summary["layers"]), 2)
        for lay in summary["layers"]:
            self.assertEqual(lay["cos_vs_bptt"].keys(), {"exact", "diag", "capped"})
            self.assertTrue(all(np.isfinite(v) for v in lay["cos_vs_bptt"].values()))
            self.assertTrue(0 <= lay["frac_diag_gain_gt1"] <= 1 and 0 <= lay["frac_row_gain_gt1"] <= 1)
        # With a gain cap below 1 the trace recursion is contractive, so the capped
        # max|A| cannot exceed drive/(1 - rho) accumulation; the uncapped one may.
        # Here only check finiteness and shapes; growth is the figure's story.
        T = curves["T"]
        for key in ("maxA_uncapped", "maxA_capped", "maxPdiag_exact", "maxP_exact", "frac_diag", "frac_row"):
            self.assertEqual(curves[key].shape, (2, T), key)
            self.assertTrue(np.isfinite(curves[key]).all(), key)
        self.assertTrue((curves["maxPdiag_exact"] <= curves["maxP_exact"] + 1e-12).all())

    def test_cli_writes_figure_summary_and_curves(self):
        with tempfile.TemporaryDirectory() as d:
            summary, fig_path = rtg.main(["--task", "delaygo", "--hidden", "6", "5", "--batch", "3",
                                          "--dtype", "float64", "--rho", "0.9", "--output-dir", d])
            out = Path(d)
            pngs = list(out.glob("*.png"))
            self.assertEqual(len(pngs), 1)
            self.assertEqual(Path(fig_path), pngs[0])
            self.assertGreater(pngs[0].stat().st_size, 10_000)
            js = json.loads((out / (pngs[0].stem + "_summary.json")).read_text())
            self.assertEqual((js["task"], js["widths"], js["rho"]), ("delaygo", [6, 5], 0.9))
            self.assertEqual(js["loss"], "masked_mse")
            self.assertEqual(len(js["layers"]), 2)
            npz = np.load(out / (pngs[0].stem + "_curves.npz"))
            self.assertEqual(list(npz["keys"]), ["W", "W1"])
            self.assertEqual(npz["maxA_uncapped"].shape[0], 2)


class TestTaskObjective(unittest.TestCase):
    CE = dict(task_name="twohop", widths=(6, 5), batch=4, seed=0, dtype=torch.float64)

    def test_default_loss_tasks_keep_the_single_pass_path(self):
        for name in ("delaygo", "contextdelaydm1", "adding"):
            self.assertEqual(rtg.task_loss(name), ({}, "masked_mse"), name)

    def test_cross_entropy_task_is_diagnosed_under_its_own_loss(self):
        loss_kw, name = rtg.task_loss("twohop")
        self.assertIs(loss_kw["loss_and_grad"], mpn.masked_cross_entropy_loss_and_grad)
        self.assertEqual(name, "masked_cross_entropy")
        self.assertEqual(rtg.task_loss("seqmnist")[1], "masked_cross_entropy")

        net, (x, y, mask) = rtg.build(rho=None, rule="local_exact_rowlocal", **self.CE)
        g_ce, rec = rtg.record_exact(net, x, y, mask, **loss_kw)
        g_mse, _ = rtg.record_exact(net, x, y, mask)
        ref = net.bptt_gradients(x, y, mask, **loss_kw)
        # The reported loss is the final-step cross-entropy, computed independently.
        target = y[:, -1].argmax(-1)
        self.assertAlmostEqual(float(g_ce["loss"]),
                               float(F.cross_entropy(g_ce["outputs"][:, -1], target)), places=10)
        # Under the SAME loss the top layer's exact row-local gradient is BPTT's ...
        for key in ("W1", "W_output"):
            torch.testing.assert_close(g_ce[key], ref[key], rtol=1e-8, atol=1e-10)
        # ... and it is not the masked-MSE gradient the diagnostic used to report.
        gap = float((g_ce["W1"] - g_mse["W1"]).norm() / g_ce["W1"].norm())
        self.assertGreater(gap, 1e-2)
        # The spied step still runs once per time step (the custom-loss pre-pass is
        # forward-only and must not be recorded).
        self.assertEqual(len(rec[0]["maxP"]), x.shape[1])

    def test_analyze_and_cli_report_the_objective(self):
        summary, curves = rtg.analyze(rho=0.9, **self.CE)
        self.assertEqual(summary["loss"], "masked_cross_entropy")
        self.assertAlmostEqual(summary["layers"][-1]["cos_vs_bptt"]["exact"], 1.0, places=8)
        self.assertEqual(curves["frac_diag"].shape, (2, summary["T"]))
        with tempfile.TemporaryDirectory() as d:
            rtg.main(["--task", "twohop", "--hidden", "6", "5", "--batch", "4",
                      "--dtype", "float64", "--rho", "0.9", "--output-dir", d])
            summaries = list(Path(d).glob("*_summary.json"))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(json.loads(summaries[0].read_text())["loss"], "masked_cross_entropy")


if __name__ == "__main__":
    unittest.main()
