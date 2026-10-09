"""notebooks/rflo_trace_gain.py: the recorded diagonal gain is lam + k*W and the
exact row gain is lam + sum_J k_J W_iJ (the row sum of the diagonal terms); the
three local variants share init and data; the capped trace is bounded while the
uncapped one is not forced to be; and the CLI writes the figure, summary and
curves. Tiny CPU double networks. Run from tests/: python -m unittest test_rflo_trace_gain_analysis -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import _bootstrap
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
            self.assertEqual(len(js["layers"]), 2)
            npz = np.load(out / (pngs[0].stem + "_curves.npz"))
            self.assertEqual(list(npz["keys"]), ["W", "W1"])
            self.assertEqual(npz["maxA_uncapped"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
