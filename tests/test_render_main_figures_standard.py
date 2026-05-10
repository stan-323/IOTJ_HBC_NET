import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_main_figures_standard as figures


class Fig8AxisScalingTests(unittest.TestCase):
    def test_fig8b_uses_complete_coarse_stress_fn_sweep(self):
        leaderboard = pd.DataFrame(
            {
                "model": ["stress_false_negative"] * 7,
                "param_name": ["p_fn"] * 7,
                "iteration": [2] * 7,
                "param_value": [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
                "stress_affected_margin_min": [0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01],
                "stress_affected_margin_min_ci95": [0.005] * 7,
            }
        )

        boundary = figures.fig8_stress_fn_coarse_sweep(leaderboard)

        np.testing.assert_allclose(
            boundary["param_value_numeric"].to_numpy(),
            np.asarray([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]),
        )
        self.assertTrue(boundary["iteration"].eq(2).all())

    def test_fig8b_margin_axis_tracks_current_leaderboard_scale(self):
        leaderboard = pd.DataFrame(
            {
                "model": ["stress_false_negative"] * 7,
                "param_name": ["p_fn"] * 7,
                "iteration": [2] * 7,
                "param_value": [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
                "stress_affected_margin_min": [0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01],
                "stress_affected_margin_min_ci95": [0.005] * 7,
            }
        )
        boundary = figures.fig8_stress_fn_coarse_sweep(leaderboard)

        y = boundary["stress_affected_margin_min"]
        err = boundary["stress_affected_margin_min_ci95"]
        lower, upper = figures.fig8_margin_axis_limits(y, err)

        self.assertLessEqual(lower, float((y - err).min()))
        self.assertGreaterEqual(upper, float((y + err).max()))
        self.assertLess(upper, 0.12)

    def test_fig8c_shortage_axis_uses_scientific_small_scale(self):
        severity = np.asarray([4.16e-5, 5.60e-6, 6.88e-5, 6.88e-5, 5.13e-5])

        scaled, scale_label, upper = figures.fig8_shortage_axis_values(severity)

        np.testing.assert_allclose(scaled, severity * 1e4)
        self.assertIn("10^{-4}", scale_label)
        self.assertGreaterEqual(upper, float(scaled.max()) * 1.15)
        self.assertLess(upper, 2.0)


if __name__ == "__main__":
    unittest.main()
