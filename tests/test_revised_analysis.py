from __future__ import annotations

import math
import unittest

import pandas as pd

from generate_revised_analysis import (
    RESPONSE_SENSITIVITY_SETTINGS,
    _coverage_rows,
    _exact_signature_catalog,
)
from rst_revised_model import ADAPTATION_LIMITED, CD, HCD, PAC, STRUCTURAL


class RevisedAnalysisTests(unittest.TestCase):
    def test_four_response_reruns_use_tau_017(self) -> None:
        self.assertEqual(
            RESPONSE_SENSITIVITY_SETTINGS[4:],
            (
                (0.17, 0.0, 0.30),
                (0.17, 500.0, 0.30),
                (0.17, 200.0, 0.15),
                (0.17, 200.0, 0.50),
            ),
        )

    def test_exact_signature_coverage_requires_complete_label_match(self) -> None:
        monte_carlo = pd.DataFrame(
            [
                {
                    "moderate_passive_failure": True,
                    "passive_active_classification": STRUCTURAL,
                    "primary_mechanism": PAC,
                    "mechanism_labels": f"{CD}; {PAC}",
                },
                {
                    "moderate_passive_failure": True,
                    "passive_active_classification": STRUCTURAL,
                    "primary_mechanism": CD,
                    "mechanism_labels": CD,
                },
                {
                    "moderate_passive_failure": True,
                    "passive_active_classification": ADAPTATION_LIMITED,
                    "primary_mechanism": HCD,
                    "mechanism_labels": f"{CD}; {HCD}",
                },
            ]
        )
        forward = pd.DataFrame(
            [
                {
                    "moderate_crossing": True,
                    "mechanism_labels_moderate": f"{CD}; {PAC}",
                },
                {
                    "moderate_crossing": True,
                    "mechanism_labels_moderate": HCD,
                },
            ]
        )
        methods = {"FST": (forward, "fst")}

        coverage = _coverage_rows(monte_carlo, methods).iloc[0]
        self.assertTrue(
            math.isclose(
                coverage["exact_multilabel_signature_coverage_full_denominator"],
                1 / 3,
            )
        )
        self.assertTrue(
            math.isclose(
                coverage["multilabel_any_coverage_full_denominator"],
                1.0,
            )
        )

        catalog = _exact_signature_catalog(monte_carlo, methods)
        discovered = catalog.set_index("exact_multilabel_signature")[
            "fst_signature_discovered"
        ]
        self.assertTrue(discovered[f"{CD}; {PAC}"])
        self.assertFalse(discovered[CD])
        self.assertFalse(discovered[f"{CD}; {HCD}"])


if __name__ == "__main__":
    unittest.main()
