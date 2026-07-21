from __future__ import annotations

import itertools
import math
import unittest

from rst_revised_model import (
    ADAPTATION_LIMITED,
    CD,
    CMD,
    HCD,
    MECHANISM_FAMILIES,
    MODERATE,
    PAC,
    RL_TO,
    Event,
    NetworkConfig,
    attribute_mechanisms,
    mechanism_guided_rst_search,
    rst_candidate_library,
    run_simulation,
    threshold_cross_info,
)


class SystemOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = NetworkConfig()
        cls.candidates = rst_candidate_library()

    def test_nominal_steady_state_for_52_periods(self) -> None:
        result = run_simulation([], self.cfg)
        self.assertEqual(len(result.service_levels), 52)
        self.assertTrue(all(math.isclose(value, 1.0, abs_tol=1e-12) for value in result.service_levels))
        self.assertTrue(all(math.isclose(value, 0.0, abs_tol=1e-12) for value in result.backlog_series))
        self.assertTrue(all(math.isclose(value, 0.0, abs_tol=1e-12) for value in result.fp_inventory_series))
        for inventory in result.tier1_inventory_series.values():
            self.assertTrue(all(math.isclose(value, 200.0, abs_tol=1e-12) for value in inventory))

    def test_nonnegative_state_and_pipeline_conservation(self) -> None:
        for pathway in self.candidates:
            for active_mode in (False, True):
                result = run_simulation(pathway.events, self.cfg, active_mode=active_mode)
                self.assertGreaterEqual(min(result.backlog_series), -1e-10)
                self.assertGreaterEqual(min(result.fp_inventory_series), -1e-10)
                self.assertGreaterEqual(min(result.reserve_series), -1e-10)
                for inventory in result.tier1_inventory_series.values():
                    self.assertGreaterEqual(min(inventory), -1e-10)
                self.assertLessEqual(
                    max(row["max_pipeline_conservation_error"] for row in result.trace),
                    1e-9,
                )
                self.assertLessEqual(
                    max(row["corridor_pipeline_conservation_error"] for row in result.trace),
                    1e-9,
                )
                self.assertLessEqual(
                    max(row["max_tier1_inventory_balance_error"] for row in result.trace),
                    1e-9,
                )

    def test_corridor_and_rerouting_bounds(self) -> None:
        for pathway in self.candidates:
            result = run_simulation(pathway.events, self.cfg, active_mode=True)
            for row in result.trace:
                self.assertLessEqual(row["corridor_throughput"], row["corridor_capacity"] + 1e-9)
                self.assertLessEqual(row["rerouted_flow"], row["blocked_flow"] + 1e-9)
                self.assertLessEqual(row["rerouted_flow"], self.cfg.active_alt_corridor_units + 1e-9)
                self.assertLessEqual(row["rerouted_flow"], row["pre_response_shortfall"] + 1e-9)

    def test_reserve_monotonicity_and_release_cap(self) -> None:
        for pathway in self.candidates:
            result = run_simulation(pathway.events, self.cfg, active_mode=True)
            for previous, current in zip(result.reserve_series, result.reserve_series[1:]):
                self.assertLessEqual(current, previous + 1e-9)
            self.assertLessEqual(
                sum(result.reserve_release_series),
                self.cfg.active_reserve_release_frac * self.cfg.protected_reserve_capacity + 1e-9,
            )
            self.assertGreaterEqual(
                min(result.reserve_series),
                (1.0 - self.cfg.active_reserve_release_frac)
                * self.cfg.protected_reserve_capacity
                - 1e-9,
            )


class MechanismAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = NetworkConfig()
        cls.pathways = {pathway.pathway_id: pathway for pathway in rst_candidate_library()}

    def _attribute(self, pathway_id: str):
        pathway = self.pathways[pathway_id]
        passive = run_simulation(pathway.events, self.cfg)
        return attribute_mechanisms(
            pathway.events,
            self.cfg,
            threshold=MODERATE,
            passive_result=passive,
        )

    def test_causal_prefix_excludes_late_events(self) -> None:
        base = self.pathways["RST-01"]
        base_result = run_simulation(base.events, self.cfg)
        crossed, first_qualified = threshold_cross_info(base_result.service_levels, MODERATE)
        self.assertTrue(crossed)
        self.assertIsNotNone(first_qualified)
        late_event = Event("demand", 0.40, int(first_qualified) + 2, 4, 2)
        extended_events = (*base.events, late_event)
        extended_result = run_simulation(extended_events, self.cfg)
        base_labels = attribute_mechanisms(
            base.events, self.cfg, passive_result=base_result
        ).labels
        extended = attribute_mechanisms(
            extended_events, self.cfg, passive_result=extended_result
        )
        self.assertEqual(base_labels, extended.labels)
        self.assertNotIn(1, extended.causal_prefix_indices)

    def test_label_order_invariance(self) -> None:
        pathway = self.pathways["RST-46"]
        passive = run_simulation(pathway.events, self.cfg)
        expected = attribute_mechanisms(
            pathway.events, self.cfg, passive_result=passive
        ).labels
        for order in itertools.permutations(MECHANISM_FAMILIES):
            observed = attribute_mechanisms(
                pathway.events,
                self.cfg,
                passive_result=passive,
                label_order=order,
            ).labels
            self.assertEqual(expected, observed)

    def test_single_event_cd(self) -> None:
        attribution = self._attribute("RST-05")
        self.assertIn(CD, attribution.labels)
        self.assertEqual(attribution.single_event_crossings, (0,))

    def test_hcd_removal_counterfactual(self) -> None:
        attribution = self._attribute("RST-01")
        self.assertIn(HCD, attribution.labels)
        self.assertEqual(attribution.hcd_removal_elements, ("T2P",))

    def test_cmd_constituent_events_do_not_fail(self) -> None:
        attribution = self._attribute("RST-23")
        self.assertIn(CMD, attribution.labels)
        self.assertEqual(attribution.single_event_crossings, ())

    def test_pac_direct_impact_counterfactual(self) -> None:
        attribution = self._attribute("RST-01")
        self.assertIn(PAC, attribution.labels)
        self.assertFalse(attribution.direct_impact_crossing)

    def test_rlto_recovery_window_and_removal_counterfactual(self) -> None:
        pathway = self.pathways["RST-46"]
        attribution = self._attribute("RST-46")
        self.assertIn(RL_TO, attribution.labels)
        self.assertIn((0, 1), attribution.rlto_support_pairs)

        first, second = pathway.events
        recovery_end = first.start_week + first.duration_weeks + first.recovery_weeks
        end_boundary_events = (
            first,
            Event(
                second.target,
                second.magnitude,
                recovery_end,
                second.duration_weeks,
                second.recovery_weeks,
            ),
        )
        end_result = run_simulation(end_boundary_events, self.cfg)
        end_attribution = attribute_mechanisms(
            end_boundary_events, self.cfg, passive_result=end_result
        )
        self.assertNotIn(RL_TO, end_attribution.labels)

    def test_classification_is_separate_from_mechanism_family(self) -> None:
        evaluations, _ = mechanism_guided_rst_search(
            rst_candidate_library(), self.cfg
        )
        adaptation_limited = [
            evaluation
            for evaluation in evaluations
            if evaluation.classification == ADAPTATION_LIMITED
        ]
        self.assertTrue(adaptation_limited)
        self.assertTrue(
            all(
                ADAPTATION_LIMITED not in evaluation.mechanism_labels
                for evaluation in adaptation_limited
            )
        )
        self.assertEqual(
            sum(evaluation.passive_severe_failure for evaluation in evaluations),
            39,
        )


if __name__ == "__main__":
    unittest.main()
