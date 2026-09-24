import unittest

import pandas as pd

import generate_round2_analysis as round2
import rst_revised_model as model


class Round2ProtocolTests(unittest.TestCase):
    def test_seven_libraries_have_fifty_unique_candidates(self):
        libraries, manifest = round2._build_candidate_libraries()
        self.assertEqual(set(libraries), set(round2.LIBRARY_ORDER))
        self.assertEqual(len(manifest), 350)
        for pathways in libraries.values():
            self.assertEqual(len(pathways), 50)
            self.assertEqual(len({round2._event_key(pathway.events) for pathway in pathways}), 50)

    def test_variant_changes_match_protocol(self):
        _, rows = round2._build_candidate_libraries()
        manifest = pd.DataFrame(rows)
        changed = manifest[manifest["changed_fields"].ne("")]
        self.assertEqual(set(changed.loc[changed["variant_id"] == "MI-C", "candidate_id"]), {"FST-MI-23", "FST-MI-24", "FST-MI-25", "FST-MI-26"})
        self.assertEqual(set(changed.loc[changed["variant_id"] == "MI-MDA", "candidate_id"]), {"FST-MI-13", "FST-MI-14", "FST-MI-15", "FST-MI-16"})
        self.assertEqual(set(changed.loc[changed["variant_id"] == "MI-MD", "candidate_id"]), {f"FST-MI-{i:02d}" for i in range(23, 41)})

    def test_reference_grid_sizes(self):
        cfg_a = model.topology_a_config(**round2.PRIMARY_CFG_KWARGS)
        cfg_b = model.topology_b_config(**round2.PRIMARY_CFG_KWARGS)
        self.assertEqual(len(round2._reference_grid(cfg_a, "A")), 3699)
        self.assertEqual(len(round2._reference_grid(cfg_b, "B")), 5166)

    def test_canonical_event_key_is_order_invariant(self):
        first = model.Event("T2P", 0.25, 12, 3, 3)
        second = model.Event("corridor", 0.35, 12, 3, 3)
        self.assertEqual(round2._event_key((first, second)), round2._event_key((second, first)))

    def test_topology_b_event_alias_is_canonicalized(self):
        cfg = model.topology_b_config(**round2.PRIMARY_CFG_KWARGS)
        logical = (model.Event("T2P", 0.35, 12, 3, 3),)
        explicit = (model.Event("T2P-A", 0.35, 12, 3, 3),)
        self.assertEqual(round2._event_key(logical, cfg), round2._event_key(explicit, cfg))

    def test_common_scores_are_method_scoped(self):
        fst = model.Pathway("FST-MI-01", model.CD, (model.Event("T2P", 0.3, 12, 3, 3),))
        rst = model.Pathway("FST-MI-01", model.CMD, (model.Event("corridor", 0.3, 12, 3, 3),))
        rows = pd.DataFrame([
            {"method": "MI-FST", "threshold": "moderate", "candidate_id": "FST-MI-01", "passive_failure": True, "severity": 1.0, "plausibility": 1.0},
            {"method": "MI-C", "threshold": "moderate", "candidate_id": "FST-MI-01", "passive_failure": True, "severity": 1.0, "plausibility": 1.0},
        ])
        scores = pd.DataFrame(round2._common_scores(rows, {"MI-FST": [fst], "MI-C": [rst]}))
        self.assertAlmostEqual(float(scores[(scores["method"] == "MI-FST") & (scores["element"] == "T2P")]["element_score"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(scores[(scores["method"] == "MI-C") & (scores["element"] == "corridor")]["element_score"].iloc[0]), 1.0)

    def test_rank_scores_groups_tolerance_equal_values(self):
        ranks = round2._rank_scores({"S1A": 0.5, "S1B": 0.5 + 1e-13, "T2P": 0.2})
        self.assertEqual(ranks["S1A"], ranks["S1B"])
        self.assertGreater(ranks["T2P"], ranks["S1A"])

    def test_completed_only_scaling_summary(self):
        rows = pd.DataFrame([
            {"architecture": "shared", "n": 3, "horizon": 52, "K": 4, "B": 50, "repetition": 1, "status": "completed", "completed_candidates": 50, "end_to_end_seconds": 2.0, "incremental_peak_memory_mb": 4.0, "peak_memory_mb": 100.0, "unique_simulations": 5, "cache_hits": 2, "attribution_pair_iterations": 8},
            {"architecture": "shared", "n": 3, "horizon": 52, "K": 4, "B": 50, "repetition": 2, "status": "timeout", "completed_candidates": 1, "end_to_end_seconds": 120.0, "incremental_peak_memory_mb": 8.0, "peak_memory_mb": 200.0, "unique_simulations": 1, "cache_hits": 0, "attribution_pair_iterations": 1},
        ])
        summary = round2._summarize_scaling(rows)
        self.assertEqual(int(summary.iloc[0]["completed_repetitions"]), 1)
        self.assertEqual(int(summary.iloc[0]["timeout_repetitions"]), 1)
        self.assertEqual(float(summary.iloc[0]["median_seconds"]), 2.0)

    def test_scaling_figure_filters_nonreference_axes(self):
        rows = pd.DataFrame([
            {"architecture": "shared", "n": 3, "horizon": 52, "K": 4, "B": 50, "completed_repetitions": 1, "median_seconds": 2.0},
            {"architecture": "shared", "n": 3, "horizon": 104, "K": 4, "B": 50, "completed_repetitions": 1, "median_seconds": 9.0},
        ])
        figure_data = round2._scaling_figure_data(rows)
        self.assertEqual(set(figure_data[figure_data["focal_axis"] == "n"]["horizon"]), {52})

    def test_omission_figure_uses_uncovered_frequency(self):
        catalog = pd.DataFrame([{"cohort": "archived", "topology": "A", "threshold": "moderate", "method": "RST", "signature": "x", "failure_frequency": 4, "total_failures": 10, "covered": False, "uncovered_fraction": 0.4}])
        figure_data = round2._omission_figure_data(catalog)
        self.assertEqual(int(figure_data.iloc[0]["omitted_frequency"]), 4)
        self.assertAlmostEqual(float(figure_data.iloc[0]["omission_fraction"]), 0.4)

    def test_portable_hash_normalizes_line_endings(self):
        path = round2.PROJECT_ROOT / ".hash_test_line_endings.tmp"
        try:
            path.write_bytes(b"a\r\nb\r\n")
            self.assertEqual(round2._sha256_canonical_file(path), round2._sha256_text("a\nb\n"))
        finally:
            path.unlink(missing_ok=True)

    def test_rst22_validation_reads_candidate_result(self):
        candidate_results = pd.DataFrame([{"method": "RST", "topology": "A", "candidate_id": "RST-22", "threshold": "moderate", "C0": 2.0, "Cadm": 1.0, "gamma_raw": 0.5, "gamma_clipped": 0.5}])
        self.assertTrue(round2._rst22_validation(candidate_results)["passed"])

    def test_rst22_raw_gamma_is_preserved(self):
        cfg = model.topology_a_config(**round2.PRIMARY_CFG_KWARGS)
        pathway = model.rst_candidate_library()[21]
        passive = model.run_simulation(pathway.events, cfg)
        active = model.run_simulation(pathway.events, cfg, active_mode=True)
        c0 = model.pathway_severity(passive.service_levels)
        cadm = model.pathway_severity(active.service_levels)
        gamma = 1.0 - cadm / c0
        self.assertAlmostEqual(c0, 0.19259259259259276)
        self.assertAlmostEqual(cadm, 0.26666666666666683)
        self.assertAlmostEqual(gamma, -0.38461538461538436)
        self.assertEqual(float(max(0.0, min(1.0, gamma))), 0.0)


if __name__ == "__main__":
    unittest.main()
