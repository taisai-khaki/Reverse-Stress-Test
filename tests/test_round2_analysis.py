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
