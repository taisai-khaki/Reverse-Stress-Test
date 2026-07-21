# RST IJPR Paper Implementation

This folder contains a runnable Python implementation of the paper:

`Reverse Stress Testing for Supply Chain Resilience: A Failure-First Analytical Framework for Identifying Non-Viability Pathways`

## Files

- `paper_rst_ijpr.py`: legacy submitted-model implementation retained for reproduction.
- `rst_revised_model.py`: revised operator, scenario libraries, multilabel attribution, and passive-active classification aligned with manuscript Sections 3 and 4.1-4.5.
- `generate_revised_analysis.py`: validation-gated five-seed primary, sensitivity, topology, dependence, and runtime analysis.
- `tests/test_revised_model.py`: system-operator and mechanism-attribution regression tests.
- `tests/test_revised_analysis.py`: response-setting and exact multilabel-signature coverage tests.

## What It Implements

- Stylized 3-tier weekly supply-chain simulation (52-week horizon).
- Failure thresholds:
  - Moderate: `SL_t < 0.90` for at least 2 consecutive periods.
  - Severe: `SL_t < 0.85` for at least 3 consecutive periods.
- Monte Carlo ground-truth generation.
- Forward Stress Testing (50-scenario baseline).
- Mechanism-guided RST candidate search (50-pathway budget).
- Passive vs active RST classification:
  - `Structural` vs `Adaptation-limited`.
- Failure-conditioned criticality scores.

## Revised Analysis

From this folder:

```powershell
python -m unittest discover -s tests -v
python generate_revised_analysis.py --output-dir 02_revised_analysis_outputs --seeds 42 314 1618 2026 2718 --replications-per-seed 1000 --workers 1
```

The analysis driver is resumable. Seed-level raw files already present in
`02_revised_analysis_outputs/raw_seed_runs/` are validated by row count and reused.
See `REPRODUCIBILITY.md` for the reviewer-oriented environment, validation gate,
expected archived results, and output map. Exact package versions are pinned in
`requirements.txt`.

## Legacy Reproduction

Reproduce submitted results and generate the revised tables:

```powershell
python generate_revision_outputs.py --replications 1000 --disturbance-replications 1000 --seed 42 --budget 50
```

Run the earlier reviewer R2-6 seed/topology analysis (superseded by the revised driver):

```powershell
python generate_r26_robustness.py --output-dir outputs_r26_robustness --seeds 10 20 30 40 50 --replications 1000 --budget 50
```

Quick legacy smoke test:

```powershell
python paper_rst_ijpr.py --replications 100 --seed 42 --budget 50 --output-dir outputs_smoke
```

## Outputs

- `summary.json`: top-line comparative metrics.
- `monte_carlo_ground_truth.csv`: replication-level Monte Carlo outcomes.
- `fst_results.csv`: forward stress testing scenario outcomes.
- `rst_results.csv`: candidate-level passive/active RST outcomes.
- `criticality_scores.csv`: failure-conditioned element importance.

Revision workflow outputs:

- `00_reproduce_submitted_results/`: gatekeeping audit tables and `reproduction_log.json`.
- `01_revised_simulation_outputs/`: revised Tables A-F and `revision_manifest.json`.
- `outputs_tables/fst_enhanced_results.csv`: FST-Enhanced-50 scenario outcomes.
- `outputs_tables/table6_corrected_comparison.csv`: corrected FST-original/FST-enhanced/RST comparison.
- `outputs_tables/table6b_fst_original_vs_enhanced.csv`: original vs enhanced FST baseline.
- `outputs_tables/table6c_mechanism_coverage_corrected.csv`: mechanism-family coverage diagnostics.
- `outputs_tables/summary_corrected.json`: corrected comparison summary and decision-rule result.
- `outputs_r26_robustness/r26_robustness_runs.csv`: per-seed/per-topology robustness results.
- `outputs_r26_robustness/r26_robustness_spread.csv`: mean, spread, and range by topology and overall.
- `outputs_r26_robustness/r26_robustness_topology_summary.csv`: compact topology-level summary.

Revised-model outputs:

- `02_revised_analysis_outputs/validation_system_operator.csv`: nominal-state, nonnegativity, conservation, corridor, rerouting, and reserve checks.
- `02_revised_analysis_outputs/validation_mechanism_attribution.csv`: causal-prefix, order-invariance, and mechanism-counterfactual checks.
- `02_revised_analysis_outputs/monte_carlo_primary_5000.csv`: pooled five-seed primary Monte Carlo results.
- `02_revised_analysis_outputs/primary_method_comparison.csv`: baseline FST, mechanism-balanced FST, and RST comparison.
- `02_revised_analysis_outputs/primary_exact_multilabel_signature_coverage.csv`: frequency-weighted and unique exact-signature coverage by method.
- `02_revised_analysis_outputs/primary_exact_multilabel_signature_catalog.csv`: Monte Carlo signature frequencies and method-specific exact matches.
- `02_revised_analysis_outputs/sensitivity_*.csv`: threshold, response, severity-plausibility weight, and dependence analyses.
- `02_revised_analysis_outputs/robustness_topology_*.csv`: five-seed results and spread for both topologies.
- `02_revised_analysis_outputs/runtime_*.csv`: primary-job timings and explicit scaling evidence.
- `02_revised_analysis_outputs/analysis_summary.json`: top-line results and manuscript-consistency flags.

## Notes

- The legacy and revised models are intentionally separate so the submitted-result audit remains reproducible.
- The revised operator uses zero ordinary focal-plant inventory and a separate protected 900-unit reserve, fixed tier-1 allocation shares, the stated order-up-to level of 1,100, multiplicative overlap, and the nonanticipative rerouting/reserve operator.
- Mechanism attribution is multilabel and counterfactual. Adaptation-limited is only a passive-active classification.
- Exact multilabel-signature coverage requires equality of the complete mechanism-label set; sharing only one label does not count as an exact match.
- The primary response baseline remains `tau_B=0.20D`; the four `kappa_alt` and `rho_rel` sensitivity reruns use `tau_B=0.17D` for consistency with the manuscript robustness design.
- Revised numerical results are not calibrated to retain the submitted counts. `analysis_summary.json` records the resulting discrepancies explicitly.
- The manuscript does not define a primary-label tie-break. Scalar primary-family summaries use the fixed priority `HCD, CMD, PAC, RL-TO, CD`; multilabel outputs remain available in every pathway table.
- Topology B preserves total upstream capacity at 1,000 units/week and allocates it 2:1 across T2P-A and T2P-B according to served tier-1 channels.
