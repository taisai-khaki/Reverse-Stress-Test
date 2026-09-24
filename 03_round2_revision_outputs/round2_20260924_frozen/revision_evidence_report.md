# Second-round IJPR computational evidence report

Protocol: `IJPR-round2-v1-frozen-20260924`.

## Baseline freeze

The audited source hashes match the prescribed values. The no-resume replay reproduced 2455 moderate and 1445 severe Monte Carlo failures, FST 33/11, MI-FST 46/35, and RST 48/39 with 41/7 structural/adaptation-limited pathways.

## Claim support

The calibration comparisons are fixed-library comparisons. Any positive RST-minus-MI coverage difference is conditional on the specified candidate library and model; an interval crossing zero is inconclusive rather than evidence of equivalence.

The archived primary exact-signature contrast is checked against the required 122/2,455 = 4.9694501 percentage-point difference. The new variants, validation cohort, and topology-B results are reported separately and were not used to choose a variant.

The bounded reference audit is finite-grid evidence only. It does not establish completeness over arbitrary three-event pathways, continuous magnitudes, arbitrary times, or the full disturbance space.

Scaling results are engineering measurements for the implemented operator. Timeout, memory, or incomplete-workload rows are retained as observed limits rather than silently reduced settings.

## Source hashes

- `generate_revised_analysis.py`: `28a62db0967dccdcda6eb39449da95a7fc934c3affe62e67db9ec8005db62ac2`
- `rst_revised_model.py`: `22a277deae4cea8bcbb38b216a283472c661bec9490014136346ef3285e2ab75`
- `generate_round2_analysis.py`: `1903c9a0a00343953fa3177e4135bd7ded7ffa3a057f690b3c00604125dacab4`

## Output map

- `candidate_manifest.csv` and `candidate_results.csv`: seven fixed 50-candidate libraries with full event and threshold-specific records.
- `mc_evaluation.csv`, `coverage_summary.csv`, `paired_coverage_differences.csv`, and `signature_gain_loss.csv`: archived and independent validation cohorts, both topologies, and bootstrap intervals.
- `reference_grid_manifest.csv`, `reference_grid_results.csv`, and `bounded_omission_summary.csv`: 3,699 topology-A and 5,166 topology-B reference pathways.
- `scaling_runs.csv` and `scaling_summary.csv`: actual width, horizon, event-count, and budget workloads with counterfactual accounting.
- `figure_*_data.csv` and `figures/*.svg`: underlying data and three principal figure groups.

## Bounded omission result

The reference audit contains 28 method/cohort/topology/threshold summary rows; denominators and numerators are stored in the CSV rather than represented only as percentages.

## Scaling result

131 timed workload repetitions completed; timeout and incomplete rows, if any, remain visible in `scaling_runs.csv`.
