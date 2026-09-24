# Second-round IJPR computational evidence report

Protocol: `IJPR-round2-v1-frozen-20260924`.

## Baseline freeze

The audited source hashes match the prescribed values. The no-resume replay reproduced 2455 moderate and 1445 severe Monte Carlo failures, FST 33/11, MI-FST 46/35, and RST 48/39 with 41/7 structural/adaptation-limited pathways.

## Claim support

The calibration comparisons are fixed-library comparisons. Any positive RST-minus-MI coverage difference is conditional on the specified candidate library and model; an interval crossing zero is inconclusive rather than evidence of equivalence.

The archived primary exact-signature contrast is checked against the required 122/2,455 = 4.9694501 percentage-point difference. The new variants, validation cohort, and topology-B results are reported separately and were not used to choose a variant.

The bounded reference audit is finite-grid evidence only. It does not establish completeness over arbitrary three-event pathways, continuous magnitudes, arbitrary times, or the full disturbance space.

Scaling results are engineering measurements for the implemented operator. Medians and IQRs use completed repetitions only; timeout counts and partial progress remain separate, and memory is reported as incremental process usage over the cell baseline.

## Source hashes

- `generate_revised_analysis.py`: `28a62db0967dccdcda6eb39449da95a7fc934c3affe62e67db9ec8005db62ac2`
- `rst_revised_model.py`: `22a277deae4cea8bcbb38b216a283472c661bec9490014136346ef3285e2ab75`
- `generate_round2_analysis.py`: `1e1bc187d44a209f038e22db355d05ec24bb31b67cdf0255a905f8c7e40731c2`

## Output map

- `candidate_manifest.csv` and `candidate_results.csv`: seven fixed 50-candidate libraries with full event and threshold-specific records.
- `mc_evaluation.csv`, `coverage_summary.csv`, `paired_coverage_differences.csv`, and `signature_gain_loss.csv`: archived and independent validation cohorts, both topologies, and bootstrap intervals.
- `reference_grid_manifest.csv`, `reference_grid_results.csv`, and `bounded_omission_summary.csv`: 3,699 topology-A and 5,166 topology-B reference pathways.
- `scaling_runs.csv` and `scaling_summary.csv`: actual width, horizon, event-count, and budget workloads with counterfactual accounting.
- `figure_*_data.csv` and `figures/*.svg`: underlying data and three principal figure groups.

## Bounded omission result

The reference audit contains 28 method/cohort/topology/threshold summary rows; denominators and numerators are stored in the CSV rather than represented only as percentages.

| Topology | Method | Threshold | Signature recall | Event recall |
|---|---|---|---:|---:|
| A | FST-original | moderate | 0.677598 | 0.003951 |
| A | FST-original | severe | 0.485062 | 0.002636 |
| A | MI-FST | moderate | 0.786646 | 0.002766 |
| A | MI-FST | severe | 0.666960 | 0.002636 |
| A | RST | moderate | 0.824575 | 0.003161 |
| A | RST | severe | 0.749561 | 0.004394 |
| A | MI-C | moderate | 0.786646 | 0.003951 |
| A | MI-C | severe | 0.666960 | 0.003515 |
| A | MI-M | moderate | 0.794943 | 0.005531 |
| A | MI-M | severe | 0.704745 | 0.007030 |
| A | MI-MD | moderate | 0.794943 | 0.005531 |
| A | MI-MD | severe | 0.662566 | 0.006151 |
| A | MI-MDA | moderate | 0.824575 | 0.007112 |
| A | MI-MDA | severe | 0.749561 | 0.008787 |
| B | FST-original | moderate | 0.611969 | 0.003150 |
| B | FST-original | severe | 0.367220 | 0.002075 |
| B | MI-FST | moderate | 0.798110 | 0.002205 |
| B | MI-FST | severe | 0.544606 | 0.002075 |
| B | RST | moderate | 0.836850 | 0.002205 |
| B | RST | severe | 0.625519 | 0.001037 |
| B | MI-C | moderate | 0.798110 | 0.003150 |
| B | MI-C | severe | 0.544606 | 0.002075 |
| B | MI-M | moderate | 0.803150 | 0.004409 |
| B | MI-M | severe | 0.479253 | 0.002075 |
| B | MI-MD | moderate | 0.803150 | 0.004094 |
| B | MI-MD | severe | 0.479253 | 0.002075 |
| B | MI-MDA | moderate | 0.833386 | 0.005354 |
| B | MI-MDA | severe | 0.591286 | 0.005187 |

## Scaling result

132 timed workload repetitions completed; timeout and incomplete rows, if any, remain visible in `scaling_runs.csv`.

## Variant and interval results

The coverage rows below are the observed variant results used by the comparison; bootstrap intervals are percentages on the 0-1 coverage scale.

| Cohort | Topology | Threshold | Method | Numerator/denominator | Coverage | Bootstrap 95% interval |
|---|---|---|---|---:|---:|---:|
| archived | A | moderate | FST-original | 1948/2455 | 0.793483 | [0.777689, 0.808884] |
| archived | A | moderate | MI-FST | 2093/2455 | 0.852546 | [0.838211, 0.866640] |
| archived | A | moderate | RST | 2215/2455 | 0.902240 | [0.890283, 0.913729] |
| archived | A | moderate | MI-C | 2093/2455 | 0.852546 | [0.838514, 0.866178] |
| archived | A | moderate | MI-M | 2133/2455 | 0.868839 | [0.855189, 0.882036] |
| archived | A | moderate | MI-MD | 2133/2455 | 0.868839 | [0.855043, 0.882257] |
| archived | A | moderate | MI-MDA | 2215/2455 | 0.902240 | [0.890220, 0.913876] |
| archived | A | severe | FST-original | 994/1445 | 0.687889 | [0.664138, 0.711661] |
| archived | A | severe | MI-FST | 1188/1445 | 0.822145 | [0.802256, 0.841639] |
| archived | A | severe | RST | 1284/1445 | 0.888581 | [0.872654, 0.904630] |
| archived | A | severe | MI-C | 1188/1445 | 0.822145 | [0.802120, 0.841742] |
| archived | A | severe | MI-M | 1268/1445 | 0.877509 | [0.860184, 0.894045] |
| archived | A | severe | MI-MD | 1232/1445 | 0.852595 | [0.834576, 0.870345] |
| archived | A | severe | MI-MDA | 1284/1445 | 0.888581 | [0.871983, 0.904330] |
| validation | A | moderate | FST-original | 2000/2483 | 0.805477 | [0.789642, 0.820751] |
| validation | A | moderate | MI-FST | 2114/2483 | 0.851389 | [0.837534, 0.865031] |
| validation | A | moderate | RST | 2237/2483 | 0.900926 | [0.888888, 0.912779] |
| validation | A | moderate | MI-C | 2114/2483 | 0.851389 | [0.837535, 0.864952] |
| validation | A | moderate | MI-M | 2156/2483 | 0.868304 | [0.854632, 0.881533] |
| validation | A | moderate | MI-MD | 2156/2483 | 0.868304 | [0.854910, 0.881343] |
| validation | A | moderate | MI-MDA | 2237/2483 | 0.900926 | [0.888755, 0.912531] |
| validation | A | severe | FST-original | 951/1404 | 0.677350 | [0.652569, 0.701718] |
| validation | A | severe | MI-FST | 1127/1404 | 0.802707 | [0.781634, 0.823358] |
| validation | A | severe | RST | 1240/1404 | 0.883191 | [0.865564, 0.899371] |
| validation | A | severe | MI-C | 1127/1404 | 0.802707 | [0.781831, 0.823613] |
| validation | A | severe | MI-M | 1213/1404 | 0.863960 | [0.845568, 0.881124] |
| validation | A | severe | MI-MD | 1189/1404 | 0.846866 | [0.827857, 0.865493] |
| validation | A | severe | MI-MDA | 1240/1404 | 0.883191 | [0.866118, 0.899789] |
| archived | B | moderate | FST-original | 1592/2133 | 0.746367 | [0.727314, 0.764594] |
| archived | B | moderate | MI-FST | 1818/2133 | 0.852321 | [0.837209, 0.867453] |
| archived | B | moderate | RST | 1942/2133 | 0.910455 | [0.898453, 0.922321] |
| archived | B | moderate | MI-C | 1818/2133 | 0.852321 | [0.837187, 0.867130] |
| archived | B | moderate | MI-M | 1841/2133 | 0.863104 | [0.848231, 0.877655] |
| archived | B | moderate | MI-MD | 1841/2133 | 0.863104 | [0.848372, 0.877660] |
| archived | B | moderate | MI-MDA | 1929/2133 | 0.904360 | [0.891757, 0.916708] |
| archived | B | severe | FST-original | 577/961 | 0.600416 | [0.569620, 0.631415] |
| archived | B | severe | MI-FST | 795/961 | 0.827263 | [0.802997, 0.850631] |
| archived | B | severe | RST | 833/961 | 0.866805 | [0.844937, 0.887464] |
| archived | B | severe | MI-C | 795/961 | 0.827263 | [0.803628, 0.850516] |
| archived | B | severe | MI-M | 730/961 | 0.759625 | [0.732732, 0.786308] |
| archived | B | severe | MI-MD | 730/961 | 0.759625 | [0.732322, 0.786109] |
| archived | B | severe | MI-MDA | 790/961 | 0.822060 | [0.798292, 0.845998] |
| validation | B | moderate | FST-original | 1649/2171 | 0.759558 | [0.741666, 0.777583] |
| validation | B | moderate | MI-FST | 1846/2171 | 0.850299 | [0.834994, 0.865222] |
| validation | B | moderate | RST | 1975/2171 | 0.909719 | [0.897388, 0.921868] |
| validation | B | moderate | MI-C | 1846/2171 | 0.850299 | [0.835520, 0.865456] |
| validation | B | moderate | MI-M | 1861/2171 | 0.857209 | [0.841914, 0.871700] |
| validation | B | moderate | MI-MD | 1861/2171 | 0.857209 | [0.842105, 0.871534] |
| validation | B | moderate | MI-MDA | 1957/2171 | 0.901428 | [0.888888, 0.914034] |
| validation | B | severe | FST-original | 545/931 | 0.585392 | [0.553813, 0.616953] |
| validation | B | severe | MI-FST | 742/931 | 0.796992 | [0.770562, 0.822174] |
| validation | B | severe | RST | 774/931 | 0.831364 | [0.806769, 0.854726] |
| validation | B | severe | MI-C | 742/931 | 0.796992 | [0.770943, 0.822293] |
| validation | B | severe | MI-M | 685/931 | 0.735768 | [0.707708, 0.763637] |
| validation | B | severe | MI-MD | 685/931 | 0.735768 | [0.707291, 0.763266] |
| validation | B | severe | MI-MDA | 738/931 | 0.792696 | [0.766524, 0.818089] |

## Paired comparisons

| Cohort | Topology | Threshold | Comparator | RST numerator | Comparator numerator | Denominator | Observed delta pp | Per-seed mean pp | Per-seed SD pp | Bootstrap 95% interval pp |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| archived | A | moderate | FST-original | 2215 | 1948 | 2455 | 10.875764 | 10.869214 | 1.613263 | [9.670834, 12.118780] |
| archived | A | moderate | MI-FST | 2215 | 2093 | 2455 | 4.969450 | 4.970884 | 0.692952 | [4.139323, 5.839170] |
| archived | A | moderate | MI-C | 2215 | 2093 | 2455 | 4.969450 | 4.970884 | 0.692952 | [4.149858, 5.827221] |
| archived | A | moderate | MI-M | 2215 | 2133 | 2455 | 3.340122 | 3.339062 | 0.875249 | [2.655220, 4.068384] |
| archived | A | moderate | MI-MD | 2215 | 2133 | 2455 | 3.340122 | 3.339062 | 0.875249 | [2.638080, 4.060105] |
| archived | A | moderate | MI-MDA | 2215 | 2215 | 2455 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| archived | A | severe | FST-original | 1284 | 994 | 1445 | 20.069204 | 20.081213 | 1.359574 | [17.706882, 22.476083] |
| archived | A | severe | MI-FST | 1284 | 1188 | 1445 | 6.643599 | 6.630488 | 1.850190 | [4.981624, 8.411844] |
| archived | A | severe | MI-C | 1284 | 1188 | 1445 | 6.643599 | 6.630488 | 1.850190 | [4.912257, 8.345459] |
| archived | A | severe | MI-M | 1284 | 1268 | 1445 | 1.107266 | 1.082562 | 1.864409 | [-0.149842, 2.372645] |
| archived | A | severe | MI-MD | 1284 | 1232 | 1445 | 3.598616 | 3.586856 | 0.997258 | [2.683568, 4.599324] |
| archived | A | severe | MI-MDA | 1284 | 1284 | 1445 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| validation | A | moderate | FST-original | 2237 | 2000 | 2483 | 9.544905 | 9.535216 | 0.871204 | [8.397239, 10.696122] |
| validation | A | moderate | MI-FST | 2237 | 2114 | 2483 | 4.953685 | 4.966188 | 1.005419 | [4.127874, 5.816733] |
| validation | A | moderate | MI-C | 2237 | 2114 | 2483 | 4.953685 | 4.966188 | 1.005419 | [4.106856, 5.818641] |
| validation | A | moderate | MI-M | 2237 | 2156 | 2483 | 3.262183 | 3.269245 | 0.642838 | [2.580645, 3.978893] |
| validation | A | moderate | MI-MD | 2237 | 2156 | 2483 | 3.262183 | 3.269245 | 0.642838 | [2.572347, 3.986002] |
| validation | A | moderate | MI-MDA | 2237 | 2237 | 2483 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| validation | A | severe | FST-original | 1240 | 951 | 1404 | 20.584046 | 20.572847 | 2.786653 | [18.266427, 22.877698] |
| validation | A | severe | MI-FST | 1240 | 1127 | 1404 | 8.048433 | 8.055046 | 3.173840 | [6.364285, 9.781882] |
| validation | A | severe | MI-C | 1240 | 1127 | 1404 | 8.048433 | 8.055046 | 3.173840 | [6.361144, 9.733254] |
| validation | A | severe | MI-M | 1240 | 1213 | 1404 | 1.923077 | 1.923434 | 0.823833 | [0.707139, 3.120467] |
| validation | A | severe | MI-MD | 1240 | 1189 | 1404 | 3.632479 | 3.635777 | 0.604623 | [2.698861, 4.661063] |
| validation | A | severe | MI-MDA | 1240 | 1240 | 1404 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| archived | B | moderate | FST-original | 1942 | 1592 | 2133 | 16.408814 | 16.388269 | 1.567650 | [14.865487, 18.049044] |
| archived | B | moderate | MI-FST | 1942 | 1818 | 2133 | 5.813408 | 5.814724 | 0.450425 | [4.832025, 6.840402] |
| archived | B | moderate | MI-C | 1942 | 1818 | 2133 | 5.813408 | 5.814724 | 0.450425 | [4.861103, 6.815021] |
| archived | B | moderate | MI-M | 1942 | 1841 | 2133 | 4.735115 | 4.733527 | 0.631072 | [3.842547, 5.681831] |
| archived | B | moderate | MI-MD | 1942 | 1841 | 2133 | 4.735115 | 4.733527 | 0.631072 | [3.853427, 5.663095] |
| archived | B | moderate | MI-MDA | 1942 | 1929 | 2133 | 0.609470 | 0.610991 | 0.271347 | [0.289429, 0.972684] |
| archived | B | severe | FST-original | 833 | 577 | 961 | 26.638918 | 26.515218 | 3.675272 | [23.643576, 29.657436] |
| archived | B | severe | MI-FST | 833 | 795 | 961 | 3.954214 | 3.908299 | 1.719117 | [2.111898, 5.773462] |
| archived | B | severe | MI-C | 833 | 795 | 961 | 3.954214 | 3.908299 | 1.719117 | [2.118632, 5.806486] |
| archived | B | severe | MI-M | 833 | 730 | 961 | 10.718002 | 10.643749 | 2.027142 | [8.805635, 12.689889] |
| archived | B | severe | MI-MD | 833 | 730 | 961 | 10.718002 | 10.643749 | 2.027142 | [8.832420, 12.725393] |
| archived | B | severe | MI-MDA | 833 | 790 | 961 | 4.474506 | 4.373674 | 2.020438 | [3.225725, 5.843475] |
| validation | B | moderate | FST-original | 1975 | 1649 | 2171 | 15.016122 | 15.017397 | 1.939739 | [13.534129, 16.489644] |
| validation | B | moderate | MI-FST | 1975 | 1846 | 2171 | 5.941962 | 5.958330 | 1.342930 | [4.979190, 6.951633] |
| validation | B | moderate | MI-C | 1975 | 1846 | 2171 | 5.941962 | 5.958330 | 1.342930 | [4.942250, 6.925217] |
| validation | B | moderate | MI-M | 1975 | 1861 | 2171 | 5.251036 | 5.263759 | 1.236223 | [4.335793, 6.229876] |
| validation | B | moderate | MI-MD | 1975 | 1861 | 2171 | 5.251036 | 5.263759 | 1.236223 | [4.347826, 6.215226] |
| validation | B | moderate | MI-MDA | 1975 | 1957 | 2171 | 0.829111 | 0.834664 | 0.565466 | [0.465549, 1.235127] |
| validation | B | severe | FST-original | 774 | 545 | 931 | 24.597207 | 24.564581 | 1.212384 | [21.630359, 27.657379] |
| validation | B | severe | MI-FST | 774 | 742 | 931 | 3.437164 | 3.404806 | 1.137862 | [1.649442, 5.291050] |
| validation | B | severe | MI-C | 774 | 742 | 931 | 3.437164 | 3.404806 | 1.137862 | [1.657322, 5.229456] |
| validation | B | severe | MI-M | 774 | 685 | 931 | 9.559613 | 9.531642 | 2.430204 | [7.668067, 11.471861] |
| validation | B | severe | MI-MD | 774 | 685 | 931 | 9.559613 | 9.531642 | 2.430204 | [7.773056, 11.503420] |
| validation | B | severe | MI-MDA | 774 | 738 | 931 | 3.866810 | 3.846620 | 1.844416 | [2.666598, 5.139665] |
