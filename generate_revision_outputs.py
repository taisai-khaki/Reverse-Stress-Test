#!/usr/bin/env python3
"""
Generate the manuscript-reproduction audit and revised simulation tables.

The script intentionally gates the revised outputs on exact reproduction of the
submitted results listed in the reviewer-response audit request.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from paper_rst_ijpr import (
    MODERATE,
    SEVERE,
    Event,
    FailureThreshold,
    NetworkConfig,
    Pathway,
    PathwayEval,
    classify_mechanism,
    fst_enhanced_library,
    fst_library,
    generate_random_pathway,
    mechanism_guided_rst_search,
    monte_carlo_ground_truth,
    pathway_elements,
    rst_candidate_library,
    run_fst,
    run_simulation,
    sample_duration_negative_binomial,
    threshold_cross_info,
    truncated_beta,
)


ELEMENTS = ["T2P", "corridor", "S1A", "S1B", "S1C"]

HCD = "Hidden convergent dependency"
CMD = "Compounding moderate disturbance"
PAC = "Propagation-amplified cascade"
RL_TO = "Recovery lag and temporal overlap"
ALF = "Adaptation-limited failure"
ADAPTATION_LIMITED = "Adaptation-limited"
STRUCTURAL = "Structural"
CD = "Concentrated dependency"

FAMILY_ORDER = [HCD, CMD, PAC, RL_TO, CD]


def _fst_group(category: str) -> str:
    lower_category = category.lower()
    if "single tier-1 failure" in lower_category:
        return "Individual tier-1 failures"
    if "t2p partial failure" in lower_category:
        return "T2P partial failures"
    if "corridor disruption" in lower_category:
        return "Corridor disruptions"
    if "demand spike" in lower_category:
        return "Demand spikes"
    return "Prespecified combinations"


def _family_intervention(family: str) -> str:
    interventions = {
        CD: "Diversification, redundancy, strategic inventory, or backup capacity.",
        HCD: "Upstream T2P qualification and alternative processor sourcing.",
        CMD: "Slack restoration and stress-triggered allocation.",
        PAC: "Segmentation and containment redesign.",
        RL_TO: "Recovery sequencing and time-buffered restoration.",
    }
    return interventions.get(family, "")


def _rank_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
    score_series = pd.Series(scores).sort_values(ascending=False)
    return score_series.rank(ascending=False, method="dense").astype(int).to_dict()


def _ordinal(rank: int) -> str:
    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    minimum_value = min(values)
    maximum_value = max(values)
    if math.isclose(minimum_value, maximum_value):
        return [1.0 for _ in values]
    return [(value - minimum_value) / (maximum_value - minimum_value) for value in values]


def _round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def _evals_to_dataframe(pathway_evals: Sequence[PathwayEval]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pathway_id": pathway_eval.pathway_id,
                "family": pathway_eval.family,
                "passive_failure": bool(pathway_eval.passive_failure),
                "active_failure": (
                    None if pathway_eval.active_failure is None else bool(pathway_eval.active_failure)
                ),
                "classification": pathway_eval.classification,
                "severity": float(pathway_eval.severity),
                "plausibility": float(pathway_eval.plausibility),
                "controllability": float(pathway_eval.controllability),
                "weight": float(pathway_eval.weight),
                "elements": ",".join(pathway_eval.elements),
            }
            for pathway_eval in pathway_evals
        ]
    )


def _compute_fst_element_scores(fst_df: pd.DataFrame, scenarios: Sequence[Pathway]) -> Dict[str, float]:
    crossing_ids = set(fst_df.loc[fst_df["moderate_crossing"], "scenario_id"].tolist())
    scores = {element: 0.0 for element in ELEMENTS}
    for scenario in scenarios:
        if scenario.pathway_id not in crossing_ids:
            continue
        for event in scenario.events:
            if event.target in scores:
                scores[event.target] += 1.0
    return scores


def _compute_pathway_element_scores(
    pathway_df: pd.DataFrame,
    scenarios: Sequence[Pathway],
    crossing_col: str = "moderate_crossing",
) -> Dict[str, float]:
    crossing_ids = set(pathway_df.loc[pathway_df[crossing_col], "scenario_id"].tolist())
    scores = {element: 0.0 for element in ELEMENTS}
    for scenario in scenarios:
        if scenario.pathway_id not in crossing_ids:
            continue
        for element in pathway_elements(scenario):
            scores[element] += 1.0
    return scores


def _compute_mc_element_frequencies(mc_df: pd.DataFrame) -> Dict[str, float]:
    moderate_failures = mc_df[mc_df["moderate_passive_failure"]].copy()
    if moderate_failures.empty:
        return {element: 0.0 for element in ELEMENTS}
    return {
        element: float(moderate_failures[f"has_{element}"].mean())
        for element in ELEMENTS
    }


def _coverage(discovered_families: Set[str], failures_df: pd.DataFrame) -> float:
    if failures_df.empty:
        return 0.0
    return float(failures_df["primary_family"].isin(discovered_families).mean())


def _coverage_excluding_alf(discovered_families: Set[str], failures_df: pd.DataFrame) -> float:
    non_alf_failures = failures_df[failures_df["primary_family"] != ALF]
    return _coverage(discovered_families, non_alf_failures)


def _coverage_for_families(
    discovered_families: Set[str],
    failures_df: pd.DataFrame,
    family_subset: Set[str],
) -> float:
    subset_failures = failures_df[failures_df["primary_family"].isin(family_subset)]
    return _coverage(discovered_families, subset_failures)


def _active_failure_count(rst_df: pd.DataFrame) -> int:
    return int(
        sum(
            False if pd.isna(active_failure) else bool(active_failure)
            for active_failure in rst_df["active_failure"].tolist()
        )
    )


def _passive_failure_count(rst_df: pd.DataFrame) -> int:
    return int(rst_df["passive_failure"].astype(bool).sum())


def _build_core_results(replications: int, seed: int, budget: int) -> Dict[str, object]:
    cfg = NetworkConfig()
    mc_df = monte_carlo_ground_truth(replications=replications, cfg=cfg, seed=seed)
    fst_df = run_fst(fst_library(), cfg=cfg)
    rst_evals, rst_scores = mechanism_guided_rst_search(
        candidates=rst_candidate_library(),
        cfg=cfg,
        budget=budget,
        threshold=MODERATE,
    )
    rst_severe_evals, _ = mechanism_guided_rst_search(
        candidates=rst_candidate_library(),
        cfg=cfg,
        budget=budget,
        threshold=SEVERE,
    )

    rst_df = _evals_to_dataframe(rst_evals)
    rst_severe_df = _evals_to_dataframe(rst_severe_evals)
    moderate_failures = mc_df[mc_df["moderate_passive_failure"]].copy()
    severe_failures = mc_df[mc_df["severe_passive_failure"]].copy()

    fst_scores = _compute_fst_element_scores(fst_df, fst_library())
    mc_frequencies = _compute_mc_element_frequencies(mc_df)
    fst_rank = _rank_from_scores(fst_scores)
    rst_rank = _rank_from_scores(rst_scores)
    mc_rank = _rank_from_scores(mc_frequencies)

    fst_tau, fst_tau_pvalue = kendalltau(
        [fst_scores[element] for element in ELEMENTS],
        [mc_frequencies[element] for element in ELEMENTS],
    )
    rst_tau, rst_tau_pvalue = kendalltau(
        [rst_scores[element] for element in ELEMENTS],
        [mc_frequencies[element] for element in ELEMENTS],
    )

    return {
        "cfg": cfg,
        "mc": mc_df,
        "fst": fst_df,
        "rst_evals": rst_evals,
        "rst": rst_df,
        "rst_severe": rst_severe_df,
        "moderate_failures": moderate_failures,
        "severe_failures": severe_failures,
        "fst_scores": fst_scores,
        "rst_scores": rst_scores,
        "mc_frequencies": mc_frequencies,
        "fst_rank": fst_rank,
        "rst_rank": rst_rank,
        "mc_rank": mc_rank,
        "fst_tau": float(fst_tau),
        "fst_tau_pvalue": float(fst_tau_pvalue),
        "rst_tau": float(rst_tau),
        "rst_tau_pvalue": float(rst_tau_pvalue),
        "replications": replications,
        "seed": seed,
        "budget": budget,
    }


def _write_audit_tables(core_results: Dict[str, object], audit_dir: Path) -> Dict[str, object]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    cfg = core_results["cfg"]
    fst_df = core_results["fst"].copy()
    rst_df = core_results["rst"].copy()
    rst_severe_df = core_results["rst_severe"].copy()
    mc_df = core_results["mc"].copy()

    table4_input = fst_df.copy()
    table4_input["group"] = table4_input["category"].map(_fst_group)
    table4 = (
        table4_input.groupby("group", as_index=False)
        .agg(
            scenarios=("scenario_id", "count"),
            moderate_crossings=("moderate_crossing", "sum"),
            severe_crossings=("severe_crossing", "sum"),
        )
        .sort_values("group")
    )
    _write_csv(table4, audit_dir / "table4_fst_original.csv")

    table5_rows = []
    for family in FAMILY_ORDER:
        moderate_family = rst_df[rst_df["family"] == family]
        severe_family = rst_severe_df[rst_severe_df["family"] == family]
        table5_rows.append(
            {
                "family": family,
                "moderate_passive": _passive_failure_count(moderate_family),
                "moderate_active": _active_failure_count(moderate_family),
                "severe_passive": _passive_failure_count(severe_family),
                "severe_active": _active_failure_count(severe_family),
                "intervention": _family_intervention(family),
            }
        )
    table5 = pd.DataFrame(table5_rows)
    _write_csv(table5, audit_dir / "table5_rst_pathways.csv")

    fst_discovered = set(fst_df.loc[fst_df["moderate_crossing"], "assigned_family"].tolist())
    rst_discovered = set(rst_df.loc[rst_df["passive_failure"], "family"].tolist())
    moderate_failures = core_results["moderate_failures"]
    severe_failures = core_results["severe_failures"]

    table6 = pd.DataFrame(
        [
            {"metric": "Computational budget", "FST": core_results["budget"], "RST": core_results["budget"]},
            {
                "metric": "Threshold-crossing pathways identified: moderate",
                "FST": int(fst_df["moderate_crossing"].sum()),
                "RST": _passive_failure_count(rst_df),
            },
            {
                "metric": "Threshold-crossing pathways identified: severe",
                "FST": int(fst_df["severe_crossing"].sum()),
                "RST": _passive_failure_count(rst_severe_df),
            },
            {
                "metric": "Coverage of Monte Carlo failures: moderate",
                "FST": _coverage(fst_discovered, moderate_failures),
                "RST": _coverage(rst_discovered, moderate_failures),
            },
            {
                "metric": "Coverage of Monte Carlo failures: severe",
                "FST": _coverage(fst_discovered, severe_failures),
                "RST": _coverage(rst_discovered, severe_failures),
            },
            {
                "metric": "Hidden convergence pathways identified",
                "FST": int(((fst_df["assigned_family"] == HCD) & fst_df["moderate_crossing"]).sum()),
                "RST": int(((rst_df["family"] == HCD) & rst_df["passive_failure"]).sum()),
            },
            {
                "metric": "Compounding moderate-disturbance pathways identified",
                "FST": int(((fst_df["assigned_family"] == CMD) & fst_df["moderate_crossing"]).sum()),
                "RST": int(((rst_df["family"] == CMD) & rst_df["passive_failure"]).sum()),
            },
        ]
    )
    _write_csv(table6, audit_dir / "table6_current_comparison.csv")

    table7_rows = []
    for element in ELEMENTS:
        table7_rows.append(
            {
                "element": element,
                "RST_IF": float(core_results["rst_scores"].get(element, 0.0)),
                "RST_rank": int(core_results["rst_rank"].get(element, 0)),
                "FST_IF": float(core_results["fst_scores"].get(element, 0.0)),
                "FST_rank": int(core_results["fst_rank"].get(element, 0)),
                "MC_frequency": float(core_results["mc_frequencies"].get(element, 0.0)),
                "MC_rank": int(core_results["mc_rank"].get(element, 0)),
            }
        )
    _write_csv(pd.DataFrame(table7_rows), audit_dir / "table7_criticality.csv")

    thresholds = [
        FailureThreshold("SL<90% >=1 period", 0.90, 1),
        FailureThreshold("SL<90% >=2 periods", 0.90, 2),
        FailureThreshold("SL<85% >=3 periods", 0.85, 3),
        FailureThreshold("SL<80% >=4 periods", 0.80, 4),
    ]
    table8_rows = []
    for threshold in thresholds:
        threshold_evals, scores = mechanism_guided_rst_search(
            candidates=rst_candidate_library(),
            cfg=cfg,
            budget=int(core_results["budget"]),
            threshold=threshold,
        )
        passive_count = sum(pathway_eval.passive_failure for pathway_eval in threshold_evals)
        family_counts = Counter(
            pathway_eval.family
            for pathway_eval in threshold_evals
            if pathway_eval.passive_failure
        )
        dominant_mechanism = family_counts.most_common(1)[0][0] if family_counts else "None"
        table8_rows.append(
            {
                "threshold": threshold.name,
                "passive_pathways": int(passive_count),
                "T2P_IF": float(scores.get("T2P", 0.0)),
                "dominant_mechanism": dominant_mechanism,
                "intervention_implication": _family_intervention(dominant_mechanism),
            }
        )
    _write_csv(pd.DataFrame(table8_rows), audit_dir / "table8_threshold_sensitivity.csv")

    checks = _build_reproduction_checks(core_results)
    log = {
        "can_revise": all(check["passed"] for check in checks),
        "parameters": {
            "replications": core_results["replications"],
            "seed": core_results["seed"],
            "budget": core_results["budget"],
            "horizon_weeks": cfg.horizon_weeks,
        },
        "checks": checks,
        "files": [
            "table4_fst_original.csv",
            "table5_rst_pathways.csv",
            "table6_current_comparison.csv",
            "table7_criticality.csv",
            "table8_threshold_sensitivity.csv",
            "reproduction_log.json",
        ],
    }
    with (audit_dir / "reproduction_log.json").open("w", encoding="utf-8") as output_file:
        json.dump(log, output_file, indent=2)

    return log


def _build_reproduction_checks(core_results: Dict[str, object]) -> List[Dict[str, object]]:
    mc_df = core_results["mc"]
    fst_df = core_results["fst"]
    rst_df = core_results["rst"]
    rst_severe_df = core_results["rst_severe"]

    rst_t2p_rank = int(core_results["rst_rank"]["T2P"])
    fst_t2p_rank = int(core_results["fst_rank"]["T2P"])
    rst_tau = float(core_results["rst_tau"])
    fst_tau = float(core_results["fst_tau"])

    check_specs = [
        ("Monte Carlo moderate failures", "240 / 1000", int(mc_df["moderate_passive_failure"].sum()), 240),
        ("Monte Carlo severe failures", "124 / 1000", int(mc_df["severe_passive_failure"].sum()), 124),
        ("FST-original moderate crossings", 7, int(fst_df["moderate_crossing"].sum()), 7),
        ("FST-original severe crossings", 3, int(fst_df["severe_crossing"].sum()), 3),
        ("RST moderate passive pathways", 21, _passive_failure_count(rst_df), 21),
        ("RST moderate active pathways", 11, _active_failure_count(rst_df), 11),
        ("RST severe passive pathways", 5, _passive_failure_count(rst_severe_df), 5),
        ("RST severe active pathways", 0, _active_failure_count(rst_severe_df), 0),
    ]
    checks = [
        {
            "check": check_name,
            "required": required_display,
            "actual": actual_value,
            "passed": actual_value == expected_value,
        }
        for check_name, required_display, actual_value, expected_value in check_specs
    ]

    checks.append(
        {
            "check": "T2P rank",
            "required": "1st for both FST and RST",
            "actual": {
                "FST": _ordinal(fst_t2p_rank),
                "RST": _ordinal(rst_t2p_rank),
            },
            "passed": fst_t2p_rank == 1 and rst_t2p_rank == 1,
        }
    )
    checks.append(
        {
            "check": "Kendall tau_b",
            "required": "0.882 for both methods",
            "actual": {
                "FST": _round_metric(fst_tau, 3),
                "RST": _round_metric(rst_tau, 3),
                "FST_full_precision": fst_tau,
                "RST_full_precision": rst_tau,
            },
            "passed": round(fst_tau, 3) == 0.882 and round(rst_tau, 3) == 0.882,
        }
    )
    return checks


def _enhanced_fst_results(cfg: NetworkConfig) -> Tuple[pd.DataFrame, List[Pathway]]:
    enhanced_scenarios = fst_enhanced_library()
    return run_fst(enhanced_scenarios, cfg=cfg), enhanced_scenarios


def _method_family_sets(
    core_results: Dict[str, object],
    enhanced_fst_df: pd.DataFrame,
) -> Dict[str, Set[str]]:
    fst_df = core_results["fst"]
    rst_df = core_results["rst"]
    return {
        "FST-original": set(fst_df.loc[fst_df["moderate_crossing"], "assigned_family"].tolist()),
        "FST-enhanced-50": set(enhanced_fst_df.loc[enhanced_fst_df["moderate_crossing"], "assigned_family"].tolist()),
        "RST": set(rst_df.loc[rst_df["passive_failure"], "family"].tolist()),
    }


def _mc_multilabel_sets(moderate_failures: pd.DataFrame) -> List[Set[str]]:
    label_sets = []
    for _, failure_row in moderate_failures.iterrows():
        labels = {str(failure_row["primary_family"])}
        if bool(failure_row["moderate_passive_failure"]) and not bool(failure_row["moderate_active_failure"]):
            labels.add(ALF)
        if bool(failure_row["has_T2P"]):
            labels.add(HCD)
        if int(failure_row["event_count"]) >= 2:
            labels.add(CMD)
        if bool(failure_row["has_T2P"]) and float(failure_row["max_backlog_passive"]) >= 0.30 * 900.0:
            labels.add(PAC)
        has_non_t2p_structural = any(
            bool(failure_row[f"has_{element}"])
            for element in ["corridor", "S1A", "S1B", "S1C"]
        )
        if has_non_t2p_structural and not bool(failure_row["has_T2P"]):
            labels.add(CD)
        label_sets.append(labels)
    return label_sets


def _label_weighted_coverage(discovered_families: Set[str], label_sets: Sequence[Set[str]]) -> float:
    total_labels = sum(len(label_set) for label_set in label_sets)
    if total_labels == 0:
        return 0.0
    covered_labels = sum(len(label_set & discovered_families) for label_set in label_sets)
    return float(covered_labels / total_labels)


def _reweighted_scores(
    pathway_evals: Sequence[PathwayEval],
    weights: Tuple[float, float, float],
) -> Tuple[str, str, Dict[str, float], bool]:
    severity_weight, plausibility_weight, controllability_weight = weights
    passive_evals = [pathway_eval for pathway_eval in pathway_evals if pathway_eval.passive_failure]
    severity_values = _normalize([pathway_eval.severity for pathway_eval in passive_evals])
    plausibility_values = _normalize([pathway_eval.plausibility for pathway_eval in passive_evals])
    controllability_values = _normalize([1.0 - pathway_eval.controllability for pathway_eval in passive_evals])

    weighted_rows = []
    scores = {element: 0.0 for element in ELEMENTS}
    for pathway_eval, severity_value, plausibility_value, controllability_value in zip(
        passive_evals,
        severity_values,
        plausibility_values,
        controllability_values,
    ):
        pathway_weight = (
            severity_weight * severity_value
            + plausibility_weight * plausibility_value
            + controllability_weight * controllability_value
        )
        weighted_rows.append((pathway_eval, pathway_weight))
        for element in pathway_eval.elements:
            scores[element] += pathway_weight

    total_score = sum(scores.values())
    if total_score > 0:
        scores = {element: score / total_score for element, score in scores.items()}

    top_pathway = max(weighted_rows, key=lambda weighted_row: weighted_row[1])[0]
    ranked_elements = sorted(scores, key=scores.get, reverse=True)
    rank_stable = ranked_elements[:2] == ["T2P", "corridor"]
    return top_pathway.family, top_pathway.classification, scores, rank_stable


def _write_revised_tables(
    core_results: Dict[str, object],
    revision_dir: Path,
    disturbance_replications: int,
) -> None:
    revision_dir.mkdir(parents=True, exist_ok=True)
    cfg = core_results["cfg"]
    fst_df = core_results["fst"]
    rst_df = core_results["rst"]
    moderate_failures = core_results["moderate_failures"]
    enhanced_fst_df, enhanced_scenarios = _enhanced_fst_results(cfg)
    method_families = _method_family_sets(core_results, enhanced_fst_df)

    table_a_rows = [
        {
            "Metric": "Moderate crossings",
            "FST-original": int(fst_df["moderate_crossing"].sum()),
            "FST-enhanced-50": int(enhanced_fst_df["moderate_crossing"].sum()),
        },
        {
            "Metric": "Severe crossings",
            "FST-original": int(fst_df["severe_crossing"].sum()),
            "FST-enhanced-50": int(enhanced_fst_df["severe_crossing"].sum()),
        },
    ]
    for short_label, family in [("HCD", HCD), ("CMD", CMD), ("PAC", PAC), ("RL-TO", RL_TO), ("CD", CD)]:
        table_a_rows.append(
            {
                "Metric": f"{short_label} pathways",
                "FST-original": int(((fst_df["assigned_family"] == family) & fst_df["moderate_crossing"]).sum()),
                "FST-enhanced-50": int(((enhanced_fst_df["assigned_family"] == family) & enhanced_fst_df["moderate_crossing"]).sum()),
            }
        )
    _write_csv(pd.DataFrame(table_a_rows), revision_dir / "tableA_fst_original_vs_enhanced.csv")

    enhanced_scores = _compute_pathway_element_scores(enhanced_fst_df, enhanced_scenarios)
    enhanced_tau, _ = kendalltau(
        [enhanced_scores[element] for element in ELEMENTS],
        [core_results["mc_frequencies"][element] for element in ELEMENTS],
    )
    cmd_pac_rlto = {CMD, PAC, RL_TO}
    passive_active_summary = (
        f"{_active_failure_count(rst_df)} structural / "
        f"{_passive_failure_count(rst_df) - _active_failure_count(rst_df)} adaptation-limited"
    )
    table_b = pd.DataFrame(
        [
            {
                "Metric": "Structural ranking agreement",
                "FST-original": _round_metric(core_results["fst_tau"], 3),
                "FST-enhanced-50": _round_metric(enhanced_tau, 3),
                "RST": _round_metric(core_results["rst_tau"], 3),
            },
            {
                "Metric": "Mechanism coverage including ALF",
                "FST-original": _round_metric(_coverage(method_families["FST-original"], moderate_failures)),
                "FST-enhanced-50": _round_metric(_coverage(method_families["FST-enhanced-50"], moderate_failures)),
                "RST": _round_metric(_coverage(method_families["RST"], moderate_failures)),
            },
            {
                "Metric": "Mechanism coverage excluding ALF",
                "FST-original": _round_metric(_coverage_excluding_alf(method_families["FST-original"], moderate_failures)),
                "FST-enhanced-50": _round_metric(_coverage_excluding_alf(method_families["FST-enhanced-50"], moderate_failures)),
                "RST": _round_metric(_coverage_excluding_alf(method_families["RST"], moderate_failures)),
            },
            {
                "Metric": "CMD/PAC/RL-TO coverage",
                "FST-original": _round_metric(_coverage_for_families(method_families["FST-original"], moderate_failures, cmd_pac_rlto)),
                "FST-enhanced-50": _round_metric(_coverage_for_families(method_families["FST-enhanced-50"], moderate_failures, cmd_pac_rlto)),
                "RST": _round_metric(_coverage_for_families(method_families["RST"], moderate_failures, cmd_pac_rlto)),
            },
            {
                "Metric": "Passive–active classification",
                "FST-original": "N/A",
                "FST-enhanced-50": "N/A",
                "RST": passive_active_summary,
            },
        ]
    )
    _write_csv(table_b, revision_dir / "tableB_corrected_comparison.csv")

    label_sets = _mc_multilabel_sets(moderate_failures)
    table_c_rows = []
    for method_name in ["FST-original", "FST-enhanced-50", "RST"]:
        discovered = method_families[method_name]
        table_c_rows.append(
            {
                "Method": method_name,
                "Primary-label coverage": _round_metric(_coverage(discovered, moderate_failures)),
                "Multi-label coverage": _round_metric(_label_weighted_coverage(discovered, label_sets)),
                "ALF-excluded coverage": _round_metric(_coverage_excluding_alf(discovered, moderate_failures)),
            }
        )
    _write_csv(pd.DataFrame(table_c_rows), revision_dir / "tableC_multilabel_coverage.csv")

    response_settings = [
        (0.10, 200.0, 0.30),
        (0.17, 0.0, 0.30),
        (0.17, 200.0, 0.15),
        (0.17, 200.0, 0.30),
        (0.17, 500.0, 0.30),
        (0.17, 200.0, 0.50),
        (0.25, 200.0, 0.30),
    ]
    table_d_rows = []
    for tau_b, kappa_alt, rho_rel in response_settings:
        sensitivity_cfg = NetworkConfig(
            active_backlog_trigger_ratio=tau_b,
            active_alt_corridor_units=kappa_alt,
            active_reserve_release_frac=rho_rel,
        )
        sensitivity_evals, _ = mechanism_guided_rst_search(
            candidates=rst_candidate_library(),
            cfg=sensitivity_cfg,
            budget=int(core_results["budget"]),
            threshold=MODERATE,
        )
        sensitivity_df = _evals_to_dataframe(sensitivity_evals)
        passive_pathways = _passive_failure_count(sensitivity_df)
        active_pathways = _active_failure_count(sensitivity_df)
        table_d_rows.append(
            {
                "τ_B": tau_b,
                "κ_alt": kappa_alt,
                "ρ_rel": rho_rel,
                "Passive pathways": passive_pathways,
                "Active pathways": active_pathways,
                "Adaptation-limited pathways": passive_pathways - active_pathways,
            }
        )
    _write_csv(pd.DataFrame(table_d_rows), revision_dir / "tableD_response_sensitivity.csv")

    weight_settings = [
        (0.50, 0.30, 0.20),
        (0.60, 0.20, 0.20),
        (0.40, 0.40, 0.20),
        (0.40, 0.20, 0.40),
        (0.33, 0.33, 0.34),
        (0.20, 0.60, 0.20),
        (0.20, 0.20, 0.60),
    ]
    table_e_rows = []
    for omega_c, omega_p, omega_gamma in weight_settings:
        top_family, top_classification, scores, rank_stable = _reweighted_scores(
            core_results["rst_evals"],
            (omega_c, omega_p, omega_gamma),
        )
        table_e_rows.append(
            {
                "ω_C": omega_c,
                "ω_P": omega_p,
                "ω_γ": omega_gamma,
                "Top mechanism family": top_family,
                "Top classification": top_classification,
                "T2P IF": _round_metric(scores.get("T2P", 0.0)),
                "Corridor IF": _round_metric(scores.get("corridor", 0.0)),
                "Rank stable?": bool(rank_stable),
            }
        )
    _write_csv(pd.DataFrame(table_e_rows), revision_dir / "tableE_weight_sensitivity.csv")

    table_f_rows = []
    baseline_active_pathways = _active_failure_count(rst_df)
    for disruption_model in [
        "Independent baseline",
        "Shared upstream shock",
        "Supplier-cluster shock",
        "Demand-supply coupling",
    ]:
        model_summary = _simulate_disturbance_model(
            disruption_model=disruption_model,
            replications=disturbance_replications,
            seed=int(core_results["seed"]),
            baseline_mc=core_results["mc"] if disruption_model == "Independent baseline" else None,
        )
        table_f_rows.append(
            {
                "Disruption model": disruption_model,
                "Moderate failures": model_summary["moderate_failures"],
                "Severe failures": model_summary["severe_failures"],
                "Dominant mechanism family": model_summary["dominant_mechanism_family"],
                "Dominant classification": model_summary["dominant_classification"],
                "RST active pathways": baseline_active_pathways,
                "Adaptation-limited share": _round_metric(model_summary["adaptation_limited_share"]),
            }
        )
    _write_csv(pd.DataFrame(table_f_rows), revision_dir / "tableF_disturbance_correlation.csv")

    manifest = {
        "parameters": {
            "replications": core_results["replications"],
            "disturbance_replications": disturbance_replications,
            "seed": core_results["seed"],
            "budget": core_results["budget"],
        },
        "notes": {
            "FST-enhanced-50": "Uses the same 50-scenario budget as FST-original, with a mechanism-balanced passive scenario library.",
            "Multi-label coverage": "Label-weighted coverage across inferred mechanism labels for Monte Carlo moderate failures.",
            "Response sensitivity": "τ_B is the active backlog trigger ratio used by NetworkConfig.",
        },
        "files": sorted(output_file.name for output_file in revision_dir.glob("table*.csv")),
    }
    with (revision_dir / "revision_manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)


def _conditional_target(
    rng: np.random.Generator,
    previous_target: str | None,
    disruption_model: str,
) -> str:
    targets = ["T2P", "corridor", "S1A", "S1B", "S1C", "demand"]
    if disruption_model == "Shared upstream shock":
        if previous_target in {"T2P", "corridor"}:
            probabilities = np.array([0.35, 0.30, 0.08, 0.08, 0.08, 0.11])
        else:
            probabilities = np.array([0.32, 0.24, 0.10, 0.10, 0.10, 0.14])
    elif disruption_model == "Supplier-cluster shock":
        if previous_target in {"S1A", "S1B", "S1C"}:
            probabilities = np.array([0.18, 0.16, 0.20, 0.20, 0.20, 0.06])
        else:
            probabilities = np.array([0.20, 0.16, 0.18, 0.18, 0.18, 0.10])
    elif disruption_model == "Demand-supply coupling":
        if previous_target == "demand":
            probabilities = np.array([0.34, 0.28, 0.08, 0.08, 0.08, 0.14])
        else:
            probabilities = np.array([0.26, 0.18, 0.10, 0.10, 0.10, 0.26])
    else:
        probabilities = np.array([0.28, 0.18, 0.14, 0.14, 0.14, 0.12])
    probabilities = probabilities / probabilities.sum()
    return str(rng.choice(targets, p=probabilities))


def _generate_correlated_pathway(
    rng: np.random.Generator,
    cfg: NetworkConfig,
    disruption_model: str,
) -> List[Event]:
    if disruption_model == "Independent baseline":
        events, _, _, _ = generate_random_pathway(rng, cfg)
        return events

    events: List[Event] = []
    start_week = int(rng.integers(0, cfg.mc_start_week_max + 1))
    max_events = 3
    interarrival_mean = {
        "Shared upstream shock": 8.0,
        "Supplier-cluster shock": 10.0,
        "Demand-supply coupling": 9.0,
    }[disruption_model]
    previous_target: str | None = None

    while start_week < cfg.horizon_weeks and len(events) < max_events:
        target = _conditional_target(rng, previous_target, disruption_model)
        magnitude = float(truncated_beta(rng, size=1)[0])
        if disruption_model == "Shared upstream shock" and target in {"T2P", "corridor"}:
            magnitude = min(0.80, magnitude * 1.10)
        if target == "demand":
            demand_multiplier = 1.15 if disruption_model == "Demand-supply coupling" else 1.0
            magnitude = float(np.clip(magnitude * demand_multiplier, 0.10, 0.50))
        duration = sample_duration_negative_binomial(rng)
        recovery = int(rng.integers(1, 5))
        events.append(
            Event(
                target=target,
                magnitude=magnitude,
                start_week=start_week,
                duration_weeks=duration,
                recovery_weeks=recovery,
            )
        )
        previous_target = target
        start_week += int(max(1, round(rng.exponential(interarrival_mean))))

    return events


def _simulate_disturbance_model(
    disruption_model: str,
    replications: int,
    seed: int,
    baseline_mc: pd.DataFrame | None = None,
) -> Dict[str, object]:
    if baseline_mc is not None and replications == len(baseline_mc):
        moderate_failures = baseline_mc[baseline_mc["moderate_passive_failure"]].copy()
        mechanism_failures = moderate_failures[moderate_failures["primary_family"] != ALF]
        family_counts = mechanism_failures["primary_family"].value_counts()
        dominant_mechanism_family = family_counts.index[0] if len(family_counts) else "None"
        if "passive_active_classification" in moderate_failures.columns:
            classification_counts = moderate_failures["passive_active_classification"].value_counts()
            adaptation_limited_share = (
                float((moderate_failures["passive_active_classification"] == ADAPTATION_LIMITED).mean())
                if len(moderate_failures)
                else 0.0
            )
        else:
            classification = np.where(moderate_failures["primary_family"] == ALF, ADAPTATION_LIMITED, STRUCTURAL)
            classification_counts = pd.Series(classification).value_counts()
            adaptation_limited_share = float((classification == ADAPTATION_LIMITED).mean()) if len(classification) else 0.0
        dominant_classification = classification_counts.index[0] if len(classification_counts) else "None"
        return {
            "moderate_failures": int(len(moderate_failures)),
            "severe_failures": int(baseline_mc["severe_passive_failure"].sum()),
            "dominant_mechanism_family": dominant_mechanism_family,
            "dominant_classification": dominant_classification,
            "adaptation_limited_share": adaptation_limited_share,
        }

    cfg = NetworkConfig()
    root_seed = np.random.SeedSequence(seed)
    child_seeds = root_seed.spawn(replications)
    moderate_count = 0
    severe_count = 0
    family_counter: Counter[str] = Counter()
    classification_counter: Counter[str] = Counter()

    for replication_index in range(replications):
        rng = np.random.default_rng(child_seeds[replication_index])
        events = _generate_correlated_pathway(rng, cfg, disruption_model)
        passive_result = run_simulation(events, cfg=cfg, active_mode=False)
        active_result = run_simulation(events, cfg=cfg, active_mode=True)
        moderate_crossing, _ = threshold_cross_info(passive_result.service_levels, MODERATE)
        active_crossing, _ = threshold_cross_info(active_result.service_levels, MODERATE)
        severe_crossing, _ = threshold_cross_info(passive_result.service_levels, SEVERE)
        if severe_crossing:
            severe_count += 1
        if not moderate_crossing:
            continue
        moderate_count += 1
        classification = STRUCTURAL if active_crossing else ADAPTATION_LIMITED
        classification_counter[classification] += 1
        mechanism_family = classify_mechanism(
            events=events,
            cfg=cfg,
            passive_result=passive_result,
            active_result=passive_result,
            threshold=MODERATE,
        )
        family_counter[mechanism_family] += 1

    dominant_mechanism_family = family_counter.most_common(1)[0][0] if family_counter else "None"
    dominant_classification = classification_counter.most_common(1)[0][0] if classification_counter else "None"
    adaptation_limited_share = classification_counter[ADAPTATION_LIMITED] / moderate_count if moderate_count else 0.0
    return {
        "moderate_failures": moderate_count,
        "severe_failures": severe_count,
        "dominant_mechanism_family": dominant_mechanism_family,
        "dominant_classification": dominant_classification,
        "adaptation_limited_share": adaptation_limited_share,
    }


def generate_revision_outputs(
    audit_dir: Path,
    revision_dir: Path,
    replications: int,
    disturbance_replications: int,
    seed: int,
    budget: int,
    audit_only: bool = False,
) -> Dict[str, object]:
    core_results = _build_core_results(replications=replications, seed=seed, budget=budget)
    reproduction_log = _write_audit_tables(core_results, audit_dir=audit_dir)
    if not reproduction_log["can_revise"]:
        return reproduction_log
    if not audit_only:
        _write_revised_tables(
            core_results=core_results,
            revision_dir=revision_dir,
            disturbance_replications=disturbance_replications,
        )
    return reproduction_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproduction audit files and revised simulation outputs."
    )
    parser.add_argument("--audit-dir", type=Path, default=Path("00_reproduce_submitted_results"))
    parser.add_argument("--revision-dir", type=Path, default=Path("01_revised_simulation_outputs"))
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--disturbance-replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    reproduction_log = generate_revision_outputs(
        audit_dir=args.audit_dir,
        revision_dir=args.revision_dir,
        replications=args.replications,
        disturbance_replications=args.disturbance_replications,
        seed=args.seed,
        budget=args.budget,
        audit_only=args.audit_only,
    )

    print(f"Reproduction audit written to: {args.audit_dir.resolve()}")
    if not reproduction_log["can_revise"]:
        print("Submitted results did not reproduce; revised outputs were not generated.")
        raise SystemExit(1)
    if not args.audit_only:
        print(f"Revised tables written to: {args.revision_dir.resolve()}")
    print("All reproduction checks passed.")


if __name__ == "__main__":
    main()
