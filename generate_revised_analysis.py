#!/usr/bin/env python3
"""Generate the validation, primary, sensitivity, robustness, and runtime package."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rst_revised_model import (
    ADAPTATION_LIMITED,
    CD,
    CMD,
    HCD,
    MECHANISM_FAMILIES,
    MODERATE,
    NON_FAILURE,
    PAC,
    PRIMARY_MECHANISM_PRIORITY,
    RL_TO,
    SEVERE,
    STRUCTURAL,
    UNCLASSIFIED,
    Event,
    FailureThreshold,
    NetworkConfig,
    Pathway,
    PathwayEval,
    attribute_mechanisms,
    compute_criticality,
    fst_enhanced_library,
    fst_library,
    mechanism_guided_rst_search,
    pathway_evaluations_dataframe,
    pathway_elements,
    rst_candidate_library,
    run_fst,
    run_simulation,
    simulate_monte_carlo,
    threshold_cross_info,
    topology_a_config,
    topology_b_config,
)


DEFAULT_SEEDS = (42, 314, 1618, 2026, 2718)
DEPENDENCE_MODELS = (
    "Independent baseline",
    "Shared upstream shock",
    "Supplier-cluster shock",
    "Demand-supply coupling",
)
CMD_PAC_RLTO = {CMD, PAC, RL_TO}
ELEMENTS = ("T2P", "corridor", "S1A", "S1B", "S1C")
RESPONSE_SENSITIVITY_SETTINGS = (
    (0.10, 200.0, 0.30),
    (0.17, 200.0, 0.30),
    (0.20, 200.0, 0.30),
    (0.25, 200.0, 0.30),
    (0.17, 0.0, 0.30),
    (0.17, 500.0, 0.30),
    (0.17, 200.0, 0.15),
    (0.17, 200.0, 0.50),
)


@dataclass(frozen=True)
class MonteCarloJob:
    job_id: str
    topology: str
    dependence_model: str
    seed: int
    replications: int
    cfg: NetworkConfig


def _slug(value: str) -> str:
    return "_".join(
        part for part in "".join(character.lower() if character.isalnum() else " " for character in value).split()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_mc_job(job: MonteCarloJob) -> Tuple[MonteCarloJob, pd.DataFrame, float]:
    started = time.perf_counter()
    dataframe = simulate_monte_carlo(
        replications=job.replications,
        cfg=job.cfg,
        seed=job.seed,
        dependence_model=job.dependence_model,
    )
    dataframe.insert(0, "topology", job.topology)
    elapsed = time.perf_counter() - started
    return job, dataframe, elapsed


def _load_or_run_mc_jobs(
    jobs: Sequence[MonteCarloJob],
    raw_dir: Path,
    workers: int,
    resume: bool,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, object]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}
    runtime_rows: List[Dict[str, object]] = []
    pending: List[MonteCarloJob] = []
    for job in jobs:
        csv_path = raw_dir / f"{job.job_id}.csv"
        if resume and csv_path.exists():
            dataframe = pd.read_csv(csv_path)
            if len(dataframe) == job.replications:
                results[job.job_id] = dataframe
                runtime_rows.append(
                    {
                        "workload": "Monte Carlo",
                        "job_id": job.job_id,
                        "topology": job.topology,
                        "dependence_model": job.dependence_model,
                        "seed": job.seed,
                        "size": job.replications,
                        "seconds": np.nan,
                        "units_per_second": np.nan,
                        "source": "resumed",
                    }
                )
                continue
        pending.append(job)

    if not pending:
        return results, runtime_rows

    if workers <= 1:
        completed = (_run_mc_job(job) for job in pending)
        for job, dataframe, elapsed in completed:
            dataframe.to_csv(raw_dir / f"{job.job_id}.csv", index=False)
            results[job.job_id] = dataframe
            runtime_rows.append(
                {
                    "workload": "Monte Carlo",
                    "job_id": job.job_id,
                    "topology": job.topology,
                    "dependence_model": job.dependence_model,
                    "seed": job.seed,
                    "size": job.replications,
                    "seconds": elapsed,
                    "units_per_second": job.replications / elapsed,
                    "source": "executed",
                }
            )
            print(f"Completed {job.job_id} in {elapsed:.2f}s", flush=True)
        return results, runtime_rows

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_mc_job, job): job for job in pending}
            for future in as_completed(future_map):
                job, dataframe, elapsed = future.result()
                dataframe.to_csv(raw_dir / f"{job.job_id}.csv", index=False)
                results[job.job_id] = dataframe
                runtime_rows.append(
                    {
                        "workload": "Monte Carlo",
                        "job_id": job.job_id,
                        "topology": job.topology,
                        "dependence_model": job.dependence_model,
                        "seed": job.seed,
                        "size": job.replications,
                        "seconds": elapsed,
                        "units_per_second": job.replications / elapsed,
                        "source": "executed",
                    }
                )
                print(f"Completed {job.job_id} in {elapsed:.2f}s", flush=True)
    except (OSError, PermissionError) as error:
        print(
            f"Parallel execution unavailable ({error}); falling back to one worker.",
            flush=True,
        )
        return _load_or_run_mc_jobs(
            jobs,
            raw_dir=raw_dir,
            workers=1,
            resume=True,
        )
    return results, runtime_rows


def _parse_labels(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text == UNCLASSIFIED:
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def _method_labels(dataframe: pd.DataFrame, kind: str) -> set[str]:
    if kind == "fst":
        rows = dataframe[dataframe["moderate_crossing"]]
        column = "mechanism_labels_moderate"
    else:
        rows = dataframe[dataframe["passive_failure"]]
        column = "mechanism_labels"
    labels: set[str] = set()
    for value in rows[column]:
        labels.update(_parse_labels(value))
    return labels


def _signature_text(signature: frozenset[str]) -> str:
    if not signature:
        return UNCLASSIFIED
    return "; ".join(family for family in MECHANISM_FAMILIES if family in signature)


def _method_signature_counts(
    dataframe: pd.DataFrame, kind: str
) -> Counter[frozenset[str]]:
    if kind == "fst":
        rows = dataframe[dataframe["moderate_crossing"]]
        column = "mechanism_labels_moderate"
    else:
        rows = dataframe[dataframe["passive_failure"]]
        column = "mechanism_labels"
    return Counter(frozenset(_parse_labels(value)) for value in rows[column])


def _exact_signature_catalog(
    mc: pd.DataFrame,
    methods: Mapping[str, Tuple[pd.DataFrame, str]],
) -> pd.DataFrame:
    failures = mc[mc["moderate_passive_failure"]].copy()
    failures["_exact_signature"] = failures["mechanism_labels"].map(
        lambda value: frozenset(_parse_labels(value))
    )
    method_counts = {
        method: _method_signature_counts(dataframe, kind)
        for method, (dataframe, kind) in methods.items()
    }
    signature_counts = failures["_exact_signature"].value_counts()
    ordered_signatures = sorted(
        signature_counts.index,
        key=lambda signature: (-int(signature_counts[signature]), _signature_text(signature)),
    )
    rows: List[Dict[str, object]] = []
    for index, signature in enumerate(ordered_signatures, start=1):
        matching = failures[failures["_exact_signature"] == signature]
        row: Dict[str, object] = {
            "signature_id": f"MS-{index:02d}",
            "exact_multilabel_signature": _signature_text(signature),
            "label_count": len(signature),
            "MC_failure_count": len(matching),
            "MC_failure_share": len(matching) / len(failures) if len(failures) else 0.0,
            "MC_structural_count": int(
                (matching["passive_active_classification"] == STRUCTURAL).sum()
            ),
            "MC_adaptation_limited_count": int(
                (matching["passive_active_classification"] == ADAPTATION_LIMITED).sum()
            ),
        }
        for method, counts in method_counts.items():
            prefix = _slug(method)
            row[f"{prefix}_matching_crossings"] = int(counts.get(signature, 0))
            row[f"{prefix}_signature_discovered"] = signature in counts
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_rows(
    mc: pd.DataFrame,
    methods: Mapping[str, Tuple[pd.DataFrame, str]],
) -> pd.DataFrame:
    failures = mc[mc["moderate_passive_failure"]].copy()
    focal = failures[failures["primary_mechanism"].isin(CMD_PAC_RLTO)]
    failures["_exact_signature"] = failures["mechanism_labels"].map(
        lambda value: frozenset(_parse_labels(value))
    )
    non_adaptation = failures[
        failures["passive_active_classification"] != ADAPTATION_LIMITED
    ]
    focal_multilabel = failures[
        failures["_exact_signature"].map(lambda signature: bool(signature & CMD_PAC_RLTO))
    ]
    mc_signatures = set(failures["_exact_signature"])
    rows: List[Dict[str, object]] = []
    for method, (dataframe, kind) in methods.items():
        discovered = _method_labels(dataframe, kind)
        discovered_signature_counts = _method_signature_counts(dataframe, kind)
        discovered_signatures = set(discovered_signature_counts)
        full_primary = (
            float(failures["primary_mechanism"].isin(discovered).mean())
            if len(failures)
            else 0.0
        )
        non_adaptation_primary = (
            float(non_adaptation["primary_mechanism"].isin(discovered).mean())
            if len(non_adaptation)
            else 0.0
        )
        focal_primary = (
            float(focal["primary_mechanism"].isin(discovered).mean())
            if len(focal)
            else 0.0
        )
        any_label = (
            float(
                np.mean(
                    [bool(_parse_labels(value) & discovered) for value in failures["mechanism_labels"]]
                )
            )
            if len(failures)
            else 0.0
        )
        exact_full = (
            float(failures["_exact_signature"].isin(discovered_signatures).mean())
            if len(failures)
            else 0.0
        )
        exact_non_adaptation = (
            float(
                non_adaptation["_exact_signature"].isin(discovered_signatures).mean()
            )
            if len(non_adaptation)
            else 0.0
        )
        exact_focal = (
            float(
                focal_multilabel["_exact_signature"].isin(discovered_signatures).mean()
            )
            if len(focal_multilabel)
            else 0.0
        )
        exact_unique = (
            len(mc_signatures & discovered_signatures) / len(mc_signatures)
            if mc_signatures
            else 0.0
        )
        rows.append(
            {
                "method": method,
                "discovered_mechanism_families": "; ".join(
                    family for family in MECHANISM_FAMILIES if family in discovered
                ),
                "primary_family_coverage_full_denominator": full_primary,
                "primary_family_coverage_non_adaptation_limited": non_adaptation_primary,
                "primary_family_coverage_CMD_PAC_RL_TO": focal_primary,
                "multilabel_any_coverage_full_denominator": any_label,
                "discovered_exact_multilabel_signatures": " | ".join(
                    f"[{_signature_text(signature)}]"
                    for signature in sorted(discovered_signatures, key=_signature_text)
                ),
                "discovered_exact_multilabel_signature_count": len(
                    discovered_signatures
                ),
                "exact_multilabel_signature_coverage_full_denominator": exact_full,
                "exact_multilabel_signature_coverage_non_adaptation_limited": exact_non_adaptation,
                "exact_multilabel_signature_coverage_CMD_PAC_RL_TO": exact_focal,
                "exact_multilabel_signature_coverage_unique_signatures": exact_unique,
                "moderate_failure_denominator": len(failures),
                "non_adaptation_limited_denominator": len(non_adaptation),
                "CMD_PAC_RL_TO_denominator": len(focal),
                "CMD_PAC_RL_TO_multilabel_denominator": len(focal_multilabel),
                "MC_unique_exact_signature_denominator": len(mc_signatures),
            }
        )
    return pd.DataFrame(rows)


def _dense_ranks(scores: Mapping[str, float]) -> Dict[str, int]:
    unique = sorted({round(float(value), 12) for value in scores.values()}, reverse=True)
    rank_for_value = {value: index + 1 for index, value in enumerate(unique)}
    return {
        element: rank_for_value[round(float(score), 12)]
        for element, score in scores.items()
    }


def _fst_element_scores(
    dataframe: pd.DataFrame, pathways: Sequence[Pathway]
) -> Dict[str, float]:
    path_map = {pathway.pathway_id: pathway for pathway in pathways}
    crossing = dataframe[dataframe["moderate_crossing"]]
    scores = {element: 0.0 for element in ELEMENTS}
    if crossing.empty:
        return scores
    for scenario_id in crossing["scenario_id"]:
        elements = pathway_elements(path_map[str(scenario_id)])
        if not elements:
            continue
        contribution = 1.0 / len(elements)
        for element in elements:
            scores[element] += contribution
    return {element: score / len(crossing) for element, score in scores.items()}


def _mc_element_scores(mc: pd.DataFrame) -> Dict[str, float]:
    failures = mc[mc["moderate_passive_failure"]]
    if failures.empty:
        return {element: 0.0 for element in ELEMENTS}
    return {
        element: float(failures[f"has_{element}"].mean()) for element in ELEMENTS
    }


def _criticality_table(
    mc: pd.DataFrame,
    fst_baseline: pd.DataFrame,
    fst_balanced: pd.DataFrame,
    rst_scores: Mapping[str, float],
) -> pd.DataFrame:
    score_sets = {
        "Monte Carlo frequency": _mc_element_scores(mc),
        "Baseline FST": _fst_element_scores(fst_baseline, fst_library()),
        "Mechanism-balanced FST": _fst_element_scores(fst_balanced, fst_enhanced_library()),
        "RST": dict(rst_scores),
    }
    rows: List[Dict[str, object]] = []
    for method, scores in score_sets.items():
        ranks = _dense_ranks(scores)
        for element in ELEMENTS:
            rows.append(
                {
                    "method": method,
                    "element": element,
                    "score": scores.get(element, 0.0),
                    "rank": ranks.get(element, len(ELEMENTS)),
                }
            )
    return pd.DataFrame(rows)


def _weighted_family_counts(evaluations: Sequence[PathwayEval]) -> Dict[str, float]:
    totals = {family: 0.0 for family in MECHANISM_FAMILIES}
    for evaluation in evaluations:
        if not evaluation.passive_failure:
            continue
        for family in evaluation.mechanism_labels:
            totals[family] += evaluation.weight
    return totals


def _reweight_evaluations(
    evaluations: Sequence[PathwayEval], severity_weight: float, plausibility_weight: float
) -> List[PathwayEval]:
    copied = [replace(evaluation) for evaluation in evaluations]
    failure_indices = [index for index, evaluation in enumerate(copied) if evaluation.passive_failure]

    def normalize(values: Sequence[float]) -> List[float]:
        if not values:
            return []
        minimum, maximum = min(values), max(values)
        if math.isclose(minimum, maximum, abs_tol=1e-15):
            return [0.0] * len(values)
        return [(value - minimum) / (maximum - minimum) for value in values]

    severity = normalize([copied[index].severity for index in failure_indices])
    plausibility = normalize([copied[index].plausibility for index in failure_indices])
    raw = [
        severity_weight * severity[position] + plausibility_weight * plausibility[position]
        for position in range(len(failure_indices))
    ]
    if raw and math.isclose(sum(raw), 0.0, abs_tol=1e-15):
        raw = [1.0] * len(raw)
    for position, index in enumerate(failure_indices):
        copied[index].weight = raw[position]
    return copied


def _system_validation(cfg: NetworkConfig) -> pd.DataFrame:
    nominal = run_simulation([], cfg)
    stress_pathways = [*fst_library(), *fst_enhanced_library(), *rst_candidate_library()]
    simulations = [
        run_simulation(pathway.events, cfg, active_mode=active)
        for pathway in stress_pathways
        for active in (False, True)
    ]
    minimum_state = min(
        [
            *(min(result.backlog_series) for result in simulations),
            *(min(result.fp_inventory_series) for result in simulations),
            *(min(result.reserve_series) for result in simulations),
            *(
                min(series)
                for result in simulations
                for series in result.tier1_inventory_series.values()
            ),
        ]
    )
    maximum_pipeline_error = max(
        row["max_pipeline_conservation_error"]
        for result in simulations
        for row in result.trace
    )
    maximum_corridor_pipeline_error = max(
        row["corridor_pipeline_conservation_error"]
        for result in simulations
        for row in result.trace
    )
    maximum_inventory_error = max(
        row["max_tier1_inventory_balance_error"]
        for result in simulations
        for row in result.trace
    )
    maximum_corridor_excess = max(
        row["corridor_throughput"] - row["corridor_capacity"]
        for result in simulations
        for row in result.trace
    )
    maximum_reroute_excess = max(
        max(
            row["rerouted_flow"] - row["blocked_flow"],
            row["rerouted_flow"] - cfg.active_alt_corridor_units,
            row["rerouted_flow"] - row["pre_response_shortfall"],
        )
        for result in simulations
        for row in result.trace
    )
    maximum_reserve_increase = max(
        current - previous
        for result in simulations
        for previous, current in zip(result.reserve_series, result.reserve_series[1:])
    )
    maximum_release_excess = max(
        sum(result.reserve_release_series)
        - cfg.active_reserve_release_frac * cfg.protected_reserve_capacity
        for result in simulations
    )
    maximum_reroute_used = max(
        max(result.rerouted_series) for result in simulations
    )
    maximum_cumulative_release = max(
        sum(result.reserve_release_series) for result in simulations
    )
    nominal_max_deviation = max(
        [
            *(abs(value - 1.0) for value in nominal.service_levels),
            *(abs(value) for value in nominal.backlog_series),
            *(abs(value) for value in nominal.fp_inventory_series),
            *(
                abs(value - cfg.tier1_initial_inventory)
                for series in nominal.tier1_inventory_series.values()
                for value in series
            ),
        ]
    )
    return pd.DataFrame(
        [
            {
                "check": "52-period nominal steady state",
                "requirement": "All nominal state equations hold for 52 periods",
                "observed": nominal_max_deviation,
                "tolerance": 1e-9,
                "passed": nominal_max_deviation <= 1e-9,
            },
            {
                "check": "Nonnegative inventories and backlog",
                "requirement": "Minimum state >= 0",
                "observed": minimum_state,
                "tolerance": -1e-9,
                "passed": minimum_state >= -1e-9,
            },
            {
                "check": "Tier-1 pipeline conservation",
                "requirement": "Maximum absolute residual <= 1e-9",
                "observed": maximum_pipeline_error,
                "tolerance": 1e-9,
                "passed": maximum_pipeline_error <= 1e-9,
            },
            {
                "check": "Corridor pipeline conservation",
                "requirement": "Maximum absolute residual <= 1e-9",
                "observed": maximum_corridor_pipeline_error,
                "tolerance": 1e-9,
                "passed": maximum_corridor_pipeline_error <= 1e-9,
            },
            {
                "check": "Tier-1 inventory conservation",
                "requirement": "Maximum absolute residual <= 1e-9",
                "observed": maximum_inventory_error,
                "tolerance": 1e-9,
                "passed": maximum_inventory_error <= 1e-9,
            },
            {
                "check": "Corridor throughput bound",
                "requirement": "Throughput <= effective corridor capacity",
                "observed": maximum_corridor_excess,
                "tolerance": 1e-9,
                "passed": maximum_corridor_excess <= 1e-9,
            },
            {
                "check": "Rerouting bounds",
                "requirement": "Reroute <= blocked flow, cap, and shortfall",
                "observed": maximum_reroute_excess,
                "tolerance": 1e-9,
                "passed": maximum_reroute_excess <= 1e-9,
            },
            {
                "check": "Reserve monotonicity",
                "requirement": "Reserve never increases",
                "observed": maximum_reserve_increase,
                "tolerance": 1e-9,
                "passed": maximum_reserve_increase <= 1e-9,
            },
            {
                "check": "Reserve release cap",
                "requirement": "Cumulative release <= rho_rel * reserve",
                "observed": maximum_release_excess,
                "tolerance": 1e-9,
                "passed": maximum_release_excess <= 1e-9,
            },
            {
                "check": "Rerouting response exercised",
                "requirement": "At least one validation pathway uses rerouting",
                "observed": maximum_reroute_used,
                "tolerance": 0.0,
                "passed": maximum_reroute_used > 0.0,
            },
            {
                "check": "Reserve response exercised",
                "requirement": "At least one validation pathway releases reserve",
                "observed": maximum_cumulative_release,
                "tolerance": 0.0,
                "passed": maximum_cumulative_release > 0.0,
            },
        ]
    )


def _mechanism_validation(cfg: NetworkConfig) -> pd.DataFrame:
    pathways = {pathway.pathway_id: pathway for pathway in rst_candidate_library()}

    def attribution(pathway_id: str):
        pathway = pathways[pathway_id]
        passive = run_simulation(pathway.events, cfg)
        return attribute_mechanisms(pathway.events, cfg, passive_result=passive)

    base = pathways["RST-01"]
    base_result = run_simulation(base.events, cfg)
    _, first_qualified = threshold_cross_info(base_result.service_levels, MODERATE)
    late_event = Event("demand", 0.40, int(first_qualified or 0) + 2, 4, 2)
    extended_events = (*base.events, late_event)
    extended_result = run_simulation(extended_events, cfg)
    base_attribution = attribute_mechanisms(base.events, cfg, passive_result=base_result)
    extended_attribution = attribute_mechanisms(
        extended_events, cfg, passive_result=extended_result
    )
    causal_pass = (
        base_attribution.labels == extended_attribution.labels
        and 1 not in extended_attribution.causal_prefix_indices
    )

    order_pathway = pathways["RST-46"]
    order_result = run_simulation(order_pathway.events, cfg)
    expected_labels = attribute_mechanisms(
        order_pathway.events, cfg, passive_result=order_result
    ).labels
    order_pass = all(
        attribute_mechanisms(
            order_pathway.events,
            cfg,
            passive_result=order_result,
            label_order=order,
        ).labels
        == expected_labels
        for order in itertools.permutations(MECHANISM_FAMILIES)
    )

    cd_attribution = attribution("RST-05")
    hcd_attribution = attribution("RST-01")
    cmd_attribution = attribution("RST-23")
    pac_attribution = attribution("RST-01")
    rlto_attribution = attribution("RST-46")
    first, second = order_pathway.events
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
    end_boundary_result = run_simulation(end_boundary_events, cfg)
    end_boundary_attribution = attribute_mechanisms(
        end_boundary_events, cfg, passive_result=end_boundary_result
    )
    rows = [
        {
            "check": "Causal-prefix restriction",
            "probe": "RST-01 plus a post-crossing demand event",
            "observed": f"base={base_attribution.labels}; extended={extended_attribution.labels}; prefix={extended_attribution.causal_prefix_indices}",
            "passed": causal_pass,
        },
        {
            "check": "Label-order invariance",
            "probe": "RST-46 across all 120 signature orders",
            "observed": "; ".join(expected_labels),
            "passed": order_pass,
        },
        {
            "check": "Single-event CD",
            "probe": "RST-05",
            "observed": f"labels={cd_attribution.labels}; singles={cd_attribution.single_event_crossings}",
            "passed": CD in cd_attribution.labels and cd_attribution.single_event_crossings == (0,),
        },
        {
            "check": "HCD removal counterfactual",
            "probe": "RST-01",
            "observed": f"labels={hcd_attribution.labels}; removals={hcd_attribution.hcd_removal_elements}",
            "passed": HCD in hcd_attribution.labels and hcd_attribution.hcd_removal_elements == ("T2P",),
        },
        {
            "check": "CMD constituent-event tests",
            "probe": "RST-23",
            "observed": f"labels={cmd_attribution.labels}; singles={cmd_attribution.single_event_crossings}",
            "passed": CMD in cmd_attribution.labels and not cmd_attribution.single_event_crossings,
        },
        {
            "check": "PAC direct-impact counterfactual",
            "probe": "RST-01",
            "observed": f"labels={pac_attribution.labels}; direct_failure={pac_attribution.direct_impact_crossing}",
            "passed": PAC in pac_attribution.labels and pac_attribution.direct_impact_crossing is False,
        },
        {
            "check": "RL-TO recovery-window test",
            "probe": "RST-46 and recovery-end boundary",
            "observed": f"support={rlto_attribution.rlto_support_pairs}; end_labels={end_boundary_attribution.labels}",
            "passed": (
                RL_TO in rlto_attribution.labels
                and (0, 1) in rlto_attribution.rlto_support_pairs
                and RL_TO not in end_boundary_attribution.labels
            ),
        },
    ]
    return pd.DataFrame(rows)


def _method_comparison(
    fst_baseline: pd.DataFrame,
    fst_balanced: pd.DataFrame,
    rst: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    coverage_map = coverage.set_index("method").to_dict("index")
    rows = []
    for method, dataframe, kind in (
        ("Baseline FST", fst_baseline, "fst"),
        ("Mechanism-balanced FST", fst_balanced, "fst"),
        ("RST", rst, "rst"),
    ):
        if kind == "fst":
            moderate = int(dataframe["moderate_crossing"].sum())
            severe = int(dataframe["severe_crossing"].sum())
            structural = np.nan
            adaptation = np.nan
        else:
            moderate = int(dataframe["passive_failure"].sum())
            severe = int(dataframe["passive_severe_failure"].sum())
            structural = int((dataframe["classification"] == STRUCTURAL).sum())
            adaptation = int((dataframe["classification"] == ADAPTATION_LIMITED).sum())
        cov = coverage_map[method]
        rows.append(
            {
                "method": method,
                "budget": len(dataframe),
                "moderate_threshold_crossings": moderate,
                "severe_threshold_crossings": severe,
                "structural_pathways": structural,
                "adaptation_limited_pathways": adaptation,
                "primary_family_coverage_full_denominator": cov[
                    "primary_family_coverage_full_denominator"
                ],
                "primary_family_coverage_non_adaptation_limited": cov[
                    "primary_family_coverage_non_adaptation_limited"
                ],
                "primary_family_coverage_CMD_PAC_RL_TO": cov[
                    "primary_family_coverage_CMD_PAC_RL_TO"
                ],
                "multilabel_any_coverage_full_denominator": cov[
                    "multilabel_any_coverage_full_denominator"
                ],
                "exact_multilabel_signature_coverage_full_denominator": cov[
                    "exact_multilabel_signature_coverage_full_denominator"
                ],
                "exact_multilabel_signature_coverage_non_adaptation_limited": cov[
                    "exact_multilabel_signature_coverage_non_adaptation_limited"
                ],
                "exact_multilabel_signature_coverage_CMD_PAC_RL_TO": cov[
                    "exact_multilabel_signature_coverage_CMD_PAC_RL_TO"
                ],
                "exact_multilabel_signature_coverage_unique_signatures": cov[
                    "exact_multilabel_signature_coverage_unique_signatures"
                ],
            }
        )
    return pd.DataFrame(rows)


def _threshold_sensitivity(cfg: NetworkConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    candidates = rst_candidate_library()
    for alpha in (0.90, 0.85, 0.80):
        for duration in (1, 2, 3, 4):
            threshold = FailureThreshold(f"SL<{alpha:.2f} for >={duration}", alpha, duration)
            started = time.perf_counter()
            evaluations, scores = mechanism_guided_rst_search(
                candidates, cfg, threshold=threshold
            )
            elapsed = time.perf_counter() - started
            family_weights = _weighted_family_counts(evaluations)
            dominant = max(family_weights, key=family_weights.get) if any(family_weights.values()) else UNCLASSIFIED
            rows.append(
                {
                    "service_level_threshold": alpha,
                    "duration_periods": duration,
                    "passive_pathways": sum(evaluation.passive_failure for evaluation in evaluations),
                    "structural_pathways": sum(evaluation.classification == STRUCTURAL for evaluation in evaluations),
                    "adaptation_limited_pathways": sum(
                        evaluation.classification == ADAPTATION_LIMITED for evaluation in evaluations
                    ),
                    "unclassified_passive_pathways": sum(
                        evaluation.passive_failure and not evaluation.mechanism_labels
                        for evaluation in evaluations
                    ),
                    "dominant_mechanism_family": dominant,
                    "T2P_IF": scores.get("T2P", 0.0),
                    "corridor_IF": scores.get("corridor", 0.0),
                    "runtime_seconds": elapsed,
                }
            )
            print(f"Threshold alpha={alpha:.2f}, duration={duration} completed", flush=True)
    return pd.DataFrame(rows)


def _response_sensitivity(cfg: NetworkConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    candidates = rst_candidate_library()
    for tau_ratio, alt_capacity, release_fraction in RESPONSE_SENSITIVITY_SETTINGS:
        setting_cfg = replace(
            cfg,
            active_backlog_trigger_ratio=tau_ratio,
            active_alt_corridor_units=alt_capacity,
            active_reserve_release_frac=release_fraction,
        )
        started = time.perf_counter()
        evaluations, _ = mechanism_guided_rst_search(candidates, setting_cfg)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "tau_B_over_D": tau_ratio,
                "tau_B_units": tau_ratio * cfg.demand_per_week,
                "kappa_alt": alt_capacity,
                "rho_rel": release_fraction,
                "passive_pathways": sum(evaluation.passive_failure for evaluation in evaluations),
                "structural_pathways": sum(evaluation.classification == STRUCTURAL for evaluation in evaluations),
                "adaptation_limited_pathways": sum(
                    evaluation.classification == ADAPTATION_LIMITED for evaluation in evaluations
                ),
                "runtime_seconds": elapsed,
            }
        )
        print(
            f"Response tau={tau_ratio:.2f}, kappa={alt_capacity:.0f}, rho={release_fraction:.2f} completed",
            flush=True,
        )
    return pd.DataFrame(rows)


def _weight_sensitivity(
    evaluations: Sequence[PathwayEval], baseline_scores: Mapping[str, float]
) -> pd.DataFrame:
    baseline_ranks = _dense_ranks(baseline_scores)
    rows: List[Dict[str, object]] = []
    for severity_weight, plausibility_weight in (
        (1.00, 0.00),
        (0.75, 0.25),
        (0.625, 0.375),
        (0.50, 0.50),
        (0.25, 0.75),
        (0.00, 1.00),
    ):
        reweighted = _reweight_evaluations(
            evaluations, severity_weight, plausibility_weight
        )
        scores = compute_criticality(reweighted)
        ranks = _dense_ranks(scores)
        failures = [evaluation for evaluation in reweighted if evaluation.passive_failure]
        top_pathway = max(failures, key=lambda evaluation: evaluation.weight)
        family_weights = _weighted_family_counts(reweighted)
        top_family = max(family_weights, key=family_weights.get)
        rows.append(
            {
                "omega_C": severity_weight,
                "omega_P": plausibility_weight,
                "top_pathway": top_pathway.pathway_id,
                "top_generation_family": top_pathway.family,
                "top_mechanism_family": top_family,
                "top_classification": top_pathway.classification,
                "T2P_IF": scores.get("T2P", 0.0),
                "corridor_IF": scores.get("corridor", 0.0),
                "T2P_rank": ranks.get("T2P"),
                "corridor_rank": ranks.get("corridor"),
                "rank_stable": ranks == baseline_ranks,
            }
        )
    return pd.DataFrame(rows)


def _seed_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_columns = ["topology", "dependence_model", "seed"]
    for keys, group in dataframe.groupby(group_columns, dropna=False):
        topology, dependence_model, seed = keys
        failures = group[group["moderate_passive_failure"]]
        primary_counts = failures["primary_mechanism"].value_counts()
        classification_counts = failures["passive_active_classification"].value_counts()
        rows.append(
            {
                "topology": topology,
                "dependence_model": dependence_model,
                "seed": int(seed),
                "replications": len(group),
                "moderate_failures": int(group["moderate_passive_failure"].sum()),
                "severe_failures": int(group["severe_passive_failure"].sum()),
                "structural_failures": int(classification_counts.get(STRUCTURAL, 0)),
                "adaptation_limited_failures": int(
                    classification_counts.get(ADAPTATION_LIMITED, 0)
                ),
                "adaptation_limited_share": (
                    float((failures["passive_active_classification"] == ADAPTATION_LIMITED).mean())
                    if len(failures)
                    else 0.0
                ),
                "dominant_mechanism_family": (
                    str(primary_counts.index[0]) if len(primary_counts) else UNCLASSIFIED
                ),
            }
        )
    return pd.DataFrame(rows)


def _spread_summary(
    dataframe: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    metric_columns = [
        "moderate_failures",
        "severe_failures",
        "structural_failures",
        "adaptation_limited_failures",
        "adaptation_limited_share",
    ]
    rows: List[Dict[str, object]] = []
    for keys, group in dataframe.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = group[metric].astype(float)
            rows.append(
                {
                    **key_values,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "runs": len(values),
                }
            )
    return pd.DataFrame(rows)


def _topology_robustness_rows(
    mc_by_job: Mapping[str, pd.DataFrame],
    seeds: Sequence[int],
    replications: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    method_cache: Dict[str, Dict[str, object]] = {}
    for topology, cfg in (("A", topology_a_config()), ("B", topology_b_config())):
        fst_base = run_fst(fst_library(), cfg)
        fst_balanced = run_fst(fst_enhanced_library(), cfg)
        rst_evaluations, rst_scores = mechanism_guided_rst_search(
            rst_candidate_library(), cfg
        )
        rst_frame = pathway_evaluations_dataframe(
            rst_evaluations, rst_candidate_library()
        )
        baseline_fst_ranks = _dense_ranks(
            _fst_element_scores(fst_base, fst_library())
        )
        balanced_fst_ranks = _dense_ranks(
            _fst_element_scores(fst_balanced, fst_enhanced_library())
        )
        rst_ranks = _dense_ranks(rst_scores)
        method_cache[topology] = {
            "cfg": cfg,
            "fst_baseline": fst_base,
            "fst_balanced": fst_balanced,
            "rst_evaluations": rst_evaluations,
            "rst_frame": rst_frame,
            "rst_scores": rst_scores,
        }
        for seed in seeds:
            job_id = f"topology_{topology}_independent_seed_{seed}_n_{replications}"
            mc = mc_by_job[job_id]
            coverage = _coverage_rows(
                mc,
                {
                    "Baseline FST": (fst_base, "fst"),
                    "Mechanism-balanced FST": (fst_balanced, "fst"),
                    "RST": (rst_frame, "rst"),
                },
            ).set_index("method")
            mc_ranks = _dense_ranks(_mc_element_scores(mc))
            rows.append(
                {
                    "topology": topology,
                    "topology_description": (
                        "Shared T2P, single corridor"
                        if topology == "A"
                        else "S1A/S1B via T2P-A; S1C via T2P-B; single corridor"
                    ),
                    "seed": seed,
                    "replications": len(mc),
                    "MC_moderate_failures": int(mc["moderate_passive_failure"].sum()),
                    "MC_severe_failures": int(mc["severe_passive_failure"].sum()),
                    "FST_baseline_crossings": int(fst_base["moderate_crossing"].sum()),
                    "FST_mechanism_balanced_crossings": int(
                        fst_balanced["moderate_crossing"].sum()
                    ),
                    "RST_passive_pathways": int(rst_frame["passive_failure"].sum()),
                    "RST_severe_pathways": int(
                        rst_frame["passive_severe_failure"].sum()
                    ),
                    "RST_structural_pathways": int(
                        (rst_frame["classification"] == STRUCTURAL).sum()
                    ),
                    "RST_adaptation_limited_pathways": int(
                        (rst_frame["classification"] == ADAPTATION_LIMITED).sum()
                    ),
                    "MC_upstream_processor_rank": mc_ranks.get("T2P"),
                    "MC_corridor_rank": mc_ranks.get("corridor"),
                    "baseline_FST_upstream_processor_rank": baseline_fst_ranks.get("T2P"),
                    "baseline_FST_corridor_rank": baseline_fst_ranks.get("corridor"),
                    "mechanism_balanced_FST_upstream_processor_rank": balanced_fst_ranks.get("T2P"),
                    "mechanism_balanced_FST_corridor_rank": balanced_fst_ranks.get("corridor"),
                    "RST_upstream_processor_rank": rst_ranks.get("T2P"),
                    "RST_corridor_rank": rst_ranks.get("corridor"),
                    "baseline_FST_non_adaptation_coverage": coverage.loc[
                        "Baseline FST", "primary_family_coverage_non_adaptation_limited"
                    ],
                    "mechanism_balanced_FST_non_adaptation_coverage": coverage.loc[
                        "Mechanism-balanced FST",
                        "primary_family_coverage_non_adaptation_limited",
                    ],
                    "RST_non_adaptation_coverage": coverage.loc[
                        "RST", "primary_family_coverage_non_adaptation_limited"
                    ],
                    "baseline_FST_CMD_PAC_RL_TO_coverage": coverage.loc[
                        "Baseline FST", "primary_family_coverage_CMD_PAC_RL_TO"
                    ],
                    "mechanism_balanced_FST_CMD_PAC_RL_TO_coverage": coverage.loc[
                        "Mechanism-balanced FST", "primary_family_coverage_CMD_PAC_RL_TO"
                    ],
                    "RST_CMD_PAC_RL_TO_coverage": coverage.loc[
                        "RST", "primary_family_coverage_CMD_PAC_RL_TO"
                    ],
                    "baseline_FST_exact_multilabel_signature_coverage": coverage.loc[
                        "Baseline FST",
                        "exact_multilabel_signature_coverage_full_denominator",
                    ],
                    "mechanism_balanced_FST_exact_multilabel_signature_coverage": coverage.loc[
                        "Mechanism-balanced FST",
                        "exact_multilabel_signature_coverage_full_denominator",
                    ],
                    "RST_exact_multilabel_signature_coverage": coverage.loc[
                        "RST",
                        "exact_multilabel_signature_coverage_full_denominator",
                    ],
                }
            )
    return pd.DataFrame(rows), method_cache


def _topology_summary(runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in runs.columns
        if column
        not in {"topology", "topology_description", "seed", "replications"}
    ]
    rows: List[Dict[str, object]] = []
    for topology, group in runs.groupby("topology"):
        description = str(group["topology_description"].iloc[0])
        for metric in metric_columns:
            values = group[metric].astype(float)
            rows.append(
                {
                    "topology": topology,
                    "topology_description": description,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "runs": len(values),
                }
            )
    return pd.DataFrame(rows)


def _runtime_scaling(cfg: NetworkConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for size in (100, 250, 500, 1000):
        started = time.perf_counter()
        simulate_monte_carlo(
            size,
            cfg,
            seed=9000 + size,
            dependence_model="Independent baseline",
            attribute_failures=False,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "workload": "Monte Carlo without mechanism counterfactuals",
                "size": size,
                "seconds": elapsed,
                "units_per_second": size / elapsed,
            }
        )
        print(f"Runtime scaling MC n={size} completed", flush=True)
    candidates = rst_candidate_library()
    for budget in (10, 25, 50):
        started = time.perf_counter()
        mechanism_guided_rst_search(candidates, cfg, budget=budget)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "workload": "RST with multilabel counterfactual attribution",
                "size": budget,
                "seconds": elapsed,
                "units_per_second": budget / elapsed,
            }
        )
        print(f"Runtime scaling RST budget={budget} completed", flush=True)
    return pd.DataFrame(rows)


def generate_analysis(
    output_dir: Path,
    seeds: Sequence[int],
    replications_per_seed: int,
    workers: int,
    resume: bool,
    run_dependence: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_seed_runs"
    cfg = topology_a_config()
    overall_started = time.perf_counter()

    system_validation = _system_validation(cfg)
    mechanism_validation = _mechanism_validation(cfg)
    system_validation.to_csv(output_dir / "validation_system_operator.csv", index=False)
    mechanism_validation.to_csv(
        output_dir / "validation_mechanism_attribution.csv", index=False
    )
    if not bool(system_validation["passed"].all() and mechanism_validation["passed"].all()):
        raise RuntimeError("Validation gate failed; primary computation was not started")

    jobs: List[MonteCarloJob] = []
    for topology, topology_cfg in (("A", topology_a_config()), ("B", topology_b_config())):
        for seed in seeds:
            jobs.append(
                MonteCarloJob(
                    job_id=f"topology_{topology}_independent_seed_{seed}_n_{replications_per_seed}",
                    topology=topology,
                    dependence_model="Independent baseline",
                    seed=seed,
                    replications=replications_per_seed,
                    cfg=topology_cfg,
                )
            )
    if run_dependence:
        for dependence_model in DEPENDENCE_MODELS[1:]:
            for seed in seeds:
                jobs.append(
                    MonteCarloJob(
                        job_id=f"topology_A_{_slug(dependence_model)}_seed_{seed}_n_{replications_per_seed}",
                        topology="A",
                        dependence_model=dependence_model,
                        seed=seed,
                        replications=replications_per_seed,
                        cfg=cfg,
                    )
                )
    mc_by_job, job_runtime = _load_or_run_mc_jobs(
        jobs, raw_dir=raw_dir, workers=workers, resume=resume
    )

    primary_frames = [
        mc_by_job[f"topology_A_independent_seed_{seed}_n_{replications_per_seed}"]
        for seed in seeds
    ]
    primary_mc = pd.concat(primary_frames, ignore_index=True)
    primary_mc.to_csv(output_dir / "monte_carlo_primary_5000.csv", index=False)
    primary_seed_summary = _seed_summary(primary_mc)
    primary_seed_summary.to_csv(output_dir / "primary_seed_summary.csv", index=False)

    fst_baseline = run_fst(fst_library(), cfg)
    fst_balanced = run_fst(fst_enhanced_library(), cfg)
    rst_started = time.perf_counter()
    rst_evaluations, rst_scores = mechanism_guided_rst_search(
        rst_candidate_library(), cfg
    )
    rst_elapsed = time.perf_counter() - rst_started
    rst_frame = pathway_evaluations_dataframe(
        rst_evaluations, rst_candidate_library()
    )
    fst_baseline.to_csv(output_dir / "fst_baseline_results.csv", index=False)
    fst_balanced.to_csv(
        output_dir / "fst_mechanism_balanced_results.csv", index=False
    )
    rst_frame.to_csv(output_dir / "rst_results.csv", index=False)

    coverage = _coverage_rows(
        primary_mc,
        {
            "Baseline FST": (fst_baseline, "fst"),
            "Mechanism-balanced FST": (fst_balanced, "fst"),
            "RST": (rst_frame, "rst"),
        },
    )
    coverage.to_csv(output_dir / "primary_mechanism_coverage.csv", index=False)
    exact_coverage_columns = [
        "method",
        "discovered_exact_multilabel_signatures",
        "discovered_exact_multilabel_signature_count",
        "exact_multilabel_signature_coverage_full_denominator",
        "exact_multilabel_signature_coverage_non_adaptation_limited",
        "exact_multilabel_signature_coverage_CMD_PAC_RL_TO",
        "exact_multilabel_signature_coverage_unique_signatures",
        "moderate_failure_denominator",
        "non_adaptation_limited_denominator",
        "CMD_PAC_RL_TO_multilabel_denominator",
        "MC_unique_exact_signature_denominator",
    ]
    exact_coverage = coverage[exact_coverage_columns].copy()
    exact_coverage.to_csv(
        output_dir / "primary_exact_multilabel_signature_coverage.csv", index=False
    )
    exact_signature_catalog = _exact_signature_catalog(
        primary_mc,
        {
            "Baseline FST": (fst_baseline, "fst"),
            "Mechanism-balanced FST": (fst_balanced, "fst"),
            "RST": (rst_frame, "rst"),
        },
    )
    exact_signature_catalog.to_csv(
        output_dir / "primary_exact_multilabel_signature_catalog.csv", index=False
    )
    comparison = _method_comparison(
        fst_baseline, fst_balanced, rst_frame, coverage
    )
    comparison.to_csv(output_dir / "primary_method_comparison.csv", index=False)
    criticality = _criticality_table(
        primary_mc, fst_baseline, fst_balanced, rst_scores
    )
    criticality.to_csv(output_dir / "criticality_rankings.csv", index=False)

    threshold_sensitivity = _threshold_sensitivity(cfg)
    response_sensitivity = _response_sensitivity(cfg)
    weight_sensitivity = _weight_sensitivity(rst_evaluations, rst_scores)
    threshold_sensitivity.to_csv(
        output_dir / "sensitivity_threshold.csv", index=False
    )
    response_sensitivity.to_csv(
        output_dir / "sensitivity_response.csv", index=False
    )
    weight_sensitivity.to_csv(
        output_dir / "sensitivity_severity_plausibility_weights.csv", index=False
    )

    topology_runs, topology_method_cache = _topology_robustness_rows(
        mc_by_job, seeds, replications_per_seed
    )
    topology_summary = _topology_summary(topology_runs)
    topology_runs.to_csv(
        output_dir / "robustness_topology_seed_runs.csv", index=False
    )
    topology_summary.to_csv(
        output_dir / "robustness_topology_summary.csv", index=False
    )

    dependence_runs = pd.DataFrame()
    dependence_summary = pd.DataFrame()
    if run_dependence:
        dependence_frames = primary_frames.copy()
        for dependence_model in DEPENDENCE_MODELS[1:]:
            dependence_frames.extend(
                mc_by_job[
                    f"topology_A_{_slug(dependence_model)}_seed_{seed}_n_{replications_per_seed}"
                ]
                for seed in seeds
            )
        dependence_mc = pd.concat(dependence_frames, ignore_index=True)
        dependence_runs = _seed_summary(dependence_mc)
        dependence_summary = _spread_summary(
            dependence_runs, ["dependence_model"]
        )
        dependence_runs.to_csv(
            output_dir / "sensitivity_dependence_seed_runs.csv", index=False
        )
        dependence_summary.to_csv(
            output_dir / "sensitivity_dependence_summary.csv", index=False
        )

    runtime_scaling = _runtime_scaling(cfg)
    job_runtime_frame = pd.DataFrame(job_runtime)
    rst_runtime_row = pd.DataFrame(
        [
            {
                "workload": "Primary RST",
                "job_id": "topology_A_RST_budget_50",
                "topology": "A",
                "dependence_model": "N/A",
                "seed": np.nan,
                "size": 50,
                "seconds": rst_elapsed,
                "units_per_second": 50 / rst_elapsed,
                "source": "executed",
            }
        ]
    )
    job_runtime_frame = pd.concat(
        [job_runtime_frame, rst_runtime_row], ignore_index=True
    )
    job_runtime_frame.to_csv(output_dir / "runtime_primary_jobs.csv", index=False)
    runtime_scaling.to_csv(output_dir / "runtime_scaling.csv", index=False)

    pooled_moderate = int(primary_mc["moderate_passive_failure"].sum())
    pooled_severe = int(primary_mc["severe_passive_failure"].sum())
    primary_comparison = comparison.set_index("method")
    criticality_ranks = {
        method: {
            str(row["element"]): int(row["rank"])
            for _, row in group.iterrows()
        }
        for method, group in criticality.groupby("method")
    }
    manuscript_targets = {
        "baseline_FST_moderate": 7,
        "baseline_FST_severe": 3,
        "mechanism_balanced_FST_moderate": 22,
        "mechanism_balanced_FST_severe": 9,
        "RST_passive": 21,
        "RST_severe": 5,
        "RST_structural": 11,
        "RST_adaptation_limited": 10,
    }
    observed_targets = {
        "baseline_FST_moderate": int(fst_baseline["moderate_crossing"].sum()),
        "baseline_FST_severe": int(fst_baseline["severe_crossing"].sum()),
        "mechanism_balanced_FST_moderate": int(
            fst_balanced["moderate_crossing"].sum()
        ),
        "mechanism_balanced_FST_severe": int(
            fst_balanced["severe_crossing"].sum()
        ),
        "RST_passive": int(rst_frame["passive_failure"].sum()),
        "RST_severe": int(rst_frame["passive_severe_failure"].sum()),
        "RST_structural": int((rst_frame["classification"] == STRUCTURAL).sum()),
        "RST_adaptation_limited": int(
            (rst_frame["classification"] == ADAPTATION_LIMITED).sum()
        ),
    }
    manuscript_counts_reproduced = manuscript_targets == observed_targets
    summary = {
        "status": "completed",
        "validation_passed": True,
        "primary": {
            "seeds": list(seeds),
            "replications_per_seed": replications_per_seed,
            "pooled_replications": len(primary_mc),
            "pooled_moderate_failures": pooled_moderate,
            "pooled_severe_failures": pooled_severe,
            "baseline_FST_moderate_crossings": observed_targets[
                "baseline_FST_moderate"
            ],
            "baseline_FST_severe_crossings": observed_targets[
                "baseline_FST_severe"
            ],
            "mechanism_balanced_FST_moderate_crossings": observed_targets[
                "mechanism_balanced_FST_moderate"
            ],
            "mechanism_balanced_FST_severe_crossings": observed_targets[
                "mechanism_balanced_FST_severe"
            ],
            "RST_passive_pathways": observed_targets["RST_passive"],
            "RST_severe_pathways": observed_targets["RST_severe"],
            "RST_structural_pathways": observed_targets["RST_structural"],
            "RST_adaptation_limited_pathways": observed_targets[
                "RST_adaptation_limited"
            ],
            "coverage": primary_comparison[
                [
                    "primary_family_coverage_full_denominator",
                    "primary_family_coverage_non_adaptation_limited",
                    "primary_family_coverage_CMD_PAC_RL_TO",
                    "multilabel_any_coverage_full_denominator",
                    "exact_multilabel_signature_coverage_full_denominator",
                    "exact_multilabel_signature_coverage_non_adaptation_limited",
                    "exact_multilabel_signature_coverage_CMD_PAC_RL_TO",
                    "exact_multilabel_signature_coverage_unique_signatures",
                ]
            ].to_dict("index"),
            "criticality_ranks": criticality_ranks,
        },
        "manuscript_numeric_consistency": {
            "targets": manuscript_targets,
            "observed": observed_targets,
            "reproduced": manuscript_counts_reproduced,
        },
        "model_consistency_flags": [
            "Section 4.2 uses tau_B=0.20D as the primary baseline; the four kappa_alt and rho_rel robustness reruns use the Table 8 reference tau_B=0.17D.",
            "Section 3.7 excludes controllability from structural weights; sensitivity therefore varies only omega_C and omega_P.",
            "The manuscript invokes a primary mechanism family without defining a tie-break; the package uses the documented fixed priority HCD, CMD, PAC, RL-TO, CD only for scalar summaries.",
            "Topology B keeps total upstream capacity at 1,000 and allocates it 2:1 across T2P-A and T2P-B according to served tier-1 channels.",
            "Under the revised operator, pooled Monte Carlo frequency ranks the corridor first and T2P second, while RST ranks T2P first and the corridor second.",
        ],
        "runtime_seconds": time.perf_counter() - overall_started,
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as target:
        json.dump(summary, target, indent=2)

    files = sorted(
        path.name for path in output_dir.iterdir() if path.is_file()
    )
    manifest = {
        "generator": "generate_revised_analysis.py",
        "model": "rst_revised_model.py",
        "generator_sha256": _sha256(Path(__file__)),
        "model_sha256": _sha256(Path(__file__).with_name("rst_revised_model.py")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "seeds": list(seeds),
        "replications_per_seed": replications_per_seed,
        "workers": workers,
        "dependence_models": list(DEPENDENCE_MODELS) if run_dependence else ["Independent baseline"],
        "primary_mechanism_priority": list(PRIMARY_MECHANISM_PRIORITY),
        "topology_A": asdict(topology_a_config()),
        "topology_B": asdict(topology_b_config()),
        "outputs": files,
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the revised Sections 3 and 4 computational package."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("02_revised_analysis_outputs"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--replications-per-seed", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-dependence", action="store_true")
    args = parser.parse_args()
    summary = generate_analysis(
        output_dir=args.output_dir,
        seeds=args.seeds,
        replications_per_seed=args.replications_per_seed,
        workers=args.workers,
        resume=not args.no_resume,
        run_dependence=not args.skip_dependence,
    )
    print(json.dumps(summary["primary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
