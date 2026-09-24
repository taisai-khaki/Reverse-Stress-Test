#!/usr/bin/env python3
"""Second-round IJPR computational extension.

The audited model is imported but not modified.  All new protocol variants and
outputs are isolated under 03_round2_revision_outputs/<run_id>/.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import rst_revised_model as model


PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVED_SEEDS = (42, 314, 1618, 2026, 2718)
VALIDATION_SEEDS = (7001, 7002, 7003, 7004, 7005)
BOOTSTRAP_SEED = 20260924
PROTOCOL_VERSION = "IJPR-round2-v1-frozen-20260924"
RUN_ID_DEFAULT = "round2_20260924_frozen"
PRIMARY_CFG_KWARGS = {
    "active_backlog_trigger_ratio": 0.20,
    "active_alt_corridor_units": 200.0,
    "active_reserve_release_frac": 0.30,
}
THRESHOLDS = OrderedDict((("moderate", model.MODERATE), ("severe", model.SEVERE)))
LIBRARY_ORDER = (
    "FST-original",
    "MI-FST",
    "RST",
    "MI-C",
    "MI-M",
    "MI-MD",
    "MI-MDA",
)
CAPACITY_TARGETS_A = ("T2P", "S1A", "S1B", "S1C", "corridor")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8", newline="\n")


def _write_csv(rows: Sequence[Mapping[str, Any]] | pd.DataFrame, path: Path) -> None:
    dataframe = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    dataframe.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_canonical_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts)
    return int(_sha256_text(token)[:16], 16) % (2**32)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _env_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "unavailable"
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except ImportError:
        versions["matplotlib"] = "unavailable"
    return versions


def _event_dict(event: model.Event) -> dict[str, Any]:
    return {
        "target": event.target,
        "magnitude": float(event.magnitude),
        "start_week": int(event.start_week),
        "duration_weeks": int(event.duration_weeks),
        "recovery_weeks": int(event.recovery_weeks),
    }


def _event_key(events: Sequence[model.Event], cfg: Optional[model.NetworkConfig] = None) -> str:
    normalized = sorted(
        [
            {
                **_event_dict(event),
                "target": model._profile_target(event.target, cfg) if cfg is not None else event.target,
            }
            for event in events
        ],
        key=lambda row: (
            row["target"],
            row["start_week"],
            row["duration_weeks"],
            row["recovery_weeks"],
            row["magnitude"],
        ),
    )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _event_spec(events: Sequence[model.Event]) -> str:
    return json.dumps([_event_dict(event) for event in events], separators=(",", ":"))


def _labels_text(labels: Sequence[str]) -> str:
    return "; ".join(labels) if labels else model.UNCLASSIFIED


def _clone_pathway(
    pathway: model.Pathway,
    *,
    pathway_id: Optional[str] = None,
    family: Optional[str] = None,
    events: Optional[Sequence[model.Event]] = None,
    note: Optional[str] = None,
) -> model.Pathway:
    return model.Pathway(
        pathway_id or pathway.pathway_id,
        family or pathway.family,
        tuple(events or pathway.events),
        pathway.note if note is None else note,
    )


def _capacity(target: str, magnitude: float, start: int = 12, duration: int = 4, recovery: int = 3) -> model.Event:
    return model.Event(target, magnitude, start, duration, recovery)


def _demand(magnitude: float, start: int = 12, duration: int = 4, recovery: int = 2) -> model.Event:
    return model.Event("demand", magnitude, start, duration, recovery)


def _rename_mi(library: Sequence[model.Pathway]) -> list[model.Pathway]:
    return [_clone_pathway(pathway, pathway_id=f"FST-MI-{i:02d}") for i, pathway in enumerate(library, 1)]


def _replace_events(library: Sequence[model.Pathway], index: int, events: Sequence[model.Event], family: Optional[str] = None) -> None:
    original = library[index - 1]
    library[index - 1] = _clone_pathway(original, events=events, family=family)


def _build_candidate_libraries() -> tuple[dict[str, list[model.Pathway]], list[dict[str, Any]]]:
    original_fst = model.fst_library()
    original_mi_archived = model.fst_enhanced_library()
    original_mi = _rename_mi(original_mi_archived)
    rst = model.rst_candidate_library()
    variants: dict[str, list[model.Pathway]] = {
        "FST-original": original_fst,
        "MI-FST": original_mi,
        "RST": rst,
    }
    mi_c = list(original_mi)
    for idx, t2p, corridor in (
        (23, 0.25, 0.25),
        (24, 0.25, 0.35),
        (25, 0.35, 0.25),
        (26, 0.35, 0.35),
    ):
        _replace_events(mi_c, idx, (_capacity("T2P", t2p), _capacity("corridor", corridor)), "Simultaneous T2P-corridor")
    variants["MI-C"] = mi_c

    mi_m = list(mi_c)
    for idx, supplier in zip(range(31, 37), ("S1A", "S1A", "S1B", "S1B", "S1C", "S1C")):
        t2p = 0.25 if idx % 2 else 0.35
        _replace_events(mi_m, idx, (_capacity("T2P", t2p), _capacity(supplier, 0.40)), "Simultaneous T2P-tier-1")
    for idx, t2p, demand_magnitude in (
        (37, 0.25, 0.15),
        (38, 0.25, 0.25),
        (39, 0.35, 0.15),
        (40, 0.35, 0.25),
    ):
        _replace_events(mi_m, idx, (_capacity("T2P", t2p), _demand(demand_magnitude)), "Demand-T2P combination")
    variants["MI-M"] = mi_m

    mi_md = list(mi_m)
    for idx in range(23, 41):
        old = mi_md[idx - 1]
        changed = [model.Event(e.target, e.magnitude, e.start_week, 3, e.recovery_weeks) for e in old.events]
        _replace_events(mi_md, idx, changed, old.family)
    variants["MI-MD"] = mi_md

    mi_mda = list(mi_md)
    for idx, corridor, demand_magnitude in (
        (13, 0.25, 0.15),
        (14, 0.25, 0.25),
        (15, 0.35, 0.15),
        (16, 0.35, 0.25),
    ):
        _replace_events(
            mi_mda,
            idx,
            (_capacity("corridor", corridor), _demand(demand_magnitude, duration=3)),
            "Simultaneous corridor-demand",
        )
    variants["MI-MDA"] = mi_mda

    if any(len(library) != 50 for library in variants.values()):
        raise AssertionError("Every candidate library must contain 50 pathways")
    manifest_rows: list[dict[str, Any]] = []
    variant_parent = {"MI-FST": "", "MI-C": "MI-FST", "MI-M": "MI-C", "MI-MD": "MI-M", "MI-MDA": "MI-MD"}
    parent_by_variant = {"MI-FST": original_mi_archived, "MI-C": original_mi, "MI-M": mi_c, "MI-MD": mi_m, "MI-MDA": mi_md}
    for variant, library in variants.items():
        for pathway in library:
            if variant == "MI-FST":
                parent_id = f"FST-MB-{int(pathway.pathway_id.rsplit('-', 1)[1]):02d}"
                parent_pathway = parent_by_variant[variant][int(pathway.pathway_id.rsplit('-', 1)[1]) - 1]
            elif variant in variant_parent:
                parent_id = pathway.pathway_id
                parent_pathway = parent_by_variant[variant][int(pathway.pathway_id.rsplit('-', 1)[1]) - 1]
            else:
                parent_id = ""
                parent_pathway = None
            changed: list[str] = []
            if parent_pathway is not None:
                if _event_key(parent_pathway.events) != _event_key(pathway.events):
                    changed = ["events"]
                if parent_pathway.family != pathway.family:
                    changed.append("generation_family")
            manifest_rows.append(
                {
                    "variant_id": variant,
                    "parent_variant_id": variant_parent.get(variant, ""),
                    "candidate_id": pathway.pathway_id,
                    "parent_candidate_id": parent_id,
                    "generation_family": pathway.family,
                    "event_count": len(pathway.events),
                    "event_specification": _event_spec(pathway.events),
                    "canonical_event_key": _event_key(pathway.events),
                    "changed_fields": ";".join(changed),
                }
            )
    return variants, manifest_rows


@dataclass
class AttributionRecord:
    labels: tuple[str, ...]
    crossed: bool
    first_week: Optional[int]
    prefix_indices: tuple[int, ...]
    single_crossings: tuple[int, ...]
    hcd_removal_elements: tuple[str, ...]
    direct_impact_crossing: Optional[bool]
    rlto_support_pairs: tuple[tuple[int, int], ...]
    rlto_pair_iterations: int = 0


class InstrumentedEvaluator:
    """Shared simulation cache and complete request/execution accounting."""

    def __init__(self, cfg: model.NetworkConfig) -> None:
        self.cfg = cfg
        self.cache: dict[tuple[str, bool, bool], model.SimulationResult] = {}
        self.requests: list[dict[str, Any]] = []
        self.executions: list[dict[str, Any]] = []
        self.request_index = 0

    def simulate(
        self,
        events: Sequence[model.Event],
        *,
        active: bool,
        direct: bool,
        purpose: str,
        method: str,
        candidate_id: str,
        threshold: str,
        event_indices: Sequence[int] = (),
    ) -> model.SimulationResult:
        key = (_event_key(events, self.cfg), active, direct)
        self.request_index += 1
        request_id = self.request_index
        cache_hit = key in self.cache
        if cache_hit:
            result = self.cache[key]
            elapsed = 0.0
        else:
            started = time.perf_counter()
            result = model.run_simulation(events, self.cfg, active_mode=active, direct_impact_mode=direct)
            elapsed = time.perf_counter() - started
            self.cache[key] = result
            self.executions.append(
                {
                    "execution_id": len(self.executions) + 1,
                    "request_id": request_id,
                    "method": method,
                    "candidate_id": candidate_id,
                    "threshold": threshold,
                    "purpose": purpose,
                    "active": active,
                    "direct_impact": direct,
                    "event_indices": ";".join(str(i) for i in event_indices),
                    "event_specification": _event_spec(events),
                    "elapsed_seconds": elapsed,
                }
            )
        self.requests.append(
            {
                "request_id": request_id,
                "method": method,
                "candidate_id": candidate_id,
                "threshold": threshold,
                "purpose": purpose,
                "active": active,
                "direct_impact": direct,
                "event_indices": ";".join(str(i) for i in event_indices),
                "event_specification": _event_spec(events),
                "cache_hit": cache_hit,
                "elapsed_seconds": elapsed,
            }
        )
        return result


def _attribution(
    evaluator: InstrumentedEvaluator,
    events: Sequence[model.Event],
    cfg: model.NetworkConfig,
    threshold: model.FailureThreshold,
    threshold_name: str,
    method: str,
    candidate_id: str,
    passive: model.SimulationResult,
) -> AttributionRecord:
    crossed, first_week = model.threshold_cross_info(passive.service_levels, threshold)
    if not crossed or first_week is None:
        return AttributionRecord((), False, None, (), (), (), None, ())
    prefix_indices = tuple(i for i, event in enumerate(events) if event.start_week <= first_week)

    def subset_crosses(indices: Iterable[int], *, direct: bool = False, purpose: str) -> bool:
        index_tuple = tuple(sorted(indices))
        subset = [events[index] for index in index_tuple]
        result = evaluator.simulate(
            subset,
            active=False,
            direct=direct,
            purpose=purpose,
            method=method,
            candidate_id=candidate_id,
            threshold=threshold_name,
            event_indices=index_tuple,
        )
        return model.threshold_cross_info(result.service_levels, threshold)[0]

    single_crossings = tuple(
        i for i in prefix_indices if subset_crosses((i,), purpose="singleton")
    )
    affected_elements = {
        model._profile_target(events[i].target, cfg)
        for i in prefix_indices
        if model._profile_target(events[i].target, cfg) in cfg.upstream_nodes
    }
    hcd_removals: list[str] = []
    for element in sorted(affected_elements):
        if len(model._channels_for_element(element, cfg)) < 2:
            continue
        remaining = tuple(i for i in prefix_indices if model._profile_target(events[i].target, cfg) != element)
        if not subset_crosses(remaining, purpose="upstream_node_deletion"):
            hcd_removals.append(element)
    has_upstream = any(model._profile_target(events[i].target, cfg) in cfg.upstream_nodes for i in prefix_indices)
    direct_impact_crossing: Optional[bool] = None
    if has_upstream:
        direct_impact_crossing = subset_crosses(prefix_indices, direct=True, purpose="direct_impact")
    rlto_pairs: list[tuple[int, int]] = []
    rlto_pair_iterations = 0
    for first_index in prefix_indices:
        for second_index in prefix_indices:
            if first_index == second_index:
                continue
            rlto_pair_iterations += 1
            first = events[first_index]
            second = events[second_index]
            if first.start_week > second.start_week or not model._recovery_overlap(first, second):
                continue
            remaining = tuple(i for i in prefix_indices if i != second_index)
            if not subset_crosses(remaining, purpose="later_event_deletion"):
                rlto_pairs.append((first_index, second_index))
    detector_results = {
        model.CD: bool(single_crossings),
        model.HCD: bool(hcd_removals),
        model.CMD: len(prefix_indices) >= 2 and not single_crossings,
        model.PAC: has_upstream and direct_impact_crossing is False,
        model.RL_TO: bool(rlto_pairs),
    }
    labels = tuple(family for family in model.MECHANISM_FAMILIES if detector_results[family])
    return AttributionRecord(
        labels,
        True,
        first_week,
        prefix_indices,
        single_crossings,
        tuple(hcd_removals),
        direct_impact_crossing,
        tuple(rlto_pairs),
        rlto_pair_iterations,
    )


def _trajectory_rows(
    method: str,
    topology: str,
    candidate_id: str,
    mode: str,
    result: model.SimulationResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for week, trace in enumerate(result.trace):
        rows.append(
            {
                "method": method,
                "topology": topology,
                "candidate_id": candidate_id,
                "mode": mode,
                "week": week,
                "service_level": result.service_levels[week],
                "backlog": result.backlog_series[week],
                "reserve": result.reserve_series[week + 1],
                "rerouted_flow": result.rerouted_series[week],
                "trace": json.dumps(trace, separators=(",", ":")),
            }
        )
    return rows


def _evaluate_library(
    method: str,
    pathways: Sequence[model.Pathway],
    topology: str,
    cfg: model.NetworkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, InstrumentedEvaluator]:
    evaluator = InstrumentedEvaluator(cfg)
    result_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for pathway in pathways:
        passive = evaluator.simulate(
            pathway.events,
            active=False,
            direct=False,
            purpose="candidate_passive",
            method=method,
            candidate_id=pathway.pathway_id,
            threshold="all",
        )
        moderate_cross, moderate_week = model.threshold_cross_info(passive.service_levels, model.MODERATE)
        severe_cross, severe_week = model.threshold_cross_info(passive.service_levels, model.SEVERE)
        active: Optional[model.SimulationResult] = None
        if moderate_cross or severe_cross:
            active = evaluator.simulate(
                pathway.events,
                active=True,
                direct=False,
                purpose="candidate_active",
                method=method,
                candidate_id=pathway.pathway_id,
                threshold="all",
            )
        trajectory_rows.extend(_trajectory_rows(method, topology, pathway.pathway_id, "passive", passive))
        if active is not None:
            trajectory_rows.extend(_trajectory_rows(method, topology, pathway.pathway_id, "active", active))
        for threshold_name, threshold in THRESHOLDS.items():
            passive_cross, first_week = model.threshold_cross_info(passive.service_levels, threshold)
            active_cross = None
            attr = AttributionRecord((), False, None, (), (), (), None, ())
            if passive_cross:
                if active is None:
                    raise AssertionError("Active trajectory missing for a crossing candidate")
                active_cross = model.threshold_cross_info(active.service_levels, threshold)[0]
                attr = _attribution(evaluator, pathway.events, cfg, threshold, threshold_name, method, pathway.pathway_id, passive)
                for event_index in range(len(pathway.events)):
                    counterfactual_rows.append(
                        {
                            "method": method,
                            "topology": topology,
                            "candidate_id": pathway.pathway_id,
                            "threshold": threshold_name,
                            "counterfactual_type": "constituent_single_event",
                            "event_indices": str(event_index),
                            "event_specification": _event_spec((pathway.events[event_index],)),
                            "crossing": event_index in attr.single_crossings,
                        }
                    )
                counterfactual_rows.append(
                    {
                        "method": method,
                        "topology": topology,
                        "candidate_id": pathway.pathway_id,
                        "threshold": threshold_name,
                        "counterfactual_type": "support_evidence",
                        "event_indices": ";".join(str(i) for i in attr.prefix_indices),
                        "event_specification": _event_spec(pathway.events),
                        "crossing": True,
                        "hcd_removal_elements": ";".join(attr.hcd_removal_elements),
                        "direct_impact_crossing": attr.direct_impact_crossing,
                        "rlto_support_pairs": ";".join(f"{a}:{b}" for a, b in attr.rlto_support_pairs),
                    }
                )
            passive_shortfall = model.pathway_severity(passive.service_levels, threshold) if passive_cross else 0.0
            active_shortfall = model.pathway_severity(active.service_levels, threshold) if active_cross and active is not None else 0.0
            gamma_raw = 1.0 - active_shortfall / passive_shortfall if passive_shortfall > 0 and active is not None else None
            result_rows.append(
                {
                    "method": method,
                    "topology": topology,
                    "variant_id": method,
                    "candidate_id": pathway.pathway_id,
                    "parent_candidate_id": pathway.pathway_id,
                    "generation_family": pathway.family,
                    "threshold": threshold_name,
                    "passive_failure": passive_cross,
                    "active_failure": active_cross,
                    "first_cross_week": first_week,
                    "severity": passive_shortfall,
                    "active_severity": active_shortfall,
                    "C0": passive_shortfall if passive_cross else None,
                    "Cadm": active_shortfall if passive_cross else None,
                    "gamma_raw": gamma_raw,
                    "gamma_clipped": None if gamma_raw is None else float(np.clip(gamma_raw, 0.0, 1.0)),
                    "active_worsens_shortfall": None if gamma_raw is None else bool(gamma_raw < -1e-12),
                    "plausibility": model.pathway_plausibility(pathway),
                    "mechanism_signature": _labels_text(attr.labels),
                    "mechanism_label_count": len(attr.labels),
                    "passive_active_classification": (
                        model.STRUCTURAL if active_cross else model.ADAPTATION_LIMITED if passive_cross else model.NON_FAILURE
                    ),
                    "causal_prefix_indices": ";".join(str(i) for i in attr.prefix_indices),
                    "single_event_crossings": ";".join(str(i) for i in attr.single_crossings),
                    "hcd_removal_elements": ";".join(attr.hcd_removal_elements),
                    "direct_impact_crossing": attr.direct_impact_crossing,
                    "rlto_support_pairs": ";".join(f"{a}:{b}" for a, b in attr.rlto_support_pairs),
                    "rlto_pair_iterations": attr.rlto_pair_iterations,
                    "event_specification": _event_spec(pathway.events),
                    "canonical_event_key": _event_key(pathway.events),
                }
            )
    result_df = pd.DataFrame(result_rows)
    request_df = pd.DataFrame(evaluator.requests)
    execution_df = pd.DataFrame(evaluator.executions)
    trajectory_df = pd.DataFrame(trajectory_rows)
    return result_df, request_df, execution_df, pd.DataFrame(counterfactual_rows), pd.DataFrame(trajectory_rows), evaluator


def _parse_event_spec(value: str) -> tuple[model.Event, ...]:
    rows = json.loads(value)
    return tuple(model.Event(**row) for row in rows)


def _read_archived_event_rows(topology: str, seed: int, root: Path) -> Optional[pd.DataFrame]:
    candidates = [
        root / "02_revised_analysis_outputs" / "raw_seed_runs" / f"topology_{topology}_independent_seed_{seed}_n_1000.csv",
        root / "baseline_verification" / "raw_seed_runs" / f"topology_{topology}_independent_seed_{seed}_n_1000.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def _mc_cohort(
    topology: str,
    cohort: str,
    seeds: Sequence[int],
    replications: int,
    cfg: model.NetworkConfig,
    root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        archived = _read_archived_event_rows(topology, seed, root) if cohort == "archived" else None
        if archived is not None and len(archived) == replications:
            event_specs = archived["event_specification"].tolist()
            expected_moderate = archived["moderate_passive_failure"].tolist()
        else:
            root_sequence = np.random.SeedSequence(seed)
            event_specs = []
            for child_seed in root_sequence.spawn(replications):
                rng = np.random.default_rng(child_seed)
                event_specs.append(_event_spec(model.generate_random_events(rng, cfg)))
            expected_moderate = [None] * replications
        for replication, (event_spec, expected) in enumerate(zip(event_specs, expected_moderate), 1):
            events = _parse_event_spec(event_spec)
            passive = model.run_simulation(events, cfg, active_mode=False)
            moderate_cross, moderate_week = model.threshold_cross_info(passive.service_levels, model.MODERATE)
            severe_cross, severe_week = model.threshold_cross_info(passive.service_levels, model.SEVERE)
            if expected is not None and bool(expected) != moderate_cross:
                raise AssertionError(f"Archived moderate replay mismatch for {topology}, seed {seed}, replication {replication}")
            active = model.run_simulation(events, cfg, active_mode=True) if moderate_cross else None
            for threshold_name, threshold, crossed, first_week in (
                ("moderate", model.MODERATE, moderate_cross, moderate_week),
                ("severe", model.SEVERE, severe_cross, severe_week),
            ):
                attr = model.attribute_mechanisms(events, cfg, threshold=threshold, passive_result=passive) if crossed else None
                active_cross = model.threshold_cross_info(active.service_levels, threshold)[0] if crossed and active is not None else None
                rows.append(
                    {
                        "cohort": cohort,
                        "topology": topology,
                        "seed": seed,
                        "replication": replication,
                        "threshold": threshold_name,
                        "event_specification": event_spec,
                        "canonical_event_key": _event_key(events, cfg),
                        "passive_failure": crossed,
                        "active_failure": active_cross,
                        "first_cross_week": first_week,
                        "mechanism_signature": _labels_text(attr.labels) if attr is not None else model.UNCLASSIFIED,
                        "unclassified": bool(crossed and (attr is None or not attr.labels)),
                        "passive_active_classification": (
                            model.STRUCTURAL if active_cross else model.ADAPTATION_LIMITED if crossed else model.NON_FAILURE
                        ),
                        "worst_service_level": passive.worst_service_level,
                        "moderate_passive_failure": moderate_cross,
                        "severe_passive_failure": severe_cross,
                    }
                )
    return pd.DataFrame(rows)


def _signature_set(result_df: pd.DataFrame, threshold: str) -> set[str]:
    rows = result_df[(result_df["threshold"] == threshold) & (result_df["passive_failure"])]
    return {str(value) for value in rows["mechanism_signature"] if value != model.UNCLASSIFIED}


def _coverage_metrics(failing: pd.DataFrame, discovered: set[str]) -> dict[str, Any]:
    valid = failing["mechanism_signature"].ne(model.UNCLASSIFIED)
    covered = valid & failing["mechanism_signature"].isin(discovered)
    structural = failing["passive_active_classification"].eq(model.STRUCTURAL)
    focal = failing["mechanism_signature"].map(
        lambda value: any(label in str(value).split("; ") for label in (model.CMD, model.PAC, model.RL_TO))
    )
    unique_observed = set(failing.loc[valid, "mechanism_signature"])
    return {
        "numerator": int(covered.sum()),
        "denominator": int(len(failing)),
        "coverage": float(covered.mean()) if len(failing) else np.nan,
        "structural_only_numerator": int((covered & structural).sum()),
        "structural_only_denominator": int(structural.sum()),
        "structural_only_coverage": float((covered & structural).sum() / structural.sum()) if structural.sum() else np.nan,
        "focal_numerator": int((covered & focal).sum()),
        "focal_denominator": int(focal.sum()),
        "focal_coverage": float((covered & focal).sum() / focal.sum()) if focal.sum() else np.nan,
        "observed_unique_signature_numerator": len(unique_observed & discovered),
        "observed_unique_signature_denominator": len(unique_observed),
        "observed_unique_signature_coverage": len(unique_observed & discovered) / len(unique_observed) if unique_observed else np.nan,
        "unclassified_failures": int((~valid).sum()),
    }


def _coverage_rows(mc: pd.DataFrame, result_frames: Mapping[str, pd.DataFrame], bootstrap_reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    omission_rows: list[dict[str, Any]] = []
    for (cohort, topology, threshold), group in mc.groupby(["cohort", "topology", "threshold"], sort=False):
        method_signatures = {method: _signature_set(frame[frame["topology"] == topology], threshold) for method, frame in result_frames.items()}
        failing = group[group["passive_failure"]].copy()
        all_records = {int(seed): stratum.reset_index(drop=True) for seed, stratum in group.groupby("seed")}
        for method, discovered in method_signatures.items():
            metrics = _coverage_metrics(failing, discovered)
            seed_ratios = []
            for seed, stratum in all_records.items():
                sf = stratum[stratum["passive_failure"]]
                sn = int((sf["mechanism_signature"].isin(discovered) & sf["mechanism_signature"].ne(model.UNCLASSIFIED)).sum())
                sd = int(len(sf))
                seed_ratios.append(sn / sd if sd else np.nan)
            rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, _stable_seed(cohort, topology, threshold, method)]))
            boot_n = np.zeros(bootstrap_reps, dtype=np.int64)
            boot_d = np.zeros(bootstrap_reps, dtype=np.int64)
            for stratum in all_records.values():
                population_size = len(stratum)
                failures = stratum[stratum["passive_failure"]]
                failure_count = len(failures)
                covered_count = int((failures["mechanism_signature"].isin(discovered) & failures["mechanism_signature"].ne(model.UNCLASSIFIED)).sum())
                category_counts = np.array(
                    [covered_count, failure_count - covered_count, population_size - failure_count],
                    dtype=np.int64,
                )
                sampled = rng.multinomial(population_size, category_counts / population_size, size=bootstrap_reps)
                boot_n += sampled[:, 0]
                boot_d += sampled[:, 0] + sampled[:, 1]
            boot_values = np.divide(boot_n, boot_d, out=np.full(bootstrap_reps, np.nan), where=boot_d != 0)
            summary_rows.append(
                {
                    "cohort": cohort,
                    "topology": topology,
                    "threshold": threshold,
                    "method": method,
                    **metrics,
                    "bootstrap_lower_95": float(np.nanpercentile(boot_values, 2.5)) if len(boot_values) else np.nan,
                    "bootstrap_upper_95": float(np.nanpercentile(boot_values, 97.5)) if len(boot_values) else np.nan,
                    "mean_seed_coverage": float(np.nanmean(seed_ratios)) if seed_ratios else np.nan,
                    "sd_seed_coverage": float(np.nanstd(seed_ratios, ddof=1)) if len(seed_ratios) > 1 else np.nan,
                    "discovered_signature_count": len(discovered),
                }
            )
            for seed, stratum in all_records.items():
                sf = stratum[stratum["passive_failure"]]
                sn = int((sf["mechanism_signature"].isin(discovered) & sf["mechanism_signature"].ne(model.UNCLASSIFIED)).sum())
                omission_rows.append(
                    {
                        "cohort": cohort,
                        "topology": topology,
                        "threshold": threshold,
                        "method": method,
                        "seed": seed,
                        "numerator": sn,
                        "denominator": len(sf),
                        "coverage": sn / len(sf) if len(sf) else np.nan,
                    }
                )
        rst_discovered = method_signatures["RST"]
        for comparator in (method for method in LIBRARY_ORDER if method != "RST"):
            comparator_discovered = method_signatures[comparator]
            delta_rst = np.zeros(bootstrap_reps, dtype=np.int64)
            delta_cmp = np.zeros(bootstrap_reps, dtype=np.int64)
            delta_denominator = np.zeros(bootstrap_reps, dtype=np.int64)
            observed_rst_numerator = 0
            observed_comparator_numerator = 0
            observed_denominator = 0
            per_seed_deltas: list[float] = []
            rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, 991, _stable_seed(cohort, topology, threshold, comparator)]))
            for stratum in all_records.values():
                population_size = len(stratum)
                failures = stratum[stratum["passive_failure"]]
                valid = failures["mechanism_signature"].ne(model.UNCLASSIFIED)
                rst_success = valid & failures["mechanism_signature"].isin(rst_discovered)
                cmp_success = valid & failures["mechanism_signature"].isin(comparator_discovered)
                both = int((rst_success & cmp_success).sum())
                rst_only = int((rst_success & ~cmp_success).sum())
                cmp_only = int((~rst_success & cmp_success).sum())
                neither = int((~rst_success & ~cmp_success).sum())
                nonfailure = population_size - len(failures)
                valid_count = int(valid.sum())
                observed_rst_numerator += int(rst_success.sum())
                observed_comparator_numerator += int(cmp_success.sum())
                observed_denominator += valid_count
                if valid_count:
                    per_seed_deltas.append(100.0 * (int(rst_success.sum()) - int(cmp_success.sum())) / valid_count)
                category_counts = np.array([both, rst_only, cmp_only, neither, nonfailure], dtype=np.int64)
                sampled = rng.multinomial(population_size, category_counts / population_size, size=bootstrap_reps)
                delta_rst += sampled[:, 0] + sampled[:, 1]
                delta_cmp += sampled[:, 0] + sampled[:, 2]
                delta_denominator += sampled[:, 0] + sampled[:, 1] + sampled[:, 2] + sampled[:, 3]
            delta_values = np.divide(
                100.0 * (delta_rst - delta_cmp),
                delta_denominator,
                out=np.full(bootstrap_reps, np.nan),
                where=delta_denominator != 0,
            )
            failing_signatures = set(failing["mechanism_signature"])
            gained = sorted(rst_discovered - comparator_discovered)
            lost = sorted(comparator_discovered - rst_discovered)
            for direction, signatures in (("gained", gained), ("lost", lost)):
                for signature in signatures:
                    support_method = "RST" if direction == "gained" else comparator
                    support = ";".join(
                        result_frames[support_method].loc[
                            (result_frames[support_method]["topology"] == topology)
                            & (result_frames[support_method]["threshold"] == threshold)
                            & (result_frames[support_method]["mechanism_signature"] == signature),
                            "candidate_id",
                        ].astype(str).tolist()
                    )
                    frequency = int((failing["mechanism_signature"] == signature).sum())
                    difference_rows.append(
                        {
                            "cohort": cohort,
                            "topology": topology,
                            "threshold": threshold,
                            "comparator": comparator,
                            "direction": direction,
                            "signature": signature,
                            "candidate_support_ids": support,
                            "support_method": support_method,
                            "monte_carlo_frequency": frequency,
                        }
                    )
            difference_rows.append(
                {
                    "cohort": cohort,
                    "topology": topology,
                    "threshold": threshold,
                    "comparator": comparator,
                    "direction": "paired_delta_summary",
                    "signature": "",
                    "candidate_support_ids": "",
                    "support_method": "",
                    "monte_carlo_frequency": np.nan,
                    "rst_numerator": observed_rst_numerator,
                    "comparator_numerator": observed_comparator_numerator,
                    "paired_denominator": observed_denominator,
                    "observed_delta_pp": (
                        100.0 * (observed_rst_numerator - observed_comparator_numerator) / observed_denominator
                        if observed_denominator
                        else np.nan
                    ),
                    "per_seed_delta_mean_pp": float(np.mean(per_seed_deltas)) if per_seed_deltas else np.nan,
                    "per_seed_delta_sd_pp": float(np.std(per_seed_deltas, ddof=1)) if len(per_seed_deltas) > 1 else np.nan,
                    "delta_pp_lower_95": float(np.nanpercentile(delta_values, 2.5)) if len(delta_values) else np.nan,
                    "delta_pp_upper_95": float(np.nanpercentile(delta_values, 97.5)) if len(delta_values) else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows), pd.DataFrame(omission_rows)


def _reference_grid(cfg: model.NetworkConfig, topology: str) -> list[model.Pathway]:
    targets = list(cfg.upstream_nodes) + list(cfg.tier1_nodes) + ["corridor", "demand"]
    capacity_targets = set(cfg.upstream_nodes) | set(cfg.tier1_nodes) | {"corridor"}
    magnitudes = {target: (0.25, 0.35, 0.50) if target in capacity_targets else (0.15, 0.25, 0.40) for target in targets}
    paths: list[model.Pathway] = []
    index = 1
    for target in targets:
        for magnitude in magnitudes[target]:
            for duration in (2, 3, 4):
                event = _capacity(target, magnitude, duration=duration) if target in capacity_targets else _demand(magnitude, duration=duration)
                paths.append(model.Pathway(f"REF-{topology}-S-{index:04d}", "single event", (event,)))
                index += 1
    for first, second in itertools.combinations(targets, 2):
        for first_magnitude in magnitudes[first]:
            for second_magnitude in magnitudes[second]:
                for first_duration in (2, 3, 4):
                    for second_duration in (2, 3, 4):
                        first_event = _capacity(first, first_magnitude, duration=first_duration) if first in capacity_targets else _demand(first_magnitude, duration=first_duration)
                        second_event = _capacity(second, second_magnitude, duration=second_duration) if second in capacity_targets else _demand(second_magnitude, duration=second_duration)
                        paths.append(model.Pathway(f"REF-{topology}-P-{index:04d}", "simultaneous pair", (first_event, second_event)))
                        index += 1
                        second_start = 12 + first_duration + 1
                        second_event_seq = _capacity(second, second_magnitude, start=second_start, duration=second_duration) if second in capacity_targets else _demand(second_magnitude, start=second_start, duration=second_duration)
                        paths.append(model.Pathway(f"REF-{topology}-Q-{index:04d}", "sequential pair", (first_event, second_event_seq)))
                        index += 1
                        reverse_first = _capacity(second, second_magnitude, duration=second_duration) if second in capacity_targets else _demand(second_magnitude, duration=second_duration)
                        reverse_second_start = 12 + second_duration + 1
                        reverse_second = _capacity(first, first_magnitude, start=reverse_second_start, duration=first_duration) if first in capacity_targets else _demand(first_magnitude, start=reverse_second_start, duration=first_duration)
                        paths.append(model.Pathway(f"REF-{topology}-Q-{index:04d}", "sequential pair", (reverse_first, reverse_second)))
                        index += 1
    expected = 3699 if topology == "A" else 5166
    if len(paths) != expected:
        raise AssertionError(f"Reference grid {topology} expected {expected}, found {len(paths)}")
    return paths


def _evaluate_reference_grid(
    topology: str,
    paths: Sequence[model.Pathway],
    cfg: model.NetworkConfig,
) -> tuple[pd.DataFrame, InstrumentedEvaluator]:
    evaluator = InstrumentedEvaluator(cfg)
    rows: list[dict[str, Any]] = []
    for path in paths:
        passive = evaluator.simulate(path.events, active=False, direct=False, purpose="reference_passive", method="reference-grid", candidate_id=path.pathway_id, threshold="all")
        for threshold_name, threshold in THRESHOLDS.items():
            crossed, first_week = model.threshold_cross_info(passive.service_levels, threshold)
            attr = _attribution(evaluator, path.events, cfg, threshold, threshold_name, "reference-grid", path.pathway_id, passive) if crossed else None
            rows.append(
                {
                    "topology": topology,
                    "reference_id": path.pathway_id,
                    "reference_family": path.family,
                    "threshold": threshold_name,
                    "passive_failure": crossed,
                    "first_cross_week": first_week,
                    "mechanism_signature": _labels_text(attr.labels) if attr else model.UNCLASSIFIED,
                    "unclassified": bool(crossed and (attr is None or not attr.labels)),
                    "event_specification": _event_spec(path.events),
                    "canonical_event_key": _event_key(path.events, cfg),
                }
            )
    return pd.DataFrame(rows), evaluator


def _rank_scores(scores: Mapping[str, float], tolerance: float = 1e-12) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    previous_score: Optional[float] = None
    rank = 0
    for position, (element, score) in enumerate(ordered, 1):
        if previous_score is None or not math.isclose(score, previous_score, rel_tol=1e-9, abs_tol=tolerance):
            rank = position
        ranks[element] = rank
        previous_score = score
    return ranks


def _common_scores(
    result_df: pd.DataFrame,
    pathways_by_method: Mapping[str, Sequence[model.Pathway]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, group in result_df[result_df["threshold"] == "moderate"].groupby("method", sort=False):
        path_map = {pathway.pathway_id: pathway for pathway in pathways_by_method.get(method, ())}
        failures = group[group["passive_failure"]].copy()
        severity_values = model._normalize(failures["severity"].astype(float).tolist())
        plausibility_values = model._normalize(failures["plausibility"].astype(float).tolist())
        weights = [0.625 * s + 0.375 * p for s, p in zip(severity_values, plausibility_values)]
        if weights and math.isclose(sum(weights), 0.0, abs_tol=1e-15):
            weights = [1.0] * len(weights)
        scores = {element: 0.0 for element in ("T2P", "corridor", "S1A", "S1B", "S1C")}
        eligible_weight = 0.0
        for (_, row), weight in zip(failures.iterrows(), weights):
            pathway = path_map.get(row["candidate_id"])
            elements = model.pathway_elements(pathway) if pathway is not None else ()
            if not elements:
                continue
            eligible_weight += weight
            for element in elements:
                if element in scores:
                    scores[element] += weight / len(elements)
        if eligible_weight > 0:
            scores = {element: score / eligible_weight for element, score in scores.items()}
        ranks = _rank_scores(scores)
        severity_min = float(failures["severity"].min()) if len(failures) else np.nan
        severity_max = float(failures["severity"].max()) if len(failures) else np.nan
        plausibility_min = float(failures["plausibility"].min()) if len(failures) else np.nan
        plausibility_max = float(failures["plausibility"].max()) if len(failures) else np.nan
        for element, score in scores.items():
            rows.append(
                {
                    "method": method,
                    "scoring_rule": "common severity/plausibility 0.625/0.375; equal element allocation",
                    "failure_candidates": len(failures),
                    "eligible_weight": eligible_weight,
                    "severity_min": severity_min,
                    "severity_max": severity_max,
                    "plausibility_min": plausibility_min,
                    "plausibility_max": plausibility_max,
                    "element": element,
                    "element_score": score,
                    "element_rank": ranks[element],
                }
            )
    return rows


def _scaling_cfg(n: int, horizon: int, architecture: str) -> model.NetworkConfig:
    nodes = tuple(f"S1{chr(65 + i)}" if i < 26 else f"S1-{i+1}" for i in range(n))
    shares = tuple(1.0 / n for _ in nodes)
    if architecture == "shared":
        upstream_nodes = ("T2P",)
        upstream_map = tuple("T2P" for _ in nodes)
        upstream_capacities = ((1000.0 / 3.0) * n,)
    else:
        upstream_nodes = tuple(f"T2P-{i+1:03d}" for i in range(n // 3))
        upstream_map = tuple(upstream_nodes[i // 3] for i in range(n))
        upstream_capacities = tuple(1000.0 for _ in upstream_nodes)
    return model.NetworkConfig(
        horizon_weeks=horizon,
        demand_per_week=300.0 * n,
        tier1_nodes=nodes,
        tier1_allocation_shares=shares,
        tier1_capacity=400.0,
        tier1_initial_inventory=200.0,
        tier1_order_up_to=1100.0,
        t2p_capacity=1000.0,
        tier1_upstream_nodes=upstream_map,
        upstream_nodes=upstream_nodes,
        upstream_capacities=upstream_capacities,
        corridor_capacity=400.0 * n,
        protected_reserve_capacity=300.0 * n,
        active_backlog_trigger_ratio=0.20,
        active_alt_corridor_units=(200.0 / 3.0) * n,
        active_reserve_release_frac=0.30,
    )


def _scaling_workload(cfg: model.NetworkConfig, b: int, B: int, K: int, architecture: str) -> model.Pathway:
    first_target = cfg.upstream_nodes[0]
    first_magnitude = 0.45 + 0.10 * b / (B + 1) if K >= 2 else 0.65 + 0.10 * b / (B + 1)
    events: list[model.Event] = []
    upstream_count = max(1, K // 2)
    for index in range(upstream_count):
        target = cfg.upstream_nodes[index % len(cfg.upstream_nodes)]
        events.append(_capacity(target, first_magnitude if index == 0 else 0.25, duration=3, recovery=4))
    for _ in range(K - upstream_count):
        events.append(_capacity("corridor", 0.50, start=16, duration=2, recovery=3))
    return model.Pathway(f"SCALE-{architecture}-{K}-{B}-{b:04d}", "scaling workload", tuple(events))


def _process_rss() -> tuple[Optional[float], str]:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), "psutil rss"
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            process = ctypes.windll.kernel32.GetCurrentProcess()
            memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            memory_info.restype = ctypes.c_int
            success = memory_info(
                process,
                ctypes.byref(counters),
                ctypes.sizeof(counters),
            )
            if success:
                return counters.WorkingSetSize / (1024 * 1024), "Windows GetProcessMemoryInfo working set"
        except (AttributeError, OSError):
            pass
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1024 * 1024 if sys.platform == "darwin" else 1024
        return value / scale, "resource ru_maxrss"
    except (ImportError, AttributeError, OSError):
        return None, "unavailable"


def _run_scaling_cell(architecture: str, n: int, horizon: int, K: int, B: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = _scaling_cfg(n, horizon, architecture)
    warmup = [_scaling_workload(cfg, b, B, K, architecture) for b in range(1, min(5, B) + 1)]
    attribution_warmed = False
    for pathway in warmup:
        passive = model.run_simulation(pathway.events, cfg)
        if passive.crossed_moderate:
            model.run_simulation(pathway.events, cfg, active_mode=True)
            if not attribution_warmed:
                model.attribute_mechanisms(pathway.events, cfg, passive_result=passive)
                attribution_warmed = True
    raw_rows: list[dict[str, Any]] = []
    workload_rows = [{"architecture": architecture, "n": n, "horizon": horizon, "K": K, "B": B, "candidate_id": p.pathway_id, "event_specification": _event_spec(p.events)} for p in (_scaling_workload(cfg, b, B, K, architecture) for b in range(1, B + 1))]
    for repetition in range(1, 6):
        evaluator = InstrumentedEvaluator(cfg)
        started = time.perf_counter()
        passive_time = active_time = attribution_time = 0.0
        failures = 0
        prefix_events = 0
        attribution_pair_iterations = 0
        status = "completed"
        completed = 0
        baseline_memory, memory_measurement = _process_rss()
        peak_memory = baseline_memory
        for b in range(1, B + 1):
            if time.perf_counter() - started > 120.0:
                status = "timeout"
                break
            pathway = _scaling_workload(cfg, b, B, K, architecture)
            begin = time.perf_counter()
            passive = evaluator.simulate(pathway.events, active=False, direct=False, purpose="scaling_candidate_passive", method="scaling", candidate_id=pathway.pathway_id, threshold="moderate")
            passive_time += time.perf_counter() - begin
            if passive.crossed_moderate:
                failures += 1
                if time.perf_counter() - started > 120.0:
                    status = "timeout"
                    break
                begin = time.perf_counter()
                active = evaluator.simulate(pathway.events, active=True, direct=False, purpose="scaling_candidate_active", method="scaling", candidate_id=pathway.pathway_id, threshold="moderate")
                active_time += time.perf_counter() - begin
                if time.perf_counter() - started > 120.0:
                    status = "timeout"
                    break
                begin = time.perf_counter()
                attr = _attribution(evaluator, pathway.events, cfg, model.MODERATE, "moderate", "scaling", pathway.pathway_id, passive)
                attribution_time += time.perf_counter() - begin
                prefix_events += len(attr.prefix_indices)
                attribution_pair_iterations += attr.rlto_pair_iterations
            completed += 1
            current_memory = _process_rss()[0]
            if current_memory is not None:
                peak_memory = max(peak_memory or current_memory, current_memory)
            if time.perf_counter() - started > 120.0:
                status = "timeout"
                break
        raw_rows.append(
            {
                "architecture": architecture,
                "n": n,
                "horizon": horizon,
                "K": K,
                "B": B,
                "repetition": repetition,
                "status": status,
                "completed_candidates": completed,
                "failures": failures,
                "prefix_event_count": prefix_events,
                "attribution_pair_iterations": attribution_pair_iterations,
                "passive_seconds": passive_time,
                "active_seconds": active_time,
                "attribution_seconds": attribution_time,
                "end_to_end_seconds": time.perf_counter() - started,
                "requested_simulations": len(evaluator.requests),
                "unique_simulations": len(evaluator.executions),
                "cache_hits": sum(bool(row["cache_hit"]) for row in evaluator.requests),
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory,
                "incremental_peak_memory_mb": (
                    peak_memory - baseline_memory
                    if peak_memory is not None and baseline_memory is not None
                    else None
                ),
                "memory_measurement": f"{memory_measurement}; incremental over cell baseline",
                "node_count": len(cfg.tier1_nodes) + len(cfg.upstream_nodes) + 2,
                "edge_count": 2 * len(cfg.tier1_nodes) + 1,
            }
        )
    return raw_rows, workload_rows


def _existing_robustness_index(root: Path) -> pd.DataFrame:
    entries = [
        ("02_revised_analysis_outputs/sensitivity_threshold.csv", "threshold sensitivity", "reused; no rerun"),
        ("02_revised_analysis_outputs/sensitivity_response.csv", "response sensitivity", "reused; no rerun"),
        ("02_revised_analysis_outputs/sensitivity_severity_plausibility_weights.csv", "weight sensitivity", "reused; no rerun"),
        ("02_revised_analysis_outputs/sensitivity_dependence_summary.csv", "dependence structures", "reused; no rerun"),
        ("02_revised_analysis_outputs/robustness_topology_summary.csv", "seed and topology robustness", "reused; no rerun"),
        ("02_revised_analysis_outputs/runtime_scaling.csv", "archived runtime table", "reused; limited archived scope"),
    ]
    return pd.DataFrame(
        [
            {
                "path": path,
                "exists_in_worktree": (root / path).exists(),
                "reviewer_relevance": relevance,
                "reuse_status": status,
                "interpretation_limit": "does not establish network-size scaling" if "runtime" in path else "existing analysis retained",
            }
            for path, relevance, status in entries
        ]
    )


def _deterministic_threshold_sensitivity(
    libraries: Mapping[str, Sequence[model.Pathway]],
    cfg: model.NetworkConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = [(alpha, duration) for alpha in (0.90, 0.85, 0.80) for duration in (1, 2, 3, 4)]
    for method, pathways in libraries.items():
        evaluator = InstrumentedEvaluator(cfg)
        for alpha, duration in settings:
            threshold = model.FailureThreshold(f"SL<{alpha:.2f} for >={duration} periods", alpha, duration)
            signatures: set[str] = set()
            moderate_crossings = 0
            for path in pathways:
                passive = evaluator.simulate(path.events, active=False, direct=False, purpose="deterministic_threshold_passive", method=method, candidate_id=path.pathway_id, threshold=f"{alpha:.2f}-{duration}")
                crossed = model.threshold_cross_info(passive.service_levels, threshold)[0]
                if crossed:
                    moderate_crossings += 1
                    attr = _attribution(evaluator, path.events, cfg, threshold, f"{alpha:.2f}-{duration}", method, path.pathway_id, passive)
                    if attr.labels:
                        signatures.add(_labels_text(attr.labels))
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "duration": duration,
                    "candidate_count": len(pathways),
                    "threshold_crossings": moderate_crossings,
                    "full_signatures": " | ".join(sorted(signatures)),
                    "full_signature_count": len(signatures),
                }
            )
    return pd.DataFrame(rows)


def _summarize_scaling(scaling: pd.DataFrame) -> pd.DataFrame:
    if scaling.empty:
        return pd.DataFrame()
    group_columns = ["architecture", "n", "horizon", "K", "B"]
    all_summary = scaling.groupby(group_columns, as_index=False).agg(
        total_repetitions=("repetition", "count"),
        completed_repetitions=("status", lambda values: int((values == "completed").sum())),
        timeout_repetitions=("status", lambda values: int((values == "timeout").sum())),
        partial_timeout_repetitions=(
            "completed_candidates",
            lambda values: int(
                ((scaling.loc[values.index, "status"] == "timeout") & (values > 0)).sum()
            ),
        ),
        partial_timeout_candidates=(
            "completed_candidates",
            lambda values: int(
                values[scaling.loc[values.index, "status"] == "timeout"].sum()
            ),
        ),
    )
    completed = scaling[scaling["status"] == "completed"]
    if completed.empty:
        return all_summary
    completed_summary = completed.groupby(group_columns, as_index=False).agg(
        median_seconds=("end_to_end_seconds", "median"),
        iqr_seconds=(
            "end_to_end_seconds",
            lambda values: float(np.percentile(values, 75) - np.percentile(values, 25)),
        ),
        incremental_peak_memory_mb=("incremental_peak_memory_mb", "median"),
        peak_memory_mb=("peak_memory_mb", "median"),
        median_unique_simulations=("unique_simulations", "median"),
        median_cache_hits=("cache_hits", "median"),
        median_attribution_pair_iterations=("attribution_pair_iterations", "median"),
    )
    return all_summary.merge(completed_summary, on=group_columns, how="left", validate="one_to_one")


def _scaling_figure_data(scaling_summary: pd.DataFrame) -> pd.DataFrame:
    if scaling_summary.empty:
        return pd.DataFrame()
    axes = ("n", "horizon", "K", "B")
    reference = {"n": 30, "horizon": 52, "K": 4, "B": 50}
    rows: list[dict[str, Any]] = []
    for focal_axis in axes:
        subset = scaling_summary.copy()
        for axis in axes:
            if axis != focal_axis:
                subset = subset[subset[axis] == reference[axis]]
        for row in subset.to_dict("records"):
            row["focal_axis"] = focal_axis
            row["focal_value"] = row[focal_axis]
            rows.append(row)
    return pd.DataFrame(rows)


def _omission_figure_data(omission_catalog: pd.DataFrame) -> pd.DataFrame:
    if omission_catalog.empty:
        return pd.DataFrame()
    uncovered = omission_catalog[~omission_catalog["covered"].astype(bool)].copy()
    if uncovered.empty:
        return pd.DataFrame(
            columns=[
                "cohort",
                "topology",
                "threshold",
                "method",
                "signature",
                "omitted_frequency",
                "total_failures",
                "omission_fraction",
            ]
        )
    uncovered["omitted_frequency"] = uncovered["failure_frequency"].astype(int)
    uncovered["omission_fraction"] = np.divide(
        uncovered["omitted_frequency"],
        uncovered["total_failures"],
        out=np.zeros(len(uncovered), dtype=float),
        where=uncovered["total_failures"].to_numpy() != 0,
    )
    return uncovered[
        [
            "cohort",
            "topology",
            "threshold",
            "method",
            "signature",
            "omitted_frequency",
            "total_failures",
            "omission_fraction",
        ]
    ].sort_values(
        ["cohort", "topology", "threshold", "method", "omitted_frequency", "signature"],
        ascending=[True, True, True, True, False, True],
    )


def _make_figures(
    output_dir: Path,
    differences: pd.DataFrame,
    omissions: pd.DataFrame,
    scaling: pd.DataFrame,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams["svg.hashsalt"] = "rst-ijpr-round2"
    paired = differences[differences["direction"] == "paired_delta_summary"].copy()
    if not paired.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for comparator, group in paired.groupby("comparator"):
            labels = [f"{row['cohort']} {row['topology']} {row['threshold']}" for _, row in group.iterrows()]
            observed = group["observed_delta_pp"].astype(float).to_numpy()
            lower = observed - group["delta_pp_lower_95"].astype(float).to_numpy()
            upper = group["delta_pp_upper_95"].astype(float).to_numpy() - observed
            ax.errorbar(labels, observed, yerr=[lower, upper], marker="o", linestyle="-", capsize=3, label=comparator)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("RST minus comparator coverage (percentage points)")
        ax.set_title("Paired coverage differences with bootstrap 95% intervals")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "coverage_differences.svg", metadata={"Date": None})
        plt.close(fig)
    if not omissions.empty:
        top = omissions.sort_values("omitted_frequency", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        values = top["omitted_frequency"].astype(float).tolist()
        labels = [f"{row['cohort']} {row['topology']} {row['threshold']}\n{row['method']}\n{row['signature']}" for _, row in top.iterrows()]
        ax.bar(range(len(values)), values)
        ax.set_ylabel("Uncovered failure frequency")
        ax.set_title("Observed signature omissions")
        ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right")
        fig.tight_layout()
        fig.savefig(figures_dir / "observed_omission_populations.svg", metadata={"Date": None})
        plt.close(fig)
    if not scaling.empty:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, axis, title in zip(axes, ("n", "horizon", "K", "B"), ("Network width n", "Horizon H", "Events K", "Budget B")):
            subset = scaling[(scaling["focal_axis"] == axis) & scaling["median_seconds"].notna()]
            if subset.empty:
                continue
            for architecture, group in subset.groupby("architecture"):
                ax.plot(group["focal_value"], group["median_seconds"], marker="o", label=architecture)
            ax.set_xlabel(title)
            ax.set_ylabel("Median completed-run seconds")
            ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "scaling_runtime.svg", metadata={"Date": None})
        plt.close(fig)


def _evidence_report(
    output_dir: Path,
    baseline: Mapping[str, Any],
    coverage: pd.DataFrame,
    differences: pd.DataFrame,
    bounded: pd.DataFrame,
    scaling: pd.DataFrame,
    source_hashes: Mapping[str, str],
) -> None:
    lines = [
        "# Second-round IJPR computational evidence report",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        "",
        "## Baseline freeze",
        "",
        f"The audited source hashes match the prescribed values. The no-resume replay reproduced {baseline['observed']['pooled_moderate_failures']} moderate and {baseline['observed']['pooled_severe_failures']} severe Monte Carlo failures, FST {baseline['observed']['baseline_FST_moderate_crossings']}/{baseline['observed']['baseline_FST_severe_crossings']}, MI-FST {baseline['observed']['mechanism_balanced_FST_moderate_crossings']}/{baseline['observed']['mechanism_balanced_FST_severe_crossings']}, and RST {baseline['observed']['RST_passive_pathways']}/{baseline['observed']['RST_severe_pathways']} with {baseline['observed']['RST_structural_pathways']}/{baseline['observed']['RST_adaptation_limited_pathways']} structural/adaptation-limited pathways.",
        "",
        "## Claim support",
        "",
        "The calibration comparisons are fixed-library comparisons. Any positive RST-minus-MI coverage difference is conditional on the specified candidate library and model; an interval crossing zero is inconclusive rather than evidence of equivalence.",
        "",
        "The archived primary exact-signature contrast is checked against the required 122/2,455 = 4.9694501 percentage-point difference. The new variants, validation cohort, and topology-B results are reported separately and were not used to choose a variant.",
        "",
        "The bounded reference audit is finite-grid evidence only. It does not establish completeness over arbitrary three-event pathways, continuous magnitudes, arbitrary times, or the full disturbance space.",
        "",
        "Scaling results are engineering measurements for the implemented operator. Medians and IQRs use completed repetitions only; timeout counts and partial progress remain separate, and memory is reported as incremental process usage over the cell baseline.",
        "",
        "## Source hashes",
        "",
    ]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in source_hashes.items())
    lines.extend(
        [
            "",
            "## Output map",
            "",
            "- `candidate_manifest.csv` and `candidate_results.csv`: seven fixed 50-candidate libraries with full event and threshold-specific records.",
            "- `mc_evaluation.csv`, `coverage_summary.csv`, `paired_coverage_differences.csv`, and `signature_gain_loss.csv`: archived and independent validation cohorts, both topologies, and bootstrap intervals.",
            "- `reference_grid_manifest.csv`, `reference_grid_results.csv`, and `bounded_omission_summary.csv`: 3,699 topology-A and 5,166 topology-B reference pathways.",
            "- `scaling_runs.csv` and `scaling_summary.csv`: actual width, horizon, event-count, and budget workloads with counterfactual accounting.",
            "- `figure_*_data.csv` and `figures/*.svg`: underlying data and three principal figure groups.",
        ]
    )
    if not bounded.empty:
        lines.extend(["", "## Bounded omission result", "", f"The reference audit contains {len(bounded)} method/cohort/topology/threshold summary rows; denominators and numerators are stored in the CSV rather than represented only as percentages.", "", "| Topology | Method | Threshold | Signature recall | Event recall |", "|---|---|---|---:|---:|"])
        for row in bounded.to_dict("records"):
            lines.append(f"| {row['topology']} | {row['method']} | {row['threshold']} | {row['signature_coverage']:.6f} | {row['event_specification_recall']:.6f} |" )
    if not scaling.empty:
        completed = int((scaling["status"] == "completed").sum())
        lines.extend(["", "## Scaling result", "", f"{completed} timed workload repetitions completed; timeout and incomplete rows, if any, remain visible in `scaling_runs.csv`."])
    if not coverage.empty:
        lines.extend(["", "## Variant and interval results", "", "The coverage rows below are the observed variant results used by the comparison; bootstrap intervals are percentages on the 0-1 coverage scale.", "", "| Cohort | Topology | Threshold | Method | Numerator/denominator | Coverage | Bootstrap 95% interval |", "|---|---|---|---|---:|---:|---:|"])
        for row in coverage.to_dict("records"):
            lines.append(f"| {row['cohort']} | {row['topology']} | {row['threshold']} | {row['method']} | {int(row['numerator'])}/{int(row['denominator'])} | {row['coverage']:.6f} | [{row['bootstrap_lower_95']:.6f}, {row['bootstrap_upper_95']:.6f}] |" )
    if not differences.empty:
        paired = differences[differences["direction"] == "paired_delta_summary"]
        lines.extend(["", "## Paired comparisons", "", "| Cohort | Topology | Threshold | Comparator | RST numerator | Comparator numerator | Denominator | Observed delta pp | Per-seed mean pp | Per-seed SD pp | Bootstrap 95% interval pp |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in paired.to_dict("records"):
            lines.append(f"| {row['cohort']} | {row['topology']} | {row['threshold']} | {row['comparator']} | {int(row['rst_numerator'])} | {int(row['comparator_numerator'])} | {int(row['paired_denominator'])} | {row['observed_delta_pp']:.6f} | {row['per_seed_delta_mean_pp']:.6f} | {row['per_seed_delta_sd_pp']:.6f} | [{row['delta_pp_lower_95']:.6f}, {row['delta_pp_upper_95']:.6f}] |" )
    (output_dir / "revision_evidence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _validate_reused_intermediates(
    candidate_results: pd.DataFrame,
    mc: pd.DataFrame,
    libraries: Mapping[str, Sequence[model.Pathway]],
) -> None:
    candidate_required = {
        "method",
        "topology",
        "candidate_id",
        "threshold",
        "passive_failure",
        "event_specification",
        "canonical_event_key",
    }
    if not candidate_required.issubset(candidate_results.columns):
        raise AssertionError(f"Candidate intermediate missing columns: {sorted(candidate_required - set(candidate_results.columns))}")
    expected_candidate_rows = len(LIBRARY_ORDER) * 2 * 50 * len(THRESHOLDS)
    if len(candidate_results) != expected_candidate_rows:
        raise AssertionError(f"Candidate intermediate row count mismatch: {len(candidate_results)} != {expected_candidate_rows}")
    candidate_keys = ["method", "topology", "candidate_id", "threshold"]
    if candidate_results.duplicated(candidate_keys).any():
        raise AssertionError("Candidate intermediate contains duplicate method/topology/candidate/threshold rows")
    for method in LIBRARY_ORDER:
        expected_paths = {path.pathway_id: path for path in libraries[method]}
        for topology in ("A", "B"):
            rows = candidate_results[(candidate_results["method"] == method) & (candidate_results["topology"] == topology)]
            if set(rows["candidate_id"]) != set(expected_paths):
                raise AssertionError(f"Candidate intermediate IDs mismatch for {method}/{topology}")
            for candidate_id, group in rows.groupby("candidate_id"):
                expected_spec = _event_spec(expected_paths[candidate_id].events)
                if set(group["event_specification"]) != {expected_spec}:
                    raise AssertionError(f"Candidate event specification mismatch for {method}/{topology}/{candidate_id}")
                expected_key = _event_key(expected_paths[candidate_id].events)
                if set(group["canonical_event_key"]) != {expected_key}:
                    raise AssertionError(f"Candidate canonical event key mismatch for {method}/{topology}/{candidate_id}")

    mc_required = {
        "cohort",
        "topology",
        "seed",
        "replication",
        "threshold",
        "event_specification",
        "canonical_event_key",
        "passive_failure",
        "mechanism_signature",
    }
    if not mc_required.issubset(mc.columns):
        raise AssertionError(f"Monte Carlo intermediate missing columns: {sorted(mc_required - set(mc.columns))}")
    expected_mc_rows = 2 * 2 * len(ARCHIVED_SEEDS) * 1000 * len(THRESHOLDS)
    if len(mc) != expected_mc_rows:
        raise AssertionError(f"Monte Carlo intermediate row count mismatch: {len(mc)} != {expected_mc_rows}")
    mc_keys = ["cohort", "topology", "seed", "replication", "threshold"]
    if mc.duplicated(mc_keys).any():
        raise AssertionError("Monte Carlo intermediate contains duplicate cohort/topology/seed/replication/threshold rows")
    expected_seeds = {"archived": set(ARCHIVED_SEEDS), "validation": set(VALIDATION_SEEDS)}
    for (cohort, topology), group in mc.groupby(["cohort", "topology"]):
        if set(group["seed"].astype(int)) != expected_seeds[cohort]:
            raise AssertionError(f"Monte Carlo seed mismatch for {cohort}/{topology}")
        cfg = model.topology_a_config(**PRIMARY_CFG_KWARGS) if topology == "A" else model.topology_b_config(**PRIMARY_CFG_KWARGS)
        sample = group.drop_duplicates("replication")
        for _, row in sample.iterrows():
            events = _parse_event_spec(row["event_specification"])
            expected_key = _event_key(events, cfg)
            if row["canonical_event_key"] != expected_key:
                raise AssertionError(f"Monte Carlo canonical event key mismatch for {cohort}/{topology}/{row['seed']}/{row['replication']}")


def _rst22_validation(candidate_results: pd.DataFrame) -> dict[str, Any]:
    rows = candidate_results[
        (candidate_results["method"] == "RST")
        & (candidate_results["topology"] == "A")
        & (candidate_results["candidate_id"] == "RST-22")
        & (candidate_results["threshold"] == "moderate")
    ]
    if len(rows) != 1:
        return {"check": "RST-22 raw gamma", "passed": False, "evidence": json.dumps({"matching_rows": len(rows)})}
    row = rows.iloc[0]
    c0 = float(row["C0"])
    cadm = float(row["Cadm"])
    expected_raw = 1.0 - cadm / c0 if c0 else np.nan
    expected_clipped = float(np.clip(expected_raw, 0.0, 1.0)) if np.isfinite(expected_raw) else np.nan
    passed = bool(
        np.isfinite(expected_raw)
        and np.isclose(float(row["gamma_raw"]), expected_raw, atol=1e-12, rtol=1e-12)
        and np.isclose(float(row["gamma_clipped"]), expected_clipped, atol=1e-12, rtol=1e-12)
    )
    evidence = {
        "candidate_id": row["candidate_id"],
        "C0": c0,
        "Cadm": cadm,
        "gamma_raw": float(row["gamma_raw"]),
        "expected_gamma_raw": expected_raw,
        "gamma_clipped": float(row["gamma_clipped"]),
        "expected_gamma_clipped": expected_clipped,
    }
    return {"check": "RST-22 raw gamma", "passed": passed, "evidence": json.dumps(evidence)}


def _baseline_verification(output_dir: Path, root: Path) -> dict[str, Any]:
    expected = {
        "pooled_moderate_failures": 2455,
        "pooled_severe_failures": 1445,
        "baseline_FST_moderate_crossings": 33,
        "baseline_FST_severe_crossings": 11,
        "mechanism_balanced_FST_moderate_crossings": 46,
        "mechanism_balanced_FST_severe_crossings": 35,
        "RST_passive_pathways": 48,
        "RST_severe_pathways": 39,
        "RST_structural_pathways": 41,
        "RST_adaptation_limited_pathways": 7,
    }
    summary_path = root / "baseline_verification" / "analysis_summary.json"
    if not summary_path.exists():
        summary_path = root / "02_revised_analysis_outputs" / "analysis_summary.json"
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    source = source.get("primary", source)
    observed = {key: source.get(key) for key in expected}
    passed = observed == expected
    value = {"expected": expected, "observed": observed, "passed": passed, "source_summary": str(summary_path)}
    _write_json(value, output_dir / "baseline_verification.json")
    if not passed:
        raise AssertionError(f"Baseline verification failed: {observed}")
    return value


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_root) / args.run_id
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory exists; use --overwrite to replace it: {output_dir}")
    preexisting_worktree_changes = _git_value("status", "--short")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_hashes = {
        "generate_revised_analysis.py": _sha256_file(PROJECT_ROOT / "generate_revised_analysis.py"),
        "rst_revised_model.py": _sha256_file(PROJECT_ROOT / "rst_revised_model.py"),
        "generate_round2_analysis.py": _sha256_file(Path(__file__).resolve()),
    }
    prescribed = {
        "generate_revised_analysis.py": "28a62db0967dccdcda6eb39449da95a7fc934c3affe62e67db9ec8005db62ac2",
        "rst_revised_model.py": "22a277deae4cea8bcbb38b216a283472c661bec9490014136346ef3285e2ab75",
    }
    if any(source_hashes[key].lower() != value for key, value in prescribed.items()):
        raise AssertionError("Audited source hash mismatch")
    baseline = _baseline_verification(output_dir, PROJECT_ROOT)
    libraries, candidate_manifest = _build_candidate_libraries()
    _write_csv(candidate_manifest, output_dir / "candidate_manifest.csv")

    if args.reuse_intermediate:
        candidate_results = pd.read_csv(output_dir / "candidate_results.csv")
        mc = pd.read_csv(output_dir / "mc_evaluation.csv")
        for column in ("passive_failure", "active_failure", "moderate_passive_failure", "severe_passive_failure", "unclassified"):
            if column in mc.columns:
                mc[column] = mc[column].fillna(False).astype(bool)
        _validate_reused_intermediates(candidate_results, mc, libraries)
    else:
        all_candidate_results: list[pd.DataFrame] = []
        all_requests: list[pd.DataFrame] = []
        all_executions: list[pd.DataFrame] = []
        all_counterfactuals: list[pd.DataFrame] = []
        all_trajectories: list[pd.DataFrame] = []
        for topology, cfg in (("A", model.topology_a_config(**PRIMARY_CFG_KWARGS)), ("B", model.topology_b_config(**PRIMARY_CFG_KWARGS))):
            for method in LIBRARY_ORDER:
                results, requests, executions, counterfactuals, trajectories, evaluator = _evaluate_library(method, libraries[method], topology, cfg)
                all_candidate_results.append(results)
                requests["topology"] = topology
                executions["topology"] = topology
                all_requests.append(requests)
                all_executions.append(executions)
                counterfactuals["topology"] = topology
                all_counterfactuals.append(counterfactuals)
                all_trajectories.append(trajectories)
        candidate_results = pd.concat(all_candidate_results, ignore_index=True)
        manifest_frame = pd.DataFrame(candidate_manifest).copy()
        manifest_frame["method"] = manifest_frame["variant_id"]
        candidate_results = candidate_results.drop(columns=["variant_id", "parent_candidate_id"], errors="ignore").merge(
            manifest_frame[["method", "candidate_id", "variant_id", "parent_variant_id", "parent_candidate_id"]],
            on=["method", "candidate_id"],
            how="left",
        )
        _write_csv(candidate_results, output_dir / "candidate_results.csv")
        requests_df = pd.concat(all_requests, ignore_index=True)
        executions_df = pd.concat(all_executions, ignore_index=True)
        _write_csv(requests_df, output_dir / "simulation_requests.csv")
        _write_csv(executions_df, output_dir / "simulation_executions.csv")
        cost_rows = []
        for keys, group in requests_df.groupby(["topology", "method", "purpose"], sort=False):
            topology, method, purpose = keys
            executed = executions_df[
                (executions_df["topology"] == topology)
                & (executions_df["method"] == method)
                & (executions_df["purpose"] == purpose)
            ]
            cost_rows.append(
                {
                    "topology": topology,
                    "method": method,
                    "purpose": purpose,
                    "requested_calls": len(group),
                    "unique_simulations": len(executed),
                    "cache_hits": int(group["cache_hit"].sum()),
                    "elapsed_seconds": float(executed["elapsed_seconds"].sum()) if len(executed) else 0.0,
                }
            )
        _write_csv(cost_rows, output_dir / "simulation_costs.csv")
        _write_csv(pd.concat(all_counterfactuals, ignore_index=True), output_dir / "constituent_counterfactuals.csv")
        _write_csv(pd.concat(all_trajectories, ignore_index=True), output_dir / "candidate_trajectories.csv")
        _write_csv(pd.DataFrame(_common_scores(candidate_results[candidate_results["topology"] == "A"], libraries)), output_dir / "common_scoring_sensitivity.csv")

        mc_frames: list[pd.DataFrame] = []
        for topology, cfg in (("A", model.topology_a_config(**PRIMARY_CFG_KWARGS)), ("B", model.topology_b_config(**PRIMARY_CFG_KWARGS))):
            mc_frames.append(_mc_cohort(topology, "archived", ARCHIVED_SEEDS, 1000, cfg, PROJECT_ROOT))
            mc_frames.append(_mc_cohort(topology, "validation", VALIDATION_SEEDS, 1000, cfg, PROJECT_ROOT))
        mc = pd.concat(mc_frames, ignore_index=True)
        _write_csv(mc, output_dir / "mc_evaluation.csv")
        _validate_reused_intermediates(candidate_results, mc, libraries)
    if args.reuse_intermediate:
        _write_csv(pd.DataFrame(_common_scores(candidate_results[candidate_results["topology"] == "A"], libraries)), output_dir / "common_scoring_sensitivity.csv")
    result_for_coverage = {method: candidate_results[candidate_results["method"] == method] for method in LIBRARY_ORDER}
    coverage, differences, omission_seed = _coverage_rows(mc, result_for_coverage, args.bootstrap_reps)
    _write_csv(coverage, output_dir / "coverage_summary.csv")
    _write_csv(differences, output_dir / "paired_coverage_differences.csv")
    _write_csv(differences[differences["direction"].isin(["gained", "lost"])], output_dir / "signature_gain_loss.csv")
    _write_csv(omission_seed, output_dir / "coverage_per_seed.csv")

    omission_rows: list[dict[str, Any]] = []
    for (cohort, topology, threshold), cohort_group in mc.groupby(["cohort", "topology", "threshold"], sort=False):
        failure_group = cohort_group[cohort_group["passive_failure"]]
        total_failures = len(failure_group)
        for method in LIBRARY_ORDER:
            discovered = _signature_set(result_for_coverage[method][result_for_coverage[method]["topology"] == topology], threshold)
            for signature, signature_group in failure_group.groupby("mechanism_signature", sort=False):
                covered = bool(signature in discovered and signature != model.UNCLASSIFIED)
                frequency = len(signature_group)
                omission_rows.append(
                    {
                        "cohort": cohort,
                        "topology": topology,
                        "threshold": threshold,
                        "method": method,
                        "signature": signature,
                        "failure_frequency": frequency,
                        "total_failures": total_failures,
                        "covered": covered,
                        "uncovered_fraction": frequency / total_failures if total_failures and not covered else 0.0,
                    }
                )
    omission_catalog = pd.DataFrame(omission_rows)
    omission_figure_data = _omission_figure_data(omission_catalog)
    _write_csv(omission_catalog, output_dir / "omission_catalog.csv")
    _write_csv(differences[differences["direction"] == "paired_delta_summary"], output_dir / "figure_coverage_data.csv")
    _write_csv(omission_figure_data, output_dir / "figure_omission_data.csv")

    reference_manifest_rows: list[dict[str, Any]] = []
    reference_results: list[pd.DataFrame] = []
    bounded_rows: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    for topology, cfg in (("A", model.topology_a_config(**PRIMARY_CFG_KWARGS)), ("B", model.topology_b_config(**PRIMARY_CFG_KWARGS))):
        grid = _reference_grid(cfg, topology)
        for path in grid:
            reference_manifest_rows.append(
                {
                    "topology": topology,
                    "reference_id": path.pathway_id,
                    "reference_family": path.family,
                    "event_specification": _event_spec(path.events),
                    "canonical_event_key": _event_key(path.events, cfg),
                }
            )
        grid_results, _ = _evaluate_reference_grid(topology, grid, cfg)
        reference_results.append(grid_results)
        grid_keys = {_event_key(path.events, cfg) for path in grid}
        for method in LIBRARY_ORDER:
            library = libraries[method]
            library_keys = {_event_key(path.events, cfg) for path in library}
            for threshold in THRESHOLDS:
                failures = grid_results[(grid_results["threshold"] == threshold) & (grid_results["passive_failure"])]
                signatures = set(failures["mechanism_signature"]) - {model.UNCLASSIFIED}
                discovered = _signature_set(candidate_results[(candidate_results["method"] == method) & (candidate_results["topology"] == topology)], threshold)
                covered = failures["mechanism_signature"].isin(discovered) & failures["mechanism_signature"].ne(model.UNCLASSIFIED)
                exact_events = failures["canonical_event_key"].isin(library_keys)
                bounded_rows.append(
                    {
                        "topology": topology,
                        "method": method,
                        "threshold": threshold,
                        "signature_numerator": int(covered.sum()),
                        "signature_denominator": len(failures),
                        "signature_coverage": float(covered.mean()) if len(failures) else np.nan,
                        "reference_signature_count": len(signatures),
                        "event_specification_numerator": int(exact_events.sum()),
                        "event_specification_denominator": len(failures),
                        "event_specification_recall": float(exact_events.mean()) if len(failures) else np.nan,
                        "candidate_specs_outside_grid": len(library_keys - grid_keys),
                    }
                )
                for signature in sorted(signatures - discovered):
                    example = failures[failures["mechanism_signature"] == signature].iloc[0]
                    witnesses.append(
                        {
                            "topology": topology,
                            "method": method,
                            "threshold": threshold,
                            "missing_signature": signature,
                            "witness_reference_id": example["reference_id"],
                            "witness_event_specification": example["event_specification"],
                        }
                    )
    _write_csv(reference_manifest_rows, output_dir / "reference_grid_manifest.csv")
    _write_csv(pd.concat(reference_results, ignore_index=True), output_dir / "reference_grid_results.csv")
    _write_csv(bounded_rows, output_dir / "bounded_omission_summary.csv")
    _write_csv(witnesses, output_dir / "missing_signature_witnesses.csv")

    scaling_rows: list[dict[str, Any]] = []
    workload_path = output_dir / "scaling_workloads.jsonl"
    with workload_path.open("w", encoding="utf-8", newline="\n") as stream:
        for architecture in ("shared", "partitioned"):
            settings: set[tuple[int, int, int, int]] = set()
            for n in (3, 9, 30, 90, 300):
                settings.add((n, 52, 4, 50))
            for horizon in (52, 104, 208):
                settings.add((30, horizon, 4, 50))
            for K in (1, 2, 4, 8, 16):
                settings.add((30, 52, K, 50))
            for B in (50, 100, 200, 500):
                settings.add((30, 52, 4, B))
            for n, horizon, K, B in sorted(settings):
                cfg = _scaling_cfg(n, horizon, architecture)
                for b in range(1, B + 1):
                    path = _scaling_workload(cfg, b, B, K, architecture)
                    stream.write(json.dumps({"architecture": architecture, "n": n, "horizon": horizon, "K": K, "B": B, "candidate_id": path.pathway_id, "event_specification": _event_spec(path.events)}) + "\n")
                raw, _ = _run_scaling_cell(architecture, n, horizon, K, B)
                scaling_rows.extend(raw)
    scaling_df = pd.DataFrame(scaling_rows)
    _write_csv(scaling_df, output_dir / "scaling_runs.csv")
    summary = _summarize_scaling(scaling_df)
    scaling_figure_data = _scaling_figure_data(summary)
    _write_csv(summary, output_dir / "scaling_summary.csv")
    _write_csv(scaling_figure_data, output_dir / "figure_scaling_data.csv")

    threshold_sensitivity = _deterministic_threshold_sensitivity(libraries, model.topology_a_config(**PRIMARY_CFG_KWARGS))
    _write_csv(threshold_sensitivity, output_dir / "deterministic_threshold_sensitivity.csv")
    _write_csv(_existing_robustness_index(PROJECT_ROOT), output_dir / "existing_robustness_index.csv")
    validation_rows = [
        {"check": "source hashes", "passed": True, "evidence": json.dumps(source_hashes)},
        {"check": "baseline counts", "passed": bool(baseline["passed"]), "evidence": json.dumps(baseline["observed"])},
        {"check": "candidate library sizes", "passed": all(len(value) == 50 for value in libraries.values()), "evidence": json.dumps({key: len(value) for key, value in libraries.items()})},
        {"check": "candidate canonical uniqueness", "passed": all(len({_event_key(path.events) for path in value}) == 50 for value in libraries.values()), "evidence": "canonical event keys"},
        {"check": "reference grid sizes", "passed": len(reference_manifest_rows) == 3699 + 5166, "evidence": json.dumps(pd.DataFrame(reference_manifest_rows).groupby("topology").size().to_dict())},
        {"check": "bootstrap seed fixed", "passed": BOOTSTRAP_SEED == 20260924, "evidence": str(BOOTSTRAP_SEED)},
        _rst22_validation(candidate_results),
        {"check": "five attribution labels", "passed": set(model.MECHANISM_FAMILIES) == {model.CD, model.HCD, model.CMD, model.PAC, model.RL_TO}, "evidence": json.dumps(model.MECHANISM_FAMILIES)},
    ]
    primary_rows = coverage[
        (coverage["cohort"] == "archived")
        & (coverage["topology"] == "A")
        & (coverage["threshold"] == "moderate")
        & (coverage["method"].isin(["MI-FST", "RST"]))
    ]
    primary_numerators = dict(zip(primary_rows["method"], primary_rows["numerator"]))
    primary_denominators = dict(zip(primary_rows["method"], primary_rows["denominator"]))
    primary_margin = (
        100.0 * primary_numerators["RST"] / primary_denominators["RST"]
        - 100.0 * primary_numerators["MI-FST"] / primary_denominators["MI-FST"]
    )
    expected_margin = 100.0 * 122.0 / 2455.0
    validation_rows.append(
        {
            "check": "archived RST-minus-MI-FST primary margin",
            "passed": abs(primary_margin - expected_margin) < 1e-12,
            "evidence": json.dumps({"RST_numerator": primary_numerators["RST"], "MI_FST_numerator": primary_numerators["MI-FST"], "denominator": 2455, "observed_pp": primary_margin, "expected_pp": expected_margin}),
        }
    )
    _write_csv(validation_rows, output_dir / "validation_checks.csv")

    _make_figures(output_dir, differences, omission_figure_data, scaling_figure_data)
    _evidence_report(output_dir, baseline, coverage, differences, pd.DataFrame(bounded_rows), scaling_df, source_hashes)
    output_hashes = {path.relative_to(output_dir).as_posix(): _sha256_canonical_file(path) for path in output_dir.rglob("*") if path.is_file() and path.name != "run_manifest.json"}
    manifest = {
        "run_id": args.run_id,
        "protocol_version": PROTOCOL_VERSION,
        "audited_commit": "59348388860d98c9bc74a26f55300be41d623838",
        "current_commit": _git_value("rev-parse", "HEAD"),
        "output_hash_definition": "SHA-256 over canonical LF bytes",
        "source_hashes": source_hashes,
        "environment": _env_versions(),
        "scientific_configuration": {
            "primary_response": PRIMARY_CFG_KWARGS,
            "thresholds": {name: asdict(threshold) for name, threshold in THRESHOLDS.items()},
            "archived_seeds": ARCHIVED_SEEDS,
            "validation_seeds": VALIDATION_SEEDS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": args.bootstrap_reps,
        },
        "preexisting_worktree_changes": preexisting_worktree_changes,
        "invocation": " ".join(sys.argv),
        "elapsed_seconds": time.perf_counter() - started,
        "output_hashes": output_hashes,
    }
    _write_json(manifest, output_dir / "run_manifest.json")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="03_round2_revision_outputs")
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-intermediate", action="store_true")
    args = parser.parse_args()
    output_dir = run(args)
    print(json.dumps({"output_dir": str(output_dir), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
