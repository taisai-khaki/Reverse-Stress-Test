#!/usr/bin/env python3
"""Revised RST computational model aligned with manuscript Sections 3 and 4.1-4.5."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import nbinom


CD = "Concentrated dependency"
HCD = "Hidden convergent dependency"
CMD = "Compounding moderate disturbance"
PAC = "Propagation-amplified cascade"
RL_TO = "Recovery lag and temporal overlap"
MECHANISM_FAMILIES = (CD, HCD, CMD, PAC, RL_TO)
PRIMARY_MECHANISM_PRIORITY = (HCD, CMD, PAC, RL_TO, CD)

STRUCTURAL = "Structural"
ADAPTATION_LIMITED = "Adaptation-limited"
NON_FAILURE = "Non-failure"
UNCLASSIFIED = "Unclassified"


@dataclass(frozen=True)
class FailureThreshold:
    name: str
    alpha: float
    consecutive: int


MODERATE = FailureThreshold("SL<0.90 for >=2 periods", 0.90, 2)
SEVERE = FailureThreshold("SL<0.85 for >=3 periods", 0.85, 3)


@dataclass(frozen=True)
class NetworkConfig:
    horizon_weeks: int = 52
    demand_per_week: float = 900.0
    tier1_nodes: Tuple[str, str, str] = ("S1A", "S1B", "S1C")
    tier1_allocation_shares: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    tier1_capacity: float = 400.0
    tier1_initial_inventory: float = 200.0
    tier1_order_up_to: float = 1100.0
    t2p_capacity: float = 1000.0
    t2p_to_tier1_lead_weeks: int = 2
    upstream_nodes: Tuple[str, ...] = ("T2P",)
    tier1_upstream_nodes: Tuple[str, ...] = ("T2P", "T2P", "T2P")
    upstream_capacities: Optional[Tuple[float, ...]] = None
    corridor_capacity: float = 1200.0
    corridor_lead_weeks: int = 1
    fp_initial_inventory: float = 0.0
    protected_reserve_capacity: float = 900.0
    active_backlog_trigger_ratio: float = 0.20
    active_alt_corridor_units: float = 200.0
    active_reserve_release_frac: float = 0.30
    severity_weight: float = 0.625
    plausibility_weight: float = 0.375
    mc_interarrival_mean_weeks: float = 18.0
    dependence_strength: float = 0.65

    def validate(self) -> None:
        if len(self.tier1_nodes) != len(self.tier1_allocation_shares):
            raise ValueError("tier1_allocation_shares must align with tier1_nodes")
        if len(self.tier1_nodes) != len(self.tier1_upstream_nodes):
            raise ValueError("tier1_upstream_nodes must align with tier1_nodes")
        if not math.isclose(sum(self.tier1_allocation_shares), 1.0, abs_tol=1e-12):
            raise ValueError("tier1 allocation shares must sum to one")
        if self.upstream_capacities is not None and len(self.upstream_capacities) != len(self.upstream_nodes):
            raise ValueError("upstream_capacities must align with upstream_nodes")
        if not 0.0 <= self.active_reserve_release_frac <= 1.0:
            raise ValueError("active_reserve_release_frac must be in [0, 1]")


@dataclass(frozen=True)
class Event:
    target: str
    magnitude: float
    start_week: int
    duration_weeks: int
    recovery_weeks: int


@dataclass(frozen=True)
class Pathway:
    pathway_id: str
    family: str
    events: Tuple[Event, ...]
    note: str = ""


@dataclass
class SimulationResult:
    service_levels: List[float]
    demand_series: List[float]
    backlog_series: List[float]
    fp_inventory_series: List[float]
    reserve_series: List[float]
    reserve_release_series: List[float]
    rerouted_series: List[float]
    corridor_capacity_series: List[float]
    corridor_throughput_series: List[float]
    blocked_flow_series: List[float]
    tier1_inventory_series: Dict[str, List[float]]
    trace: List[Dict[str, float]]
    crossed_moderate: bool
    crossed_severe: bool
    first_cross_week_moderate: Optional[int]
    first_cross_week_severe: Optional[int]
    worst_service_level: float


@dataclass(frozen=True)
class MechanismAttribution:
    labels: Tuple[str, ...]
    threshold_crossed: bool
    first_qualified_cross_week: Optional[int]
    causal_prefix_indices: Tuple[int, ...]
    single_event_crossings: Tuple[int, ...]
    hcd_removal_elements: Tuple[str, ...]
    direct_impact_crossing: Optional[bool]
    rlto_support_pairs: Tuple[Tuple[int, int], ...]


@dataclass
class PathwayEval:
    pathway_id: str
    family: str
    mechanism_labels: Tuple[str, ...]
    primary_mechanism: str
    passive_failure: bool
    passive_severe_failure: bool
    active_failure: Optional[bool]
    classification: str
    severity: float
    active_shortfall: float
    plausibility: float
    controllability: float
    weight: float
    elements: Tuple[str, ...]
    note: str = ""


def topology_a_config(**overrides: object) -> NetworkConfig:
    return NetworkConfig(**overrides)


def topology_b_config(**overrides: object) -> NetworkConfig:
    values: Dict[str, object] = {
        "upstream_nodes": ("T2P-A", "T2P-B"),
        "tier1_upstream_nodes": ("T2P-A", "T2P-A", "T2P-B"),
    }
    values.update(overrides)
    return NetworkConfig(**values)


def threshold_cross_info(
    service_levels: Sequence[float], threshold: FailureThreshold
) -> Tuple[bool, Optional[int]]:
    """Return the first period at which the duration-qualified condition is satisfied."""
    streak = 0
    for week, service_level in enumerate(service_levels):
        if service_level < threshold.alpha:
            streak += 1
            if streak >= threshold.consecutive:
                return True, week
        else:
            streak = 0
    return False, None


def _event_profile_for_week(event: Event, week: int) -> float:
    disruption_end = event.start_week + event.duration_weeks
    recovery_end = disruption_end + event.recovery_weeks
    if week < event.start_week:
        return 0.0
    if week < disruption_end:
        return event.magnitude
    if event.recovery_weeks > 0 and week < recovery_end:
        recovery_fraction = (week - disruption_end) / float(event.recovery_weeks)
        return event.magnitude * (1.0 - recovery_fraction)
    return 0.0


def _tier1_upstream_map(cfg: NetworkConfig) -> Dict[str, str]:
    cfg.validate()
    return dict(zip(cfg.tier1_nodes, cfg.tier1_upstream_nodes))


def _upstream_capacity_map(cfg: NetworkConfig) -> Dict[str, float]:
    cfg.validate()
    if cfg.upstream_capacities is not None:
        return dict(zip(cfg.upstream_nodes, cfg.upstream_capacities))
    mapping = _tier1_upstream_map(cfg)
    channel_counts = {
        upstream: sum(mapping[node] == upstream for node in cfg.tier1_nodes)
        for upstream in cfg.upstream_nodes
    }
    return {
        upstream: cfg.t2p_capacity * channel_counts[upstream] / len(cfg.tier1_nodes)
        for upstream in cfg.upstream_nodes
    }


def _profile_target(target: str, cfg: NetworkConfig) -> str:
    if target in cfg.upstream_nodes:
        return target
    if target == "T2P":
        return cfg.upstream_nodes[0]
    if target.startswith("T2P-") and "T2P" in cfg.upstream_nodes:
        return "T2P"
    return target


def _canonical_element(target: str) -> str:
    return "T2P" if target == "T2P" or target.startswith("T2P-") else target


def _build_profiles(events: Sequence[Event], cfg: NetworkConfig) -> Dict[str, np.ndarray]:
    targets = [*cfg.upstream_nodes, *cfg.tier1_nodes, "corridor", "demand"]
    profiles = {target: np.zeros(cfg.horizon_weeks, dtype=float) for target in targets}
    for week in range(cfg.horizon_weeks):
        for event in events:
            target = _profile_target(event.target, cfg)
            if target not in profiles:
                raise ValueError(f"Unknown event target {event.target!r} for this topology")
            magnitude = _event_profile_for_week(event, week)
            if target == "demand":
                profiles[target][week] = (1.0 + profiles[target][week]) * (1.0 + magnitude) - 1.0
            else:
                profiles[target][week] = 1.0 - (1.0 - profiles[target][week]) * (1.0 - magnitude)
                profiles[target][week] = float(np.clip(profiles[target][week], 0.0, 1.0))
    return profiles


def run_simulation(
    events: Sequence[Event],
    cfg: NetworkConfig,
    active_mode: bool = False,
    direct_impact_mode: bool = False,
) -> SimulationResult:
    """Run the revised 52-period system operator from manuscript Equations (1)-(49)."""
    cfg.validate()
    profiles = _build_profiles(events, cfg)
    upstream_for_tier1 = _tier1_upstream_map(cfg)
    upstream_capacity = _upstream_capacity_map(cfg)
    shares = dict(zip(cfg.tier1_nodes, cfg.tier1_allocation_shares))
    nominal_slot = {
        node: shares[node] * cfg.demand_per_week for node in cfg.tier1_nodes
    }

    tier1_inventory = {node: cfg.tier1_initial_inventory for node in cfg.tier1_nodes}
    tier1_pipeline = {
        node: [nominal_slot[node]] * (cfg.t2p_to_tier1_lead_weeks + 1)
        for node in cfg.tier1_nodes
    }
    corridor_pipeline = [cfg.demand_per_week] * cfg.corridor_lead_weeks
    fp_inventory = cfg.fp_initial_inventory
    reserve = cfg.protected_reserve_capacity
    backlog = 0.0

    service_levels: List[float] = []
    demands: List[float] = []
    backlogs: List[float] = []
    fp_inventories: List[float] = []
    reserves: List[float] = [reserve]
    reserve_releases: List[float] = []
    rerouted: List[float] = []
    corridor_capacities: List[float] = []
    corridor_throughputs: List[float] = []
    blocked_flows: List[float] = []
    tier1_inventories = {node: [] for node in cfg.tier1_nodes}
    trace: List[Dict[str, float]] = []

    epsilon = 1e-12
    for week in range(cfg.horizon_weeks):
        if direct_impact_mode:
            tier1_inventory = {node: cfg.tier1_initial_inventory for node in cfg.tier1_nodes}
            tier1_pipeline = {
                node: [nominal_slot[node]] * (cfg.t2p_to_tier1_lead_weeks + 1)
                for node in cfg.tier1_nodes
            }
            corridor_pipeline = [cfg.demand_per_week] * cfg.corridor_lead_weeks
            fp_inventory = cfg.fp_initial_inventory
            reserve = cfg.protected_reserve_capacity
            backlog = 0.0

        demand = cfg.demand_per_week * (1.0 + profiles["demand"][week])
        total_obligation = demand + backlog
        requested_shipments = {
            node: shares[node] * total_obligation for node in cfg.tier1_nodes
        }

        prior_pipeline = {node: list(tier1_pipeline[node]) for node in cfg.tier1_nodes}
        tier1_available = {
            node: tier1_inventory[node] + prior_pipeline[node][0]
            for node in cfg.tier1_nodes
        }
        effective_tier1_capacity = {
            node: cfg.tier1_capacity * (1.0 - profiles[node][week])
            for node in cfg.tier1_nodes
        }
        gross_shipments = {
            node: min(
                tier1_available[node],
                effective_tier1_capacity[node],
                requested_shipments[node],
            )
            for node in cfg.tier1_nodes
        }

        corridor_capacity = cfg.corridor_capacity * (1.0 - profiles["corridor"][week])
        gross_total = sum(gross_shipments.values())
        corridor_factor = min(1.0, corridor_capacity / max(gross_total, epsilon))
        normal_shipments = {
            node: corridor_factor * gross_shipments[node] for node in cfg.tier1_nodes
        }
        normal_throughput = sum(normal_shipments.values())
        blocked = {
            node: max(0.0, gross_shipments[node] - normal_shipments[node])
            for node in cfg.tier1_nodes
        }
        blocked_total = sum(blocked.values())

        corridor_pipeline_before = sum(corridor_pipeline)
        corridor_receipt = corridor_pipeline.pop(0)
        corridor_pipeline.append(0.0)

        inherited_backlog = backlog
        pre_response_shortfall = max(
            0.0, total_obligation - (fp_inventory + corridor_receipt)
        )
        activation = active_mode and inherited_backlog >= (
            cfg.active_backlog_trigger_ratio * cfg.demand_per_week
        )
        reroute_total = 0.0
        if activation:
            reroute_total = min(
                cfg.active_alt_corridor_units,
                blocked_total,
                pre_response_shortfall,
            )
        reroute_by_node = {
            node: (
                reroute_total * blocked[node] / blocked_total
                if blocked_total > epsilon
                else 0.0
            )
            for node in cfg.tier1_nodes
        }

        reserve_floor = (1.0 - cfg.active_reserve_release_frac) * cfg.protected_reserve_capacity
        reserve_release = 0.0
        if activation:
            reserve_release = min(
                max(0.0, reserve - reserve_floor),
                max(0.0, pre_response_shortfall - reroute_total),
            )
        reserve = max(reserve_floor, reserve - reserve_release)

        allocable_supply = fp_inventory + corridor_receipt + reroute_total + reserve_release
        filled = min(allocable_supply, total_obligation)
        fp_inventory = max(0.0, allocable_supply - filled)
        backlog = max(0.0, total_obligation - filled)
        service_level = min(1.0, filled / max(demand, epsilon))

        ending_tier1_inventory: Dict[str, float] = {}
        orders: Dict[str, float] = {}
        for node in cfg.tier1_nodes:
            ending_tier1_inventory[node] = max(
                0.0,
                tier1_available[node] - normal_shipments[node] - reroute_by_node[node],
            )
            future_pipeline = sum(prior_pipeline[node][1:])
            orders[node] = max(
                0.0,
                cfg.tier1_order_up_to - ending_tier1_inventory[node] - future_pipeline,
            )

        effective_upstream_capacity = {
            upstream: upstream_capacity[upstream] * (1.0 - profiles[upstream][week])
            for upstream in cfg.upstream_nodes
        }
        dispatch = {node: 0.0 for node in cfg.tier1_nodes}
        for upstream in cfg.upstream_nodes:
            connected_nodes = [
                node for node in cfg.tier1_nodes if upstream_for_tier1[node] == upstream
            ]
            order_total = sum(orders[node] for node in connected_nodes)
            dispatch_total = min(effective_upstream_capacity[upstream], order_total)
            if order_total > epsilon:
                for node in connected_nodes:
                    dispatch[node] = dispatch_total * orders[node] / order_total

        max_pipeline_error = 0.0
        max_inventory_error = 0.0
        for node in cfg.tier1_nodes:
            new_pipeline = prior_pipeline[node][1:] + [dispatch[node]]
            arriving = prior_pipeline[node][0]
            pipeline_error = (
                sum(prior_pipeline[node]) + dispatch[node] - arriving - sum(new_pipeline)
            )
            inventory_error = (
                tier1_available[node]
                - normal_shipments[node]
                - reroute_by_node[node]
                - ending_tier1_inventory[node]
            )
            max_pipeline_error = max(max_pipeline_error, abs(pipeline_error))
            max_inventory_error = max(max_inventory_error, abs(inventory_error))
            tier1_pipeline[node] = new_pipeline
            tier1_inventory[node] = ending_tier1_inventory[node]

        corridor_pipeline[-1] += normal_throughput
        corridor_pipeline_error = (
            corridor_pipeline_before
            + normal_throughput
            - corridor_receipt
            - sum(corridor_pipeline)
        )

        service_levels.append(float(service_level))
        demands.append(float(demand))
        backlogs.append(float(backlog))
        fp_inventories.append(float(fp_inventory))
        reserves.append(float(reserve))
        reserve_releases.append(float(reserve_release))
        rerouted.append(float(reroute_total))
        corridor_capacities.append(float(corridor_capacity))
        corridor_throughputs.append(float(normal_throughput))
        blocked_flows.append(float(blocked_total))
        for node in cfg.tier1_nodes:
            tier1_inventories[node].append(float(tier1_inventory[node]))

        trace.append(
            {
                "week": float(week),
                "demand": float(demand),
                "inherited_backlog": float(inherited_backlog),
                "ending_backlog": float(backlog),
                "fp_inventory": float(fp_inventory),
                "reserve": float(reserve),
                "reserve_release": float(reserve_release),
                "corridor_capacity": float(corridor_capacity),
                "gross_shipments": float(gross_total),
                "corridor_throughput": float(normal_throughput),
                "blocked_flow": float(blocked_total),
                "rerouted_flow": float(reroute_total),
                "pre_response_shortfall": float(pre_response_shortfall),
                "max_pipeline_conservation_error": float(max_pipeline_error),
                "corridor_pipeline_conservation_error": float(abs(corridor_pipeline_error)),
                "max_tier1_inventory_balance_error": float(max_inventory_error),
                "min_tier1_inventory": float(min(tier1_inventory.values())),
                "min_tier1_pipeline": float(
                    min(min(values) for values in tier1_pipeline.values())
                ),
            }
        )

    crossed_moderate, first_moderate = threshold_cross_info(service_levels, MODERATE)
    crossed_severe, first_severe = threshold_cross_info(service_levels, SEVERE)
    return SimulationResult(
        service_levels=service_levels,
        demand_series=demands,
        backlog_series=backlogs,
        fp_inventory_series=fp_inventories,
        reserve_series=reserves,
        reserve_release_series=reserve_releases,
        rerouted_series=rerouted,
        corridor_capacity_series=corridor_capacities,
        corridor_throughput_series=corridor_throughputs,
        blocked_flow_series=blocked_flows,
        tier1_inventory_series=tier1_inventories,
        trace=trace,
        crossed_moderate=crossed_moderate,
        crossed_severe=crossed_severe,
        first_cross_week_moderate=first_moderate,
        first_cross_week_severe=first_severe,
        worst_service_level=float(min(service_levels) if service_levels else 1.0),
    )


def _capacity_event(
    target: str,
    magnitude: float,
    start: int,
    duration: int,
    recovery: int = 3,
) -> Event:
    return Event(target, magnitude, start, duration, recovery)


def _demand_event(magnitude: float, start: int, duration: int, recovery: int = 2) -> Event:
    return Event("demand", magnitude, start, duration, recovery)


def fst_library() -> List[Pathway]:
    out: List[Pathway] = []
    start = 12
    index = 1

    def add(category: str, events: Sequence[Event]) -> None:
        nonlocal index
        out.append(Pathway(f"FST-{index:02d}", category, tuple(events)))
        index += 1

    for supplier in ("S1A", "S1B", "S1C"):
        for magnitude in (0.30, 0.50, 0.70):
            for duration in (2, 4, 6):
                add("Tier-1 supplier failure", [_capacity_event(supplier, magnitude, start, duration)])
    for magnitude in (0.20, 0.35, 0.50, 0.65):
        for duration in (3, 6):
            add("Shared-processor failure", [_capacity_event("T2P", magnitude, start, duration)])
    for magnitude in (0.30, 0.50, 0.70):
        for duration in (2, 4):
            add("Corridor failure", [_capacity_event("corridor", magnitude, start, duration)])
    for magnitude in (0.10, 0.20, 0.30, 0.40):
        add("Demand spike", [_demand_event(magnitude, start, 3)])

    combinations = [
        [_capacity_event("T2P", 0.30, start, 3), _capacity_event("corridor", 0.30, start, 3)],
        [_capacity_event("T2P", 0.40, start, 3), _capacity_event("S1A", 0.40, start, 3)],
        [_capacity_event("T2P", 0.40, start, 3), _demand_event(0.20, start, 3)],
        [_capacity_event("corridor", 0.40, start, 3), _demand_event(0.20, start, 3)],
        [_capacity_event("S1A", 0.40, start, 3), _capacity_event("S1B", 0.40, start, 3)],
    ]
    for events in combinations:
        add("Prespecified combination", events)
    if len(out) != 50:
        raise AssertionError(f"Baseline FST must contain 50 scenarios, found {len(out)}")
    return out


def fst_enhanced_library() -> List[Pathway]:
    out: List[Pathway] = []
    start = 12
    index = 1

    def add(category: str, events: Sequence[Event]) -> None:
        nonlocal index
        out.append(Pathway(f"FST-MB-{index:02d}", category, tuple(events)))
        index += 1

    for supplier in ("S1A", "S1B", "S1C"):
        for magnitude in (0.50, 0.70):
            add("Tier-1 supplier failure", [_capacity_event(supplier, magnitude, start, 4)])
    for magnitude in (0.20, 0.35, 0.50, 0.65, 0.80):
        for duration in (3, 6):
            add("Shared-processor failure", [_capacity_event("T2P", magnitude, start, duration)])
    for magnitude in (0.30, 0.50, 0.70):
        for duration in (2, 4):
            add("Corridor failure", [_capacity_event("corridor", magnitude, start, duration)])
    for t2p_magnitude in (0.30, 0.50, 0.70, 0.80):
        for corridor_magnitude in (0.30, 0.60):
            add(
                "Simultaneous T2P-corridor",
                [
                    _capacity_event("T2P", t2p_magnitude, start, 4),
                    _capacity_event("corridor", corridor_magnitude, start, 4),
                ],
            )
    for supplier in ("S1A", "S1B", "S1C"):
        for t2p_magnitude in (0.40, 0.60):
            add(
                "Simultaneous T2P-tier-1",
                [
                    _capacity_event("T2P", t2p_magnitude, start, 4),
                    _capacity_event(supplier, 0.50, start, 4),
                ],
            )
    for t2p_magnitude in (0.40, 0.60):
        for demand_magnitude in (0.20, 0.40):
            add(
                "Demand-T2P combination",
                [
                    _capacity_event("T2P", t2p_magnitude, start, 4),
                    _demand_event(demand_magnitude, start, 4),
                ],
            )
    for first_magnitude in (0.30, 0.40, 0.50, 0.60, 0.70):
        add(
            "Sequential T2P-corridor",
            [
                _capacity_event("T2P", first_magnitude, start, 3, 3),
                _capacity_event("corridor", 0.50, start + 3, 2, 3),
            ],
        )
    for first_magnitude in (0.30, 0.40, 0.50, 0.60, 0.70):
        add(
            "Sequential corridor-T2P",
            [
                _capacity_event("corridor", first_magnitude, start, 3, 3),
                _capacity_event("T2P", 0.50, start + 3, 2, 3),
            ],
        )
    if len(out) != 50:
        raise AssertionError(f"Mechanism-balanced FST must contain 50 scenarios, found {len(out)}")
    return out


def rst_candidate_library() -> List[Pathway]:
    out: List[Pathway] = []
    start = 12
    index = 1

    def add(family: str, events: Sequence[Event]) -> None:
        nonlocal index
        out.append(Pathway(f"RST-{index:02d}", family, tuple(events)))
        index += 1

    for target in ("T2P", "corridor"):
        for magnitude, duration in ((0.45, 2), (0.55, 3), (0.65, 4), (0.75, 5)):
            add(CD, [_capacity_event(target, magnitude, start, duration)])
    for magnitude in (0.20, 0.30, 0.40, 0.50, 0.60):
        for duration in (4, 7):
            add(HCD, [_capacity_event("T2P", magnitude, start, duration)])
    for t2p_magnitude in (0.25, 0.35):
        for corridor_magnitude in (0.25, 0.35):
            add(
                CMD,
                [
                    _capacity_event("T2P", t2p_magnitude, start, 3),
                    _capacity_event("corridor", corridor_magnitude, start, 3),
                ],
            )
    for t2p_magnitude in (0.25, 0.35):
        for demand_magnitude in (0.15, 0.25):
            add(
                CMD,
                [
                    _capacity_event("T2P", t2p_magnitude, start, 3),
                    _demand_event(demand_magnitude, start, 3),
                ],
            )
    for supplier in ("S1A", "S1B", "S1C"):
        add(
            CMD,
            [
                _capacity_event("T2P", 0.30, start, 3),
                _capacity_event(supplier, 0.40, start, 3),
            ],
        )
    add(CMD, [_capacity_event("corridor", 0.30, start, 3), _demand_event(0.20, start, 3)])
    for magnitude in (0.25, 0.35, 0.45, 0.55, 0.65):
        add(PAC, [_capacity_event("T2P", magnitude, start, 6, 4)])
    for magnitude in (0.25, 0.35, 0.45, 0.55, 0.65):
        add(
            PAC,
            [
                _capacity_event("T2P", magnitude, start, 5, 4),
                _demand_event(0.15, start + 2, 4, 2),
            ],
        )
    for first_magnitude in (0.30, 0.40, 0.50, 0.60, 0.70):
        add(
            RL_TO,
            [
                _capacity_event("T2P", first_magnitude, start, 3, 4),
                _capacity_event("corridor", 0.50, start + 4, 2, 3),
            ],
        )
    for first_magnitude in (0.30, 0.40, 0.50, 0.60, 0.70):
        add(
            RL_TO,
            [
                _capacity_event("corridor", first_magnitude, start, 3, 4),
                _capacity_event("T2P", 0.50, start + 4, 2, 3),
            ],
        )
    if len(out) != 50:
        raise AssertionError(f"RST must contain 50 candidates, found {len(out)}")
    return out


def pathway_elements(pathway: Pathway) -> Tuple[str, ...]:
    structural = {
        _canonical_element(event.target)
        for event in pathway.events
        if event.target != "demand"
    }
    return tuple(sorted(structural))


def pathway_severity(
    service_levels: Sequence[float], threshold: FailureThreshold = MODERATE
) -> float:
    return float(sum(max(0.0, threshold.alpha - value) for value in service_levels))


_BETA_MODE = (2.0 - 1.0) / (2.0 + 5.0 - 2.0)
_NB_MEAN_Y = 2.2
_NB_VARIANCE_Y = 8.1
_NB_P = _NB_MEAN_Y / _NB_VARIANCE_Y
_NB_R = _NB_MEAN_Y * _NB_P / (1.0 - _NB_P)
_NB_MAX_PMF = max(float(nbinom.pmf(value, _NB_R, _NB_P)) for value in range(200))


def _normalized_beta_likelihood(value: float, low: float, high: float) -> float:
    if value < low or value > high:
        return 0.0
    scaled = (value - low) / (high - low)
    density = float(beta_dist.pdf(scaled, 2.0, 5.0) / (high - low))
    mode_value = low + (high - low) * _BETA_MODE
    mode_scaled = (mode_value - low) / (high - low)
    maximum = float(beta_dist.pdf(mode_scaled, 2.0, 5.0) / (high - low))
    return density / maximum if maximum > 0 else 0.0


def pathway_plausibility(pathway: Pathway) -> float:
    if not pathway.events:
        return 1.0
    likelihoods: List[float] = []
    for event in pathway.events:
        if event.target == "demand":
            likelihoods.append(_normalized_beta_likelihood(event.magnitude, 0.05, 0.40))
        else:
            likelihoods.append(_normalized_beta_likelihood(event.magnitude, 0.15, 0.80))
        duration_mass = float(
            nbinom.pmf(max(0, event.duration_weeks - 1), _NB_R, _NB_P)
        )
        likelihoods.append(duration_mass / _NB_MAX_PMF if _NB_MAX_PMF > 0 else 0.0)
        likelihoods.append(1.0 if event.recovery_weeks in (1, 2, 3, 4) else 0.0)
    return float(
        math.exp(
            sum(math.log(max(value, 1e-12)) for value in likelihoods)
            / len(likelihoods)
        )
    )


def _channels_for_element(element: str, cfg: NetworkConfig) -> Tuple[str, ...]:
    if element == "corridor":
        return tuple(cfg.tier1_nodes)
    if element in cfg.upstream_nodes:
        mapping = _tier1_upstream_map(cfg)
        return tuple(node for node in cfg.tier1_nodes if mapping[node] == element)
    if element in cfg.tier1_nodes:
        return (element,)
    return ()


def _recovery_overlap(first: Event, second: Event) -> bool:
    recovery_start = first.start_week + first.duration_weeks
    recovery_end = recovery_start + first.recovery_weeks
    return recovery_start <= second.start_week < recovery_end


def attribute_mechanisms(
    events: Sequence[Event],
    cfg: NetworkConfig,
    threshold: FailureThreshold = MODERATE,
    passive_result: Optional[SimulationResult] = None,
    label_order: Sequence[str] = MECHANISM_FAMILIES,
) -> MechanismAttribution:
    if set(label_order) != set(MECHANISM_FAMILIES) or len(label_order) != len(MECHANISM_FAMILIES):
        raise ValueError("label_order must contain each mechanism family exactly once")
    passive = passive_result or run_simulation(events, cfg, active_mode=False)
    crossed, first_qualified = threshold_cross_info(passive.service_levels, threshold)
    if not crossed or first_qualified is None:
        return MechanismAttribution((), False, None, (), (), (), None, ())

    prefix_indices = tuple(
        index for index, event in enumerate(events) if event.start_week <= first_qualified
    )
    prefix_events = tuple(events[index] for index in prefix_indices)
    simulation_cache: Dict[Tuple[Tuple[int, ...], bool], bool] = {}

    def subset_crosses(indices: Iterable[int], direct: bool = False) -> bool:
        index_tuple = tuple(sorted(indices))
        key = (index_tuple, direct)
        if key not in simulation_cache:
            subset = [events[index] for index in index_tuple]
            result = run_simulation(
                subset,
                cfg,
                active_mode=False,
                direct_impact_mode=direct,
            )
            simulation_cache[key] = threshold_cross_info(result.service_levels, threshold)[0]
        return simulation_cache[key]

    single_crossings = tuple(
        index for index in prefix_indices if subset_crosses((index,))
    )

    affected_elements = {
        _profile_target(events[index].target, cfg)
        for index in prefix_indices
        if _profile_target(events[index].target, cfg) in cfg.upstream_nodes
    }
    hcd_removal_elements: List[str] = []
    for element in sorted(affected_elements):
        if len(_channels_for_element(element, cfg)) < 2:
            continue
        remaining = tuple(
            index
            for index in prefix_indices
            if _profile_target(events[index].target, cfg) != element
        )
        if not subset_crosses(remaining):
            hcd_removal_elements.append(element)

    has_upstream = any(
        _profile_target(events[index].target, cfg) in cfg.upstream_nodes
        for index in prefix_indices
    )
    direct_impact_crossing: Optional[bool] = None
    if has_upstream:
        direct_impact_crossing = subset_crosses(prefix_indices, direct=True)

    rlto_pairs: List[Tuple[int, int]] = []
    for first_index in prefix_indices:
        for second_index in prefix_indices:
            if first_index == second_index:
                continue
            first = events[first_index]
            second = events[second_index]
            if first.start_week > second.start_week:
                continue
            if not _recovery_overlap(first, second):
                continue
            remaining = tuple(index for index in prefix_indices if index != second_index)
            if not subset_crosses(remaining):
                rlto_pairs.append((first_index, second_index))

    detector_results: Mapping[str, bool] = {
        CD: bool(single_crossings),
        HCD: bool(hcd_removal_elements),
        CMD: len(prefix_indices) >= 2 and not single_crossings,
        PAC: has_upstream and direct_impact_crossing is False,
        RL_TO: bool(rlto_pairs),
    }
    labels = {family for family in label_order if detector_results[family]}
    ordered_labels = tuple(family for family in MECHANISM_FAMILIES if family in labels)
    return MechanismAttribution(
        labels=ordered_labels,
        threshold_crossed=True,
        first_qualified_cross_week=first_qualified,
        causal_prefix_indices=prefix_indices,
        single_event_crossings=single_crossings,
        hcd_removal_elements=tuple(hcd_removal_elements),
        direct_impact_crossing=direct_impact_crossing,
        rlto_support_pairs=tuple(rlto_pairs),
    )


def primary_mechanism(labels: Sequence[str]) -> str:
    label_set = set(labels)
    for family in PRIMARY_MECHANISM_PRIORITY:
        if family in label_set:
            return family
    return UNCLASSIFIED


def classify_mechanism(
    events: Sequence[Event],
    cfg: NetworkConfig,
    passive_result: SimulationResult,
    active_result: Optional[SimulationResult] = None,
    threshold: FailureThreshold = MODERATE,
) -> str:
    attribution = attribute_mechanisms(
        events, cfg, threshold=threshold, passive_result=passive_result
    )
    if not attribution.threshold_crossed:
        return NON_FAILURE
    return primary_mechanism(attribution.labels)


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum, abs_tol=1e-15):
        return [0.0] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def compute_criticality(evaluations: Sequence[PathwayEval]) -> Dict[str, float]:
    elements = ("T2P", "corridor", "S1A", "S1B", "S1C")
    scores = {element: 0.0 for element in elements}
    eligible = [evaluation for evaluation in evaluations if evaluation.passive_failure and evaluation.elements]
    denominator = sum(evaluation.weight for evaluation in eligible)
    if denominator <= 0:
        return scores
    for evaluation in eligible:
        contribution = evaluation.weight / len(evaluation.elements)
        for element in evaluation.elements:
            if element in scores:
                scores[element] += contribution
    return {element: score / denominator for element, score in scores.items()}


def mechanism_guided_rst_search(
    candidates: Sequence[Pathway],
    cfg: NetworkConfig,
    budget: int = 50,
    threshold: FailureThreshold = MODERATE,
    severity_weight: Optional[float] = None,
    plausibility_weight: Optional[float] = None,
) -> Tuple[List[PathwayEval], Dict[str, float]]:
    severity_weight = cfg.severity_weight if severity_weight is None else severity_weight
    plausibility_weight = cfg.plausibility_weight if plausibility_weight is None else plausibility_weight
    if severity_weight < 0 or plausibility_weight < 0:
        raise ValueError("pathway weights must be nonnegative")
    if not math.isclose(severity_weight + plausibility_weight, 1.0, abs_tol=1e-12):
        raise ValueError("severity and plausibility weights must sum to one")

    evaluations: List[PathwayEval] = []
    for pathway in list(candidates)[:budget]:
        passive = run_simulation(pathway.events, cfg, active_mode=False)
        passive_failure = threshold_cross_info(passive.service_levels, threshold)[0]
        passive_severe_failure = threshold_cross_info(passive.service_levels, SEVERE)[0]
        if passive_failure:
            active = run_simulation(pathway.events, cfg, active_mode=True)
            active_failure = threshold_cross_info(active.service_levels, threshold)[0]
            attribution = attribute_mechanisms(
                pathway.events,
                cfg,
                threshold=threshold,
                passive_result=passive,
            )
            passive_shortfall = pathway_severity(passive.service_levels, threshold)
            active_shortfall = pathway_severity(active.service_levels, threshold)
            controllability = (
                1.0 - active_shortfall / passive_shortfall
                if passive_shortfall > 0
                else 0.0
            )
            classification = STRUCTURAL if active_failure else ADAPTATION_LIMITED
        else:
            active_failure = None
            attribution = MechanismAttribution((), False, None, (), (), (), None, ())
            passive_shortfall = 0.0
            active_shortfall = 0.0
            controllability = 0.0
            classification = NON_FAILURE
        evaluations.append(
            PathwayEval(
                pathway_id=pathway.pathway_id,
                family=pathway.family,
                mechanism_labels=attribution.labels,
                primary_mechanism=primary_mechanism(attribution.labels),
                passive_failure=passive_failure,
                passive_severe_failure=passive_severe_failure,
                active_failure=active_failure,
                classification=classification,
                severity=passive_shortfall,
                active_shortfall=active_shortfall,
                plausibility=pathway_plausibility(pathway),
                controllability=float(np.clip(controllability, 0.0, 1.0)),
                weight=0.0,
                elements=pathway_elements(pathway),
                note=pathway.note,
            )
        )

    failure_indices = [
        index for index, evaluation in enumerate(evaluations) if evaluation.passive_failure
    ]
    severity_normalized = _normalize(
        [evaluations[index].severity for index in failure_indices]
    )
    plausibility_normalized = _normalize(
        [evaluations[index].plausibility for index in failure_indices]
    )
    raw_weights = [
        severity_weight * severity_normalized[position]
        + plausibility_weight * plausibility_normalized[position]
        for position in range(len(failure_indices))
    ]
    if failure_indices and math.isclose(sum(raw_weights), 0.0, abs_tol=1e-15):
        raw_weights = [1.0] * len(failure_indices)
    for position, index in enumerate(failure_indices):
        evaluations[index].weight = raw_weights[position]
    return evaluations, compute_criticality(evaluations)


def _labels_to_text(labels: Sequence[str]) -> str:
    return "; ".join(labels) if labels else UNCLASSIFIED


def run_fst(scenarios: Sequence[Pathway], cfg: NetworkConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        simulation = run_simulation(scenario.events, cfg, active_mode=False)
        moderate_crossing = threshold_cross_info(simulation.service_levels, MODERATE)[0]
        severe_crossing = threshold_cross_info(simulation.service_levels, SEVERE)[0]
        moderate_attribution = (
            attribute_mechanisms(
                scenario.events,
                cfg,
                threshold=MODERATE,
                passive_result=simulation,
            )
            if moderate_crossing
            else MechanismAttribution((), False, None, (), (), (), None, ())
        )
        severe_attribution = (
            attribute_mechanisms(
                scenario.events,
                cfg,
                threshold=SEVERE,
                passive_result=simulation,
            )
            if severe_crossing
            else MechanismAttribution((), False, None, (), (), (), None, ())
        )
        rows.append(
            {
                "scenario_id": scenario.pathway_id,
                "category": scenario.family,
                "moderate_crossing": moderate_crossing,
                "severe_crossing": severe_crossing,
                "mechanism_labels_moderate": _labels_to_text(moderate_attribution.labels),
                "primary_mechanism_moderate": primary_mechanism(moderate_attribution.labels),
                "mechanism_labels_severe": _labels_to_text(severe_attribution.labels),
                "primary_mechanism_severe": primary_mechanism(severe_attribution.labels),
                "elements": "; ".join(pathway_elements(scenario)),
                "event_specification": json.dumps(
                    [asdict(event) for event in scenario.events], separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(rows)


def sample_duration_negative_binomial(rng: np.random.Generator) -> int:
    return 1 + int(rng.negative_binomial(n=_NB_R, p=_NB_P))


def _sample_event_class(
    rng: np.random.Generator,
    dependence_model: str,
    previous_class: Optional[str],
) -> str:
    classes = ("upstream", "tier1", "corridor", "demand")
    if dependence_model == "Independent baseline" or previous_class is None:
        return str(rng.choice(classes))
    strength = 0.65
    if dependence_model == "Shared upstream shock" and previous_class == "upstream":
        return "upstream" if rng.random() < strength else str(rng.choice(classes))
    if dependence_model == "Supplier-cluster shock" and previous_class == "tier1":
        return "tier1" if rng.random() < strength else str(rng.choice(classes))
    if dependence_model == "Demand-supply coupling":
        if previous_class == "demand" and rng.random() < strength:
            return str(rng.choice(("upstream", "tier1", "corridor")))
        if previous_class != "demand" and rng.random() < strength:
            return "demand"
    return str(rng.choice(classes))


def generate_random_events(
    rng: np.random.Generator,
    cfg: NetworkConfig,
    dependence_model: str = "Independent baseline",
) -> List[Event]:
    supported = {
        "Independent baseline",
        "Shared upstream shock",
        "Supplier-cluster shock",
        "Demand-supply coupling",
    }
    if dependence_model not in supported:
        raise ValueError(f"Unsupported dependence model: {dependence_model}")
    events: List[Event] = []
    start_week = max(1, int(math.ceil(rng.exponential(cfg.mc_interarrival_mean_weeks))))
    previous_class: Optional[str] = None
    previous_tier1: Optional[str] = None
    upstream_capacity = _upstream_capacity_map(cfg)
    upstream_probabilities = np.array(
        [upstream_capacity[node] for node in cfg.upstream_nodes], dtype=float
    )
    upstream_probabilities /= upstream_probabilities.sum()

    while start_week < cfg.horizon_weeks:
        event_class = _sample_event_class(
            rng, dependence_model, previous_class
        )
        if event_class == "upstream":
            target = str(rng.choice(cfg.upstream_nodes, p=upstream_probabilities))
        elif event_class == "tier1":
            choices = list(cfg.tier1_nodes)
            if dependence_model == "Supplier-cluster shock" and previous_tier1 in choices:
                choices = [node for node in choices if node != previous_tier1]
            target = str(rng.choice(choices))
            previous_tier1 = target
        else:
            target = event_class
        if target == "demand":
            magnitude = 0.05 + 0.35 * float(rng.beta(2.0, 5.0))
        else:
            magnitude = 0.15 + 0.65 * float(rng.beta(2.0, 5.0))
        events.append(
            Event(
                target=target,
                magnitude=magnitude,
                start_week=start_week,
                duration_weeks=sample_duration_negative_binomial(rng),
                recovery_weeks=int(rng.integers(1, 5)),
            )
        )
        previous_class = event_class
        start_week += max(
            1, int(math.ceil(rng.exponential(cfg.mc_interarrival_mean_weeks)))
        )
    return events


def simulate_monte_carlo(
    replications: int,
    cfg: NetworkConfig,
    seed: int,
    dependence_model: str = "Independent baseline",
    attribute_failures: bool = True,
) -> pd.DataFrame:
    root = np.random.SeedSequence(seed)
    child_seeds = root.spawn(replications)
    rows: List[Dict[str, object]] = []
    for replication, child_seed in enumerate(child_seeds, start=1):
        rng = np.random.default_rng(child_seed)
        events = generate_random_events(rng, cfg, dependence_model)
        passive = run_simulation(events, cfg, active_mode=False)
        moderate_failure = threshold_cross_info(passive.service_levels, MODERATE)[0]
        severe_failure = threshold_cross_info(passive.service_levels, SEVERE)[0]
        active_failure: Optional[bool] = None
        classification = NON_FAILURE
        labels: Tuple[str, ...] = ()
        if moderate_failure:
            active = run_simulation(events, cfg, active_mode=True)
            active_failure = threshold_cross_info(active.service_levels, MODERATE)[0]
            classification = STRUCTURAL if active_failure else ADAPTATION_LIMITED
            if attribute_failures:
                labels = attribute_mechanisms(
                    events,
                    cfg,
                    threshold=MODERATE,
                    passive_result=passive,
                ).labels
        targets = {_canonical_element(event.target) for event in events}
        rows.append(
            {
                "seed": seed,
                "replication": replication,
                "dependence_model": dependence_model,
                "moderate_passive_failure": moderate_failure,
                "moderate_active_failure": active_failure,
                "severe_passive_failure": severe_failure,
                "passive_active_classification": classification,
                "mechanism_labels": _labels_to_text(labels),
                "primary_mechanism": primary_mechanism(labels),
                "event_count": len(events),
                "worst_service_level": passive.worst_service_level,
                "max_backlog_passive": max(passive.backlog_series, default=0.0),
                "has_T2P": "T2P" in targets,
                "has_corridor": "corridor" in targets,
                "has_S1A": "S1A" in targets,
                "has_S1B": "S1B" in targets,
                "has_S1C": "S1C" in targets,
                "event_specification": json.dumps(
                    [asdict(event) for event in events], separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_ground_truth(
    replications: int,
    cfg: NetworkConfig,
    seed: int = 42,
) -> pd.DataFrame:
    return simulate_monte_carlo(replications, cfg, seed)


def pathway_evaluations_dataframe(
    evaluations: Sequence[PathwayEval], pathways: Optional[Sequence[Pathway]] = None
) -> pd.DataFrame:
    pathway_map = {pathway.pathway_id: pathway for pathway in pathways or []}
    rows: List[Dict[str, object]] = []
    for evaluation in evaluations:
        pathway = pathway_map.get(evaluation.pathway_id)
        rows.append(
            {
                "pathway_id": evaluation.pathway_id,
                "generation_family": evaluation.family,
                "mechanism_labels": _labels_to_text(evaluation.mechanism_labels),
                "primary_mechanism": evaluation.primary_mechanism,
                "passive_failure": evaluation.passive_failure,
                "passive_severe_failure": evaluation.passive_severe_failure,
                "active_failure": evaluation.active_failure,
                "classification": evaluation.classification,
                "passive_cumulative_shortfall": evaluation.severity,
                "active_cumulative_shortfall": evaluation.active_shortfall,
                "plausibility": evaluation.plausibility,
                "controllability": evaluation.controllability,
                "structural_weight": evaluation.weight,
                "elements": "; ".join(evaluation.elements),
                "event_specification": (
                    json.dumps(
                        [asdict(event) for event in pathway.events], separators=(",", ":")
                    )
                    if pathway is not None
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)
