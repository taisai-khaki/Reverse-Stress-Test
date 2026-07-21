#!/usr/bin/env python3
"""
Reverse Stress Testing (RST) reference implementation aligned to:

"Reverse Stress Testing for Supply Chain Resilience:
A Failure-First Analytical Framework for Identifying Non-Viability Pathways"

This script provides:
1) A weekly discrete-event simulation of a stylized 3-tier supply chain.
2) Monte Carlo generation of failure "ground truth" events.
3) Forward Stress Testing (FST) baseline with a 50-scenario library.
4) Mechanism-guided RST search with passive/active evaluation.
5) Failure-conditioned criticality scores.

Notes:
- This is a transparent research implementation from paper text, not official author code.
- Some operational choices are calibrated from the textual description where exact equations
  were not fully specified in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import expon, nbinom


# ---------------------------
# Data classes
# ---------------------------


@dataclass(frozen=True)
class NetworkConfig:
    horizon_weeks: int = 52
    demand_per_week: float = 900.0
    fp_initial_inventory: float = 900.0

    t2p_capacity: float = 1000.0
    t2p_to_tier1_lead_weeks: int = 2
    upstream_nodes: Tuple[str, ...] = ("T2P",)
    tier1_upstream_nodes: Tuple[str, ...] = ("T2P", "T2P", "T2P")

    tier1_nodes: Tuple[str, str, str] = ("S1A", "S1B", "S1C")
    tier1_capacity: float = 400.0
    tier1_initial_inventory: float = 200.0
    tier1_base_stock_target: float = 1175.0
    tier1_to_fp_lead_weeks: int = 1

    corridor_capacity: float = 1200.0

    # Class structure for allocation equations
    critical_class_share: float = 0.50

    # Active-response parameters (Eq. 25-29 style)
    active_backlog_trigger_ratio: float = 0.17  # calibrated near paper passive/active split
    active_corridor_trigger: float = 0.40       # rerouting only if corridor loss >= 40%
    active_alt_corridor_units: float = 200.0    # kappa_alt
    active_reserve_release_frac: float = 0.30   # rho_rel
    active_lambda_backlog: float = 10.0
    active_lambda_cost: float = 1.0

    # Monte Carlo pathway generation controls
    mc_interarrival_mean_weeks: float = 18.0
    mc_max_events: int = 2
    mc_start_week_max: int = 5
    mc_target_probs: Tuple[float, float, float, float, float, float] = (
        0.28,  # T2P
        0.18,  # corridor
        0.14,  # S1A
        0.14,  # S1B
        0.14,  # S1C
        0.12,  # demand
    )


@dataclass(frozen=True)
class FailureThreshold:
    name: str
    alpha: float          # service-level boundary (SL_t < alpha)
    consecutive: int      # required consecutive periods below alpha


@dataclass(frozen=True)
class Event:
    target: str           # T2P | corridor | S1A | S1B | S1C | demand
    magnitude: float      # fractional loss for capacities; fractional increase for demand
    start_week: int       # 0-indexed
    duration_weeks: int
    recovery_weeks: int = 0


@dataclass(frozen=True)
class Pathway:
    pathway_id: str
    family: str
    events: Tuple[Event, ...]
    note: str = ""


@dataclass
class SimulationResult:
    service_levels: List[float]
    backlog_series: List[float]
    fp_inventory_series: List[float]
    tier1_inventory_series: Dict[str, List[float]]
    crossed_moderate: bool
    crossed_severe: bool
    first_cross_week_moderate: Optional[int]
    first_cross_week_severe: Optional[int]
    worst_service_level: float


@dataclass
class PathwayEval:
    pathway_id: str
    family: str
    passive_failure: bool
    active_failure: Optional[bool]
    classification: str
    severity: float
    plausibility: float
    controllability: float
    weight: float
    elements: Tuple[str, ...]
    note: str = ""


# ---------------------------
# Thresholds (paper Section 5.3)
# ---------------------------


MODERATE = FailureThreshold("moderate", alpha=0.90, consecutive=2)
SEVERE = FailureThreshold("severe", alpha=0.85, consecutive=3)


# ---------------------------
# Utilities
# ---------------------------


def truncated_beta(
    rng: np.random.Generator,
    a: float = 2.0,
    b: float = 5.0,
    low: float = 0.15,
    high: float = 0.80,
    size: Optional[int] = None,
) -> np.ndarray:
    x = rng.beta(a, b, size=size)
    return low + (high - low) * x


def sample_duration_negative_binomial(rng: np.random.Generator) -> int:
    # Paper: mean 3.2, variance 8.1
    mean = 3.2
    var = 8.1
    p = mean / var
    n = mean * p / (1.0 - p)
    d = int(rng.negative_binomial(n=n, p=p))
    return max(1, d)


def threshold_cross_info(service_levels: Sequence[float], threshold: FailureThreshold) -> Tuple[bool, Optional[int]]:
    streak = 0
    for t, sl in enumerate(service_levels):
        if sl < threshold.alpha:
            streak += 1
            if streak >= threshold.consecutive:
                return True, t - threshold.consecutive + 1
        else:
            streak = 0
    return False, None


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _scaled_beta_pdf(x: float, a: float = 2.0, b: float = 5.0, low: float = 0.15, high: float = 0.80) -> float:
    if x < low or x > high:
        return 1e-12
    z = (x - low) / (high - low)
    return float(beta_dist.pdf(z, a, b) / (high - low))


def _tier1_upstream_map(cfg: NetworkConfig) -> Dict[str, str]:
    if len(cfg.tier1_upstream_nodes) != len(cfg.tier1_nodes):
        raise ValueError("tier1_upstream_nodes must align with tier1_nodes")
    return dict(zip(cfg.tier1_nodes, cfg.tier1_upstream_nodes))


def _upstream_capacity_map(cfg: NetworkConfig) -> Dict[str, float]:
    upstream_for_tier1 = _tier1_upstream_map(cfg)
    counts = {upstream: 0 for upstream in cfg.upstream_nodes}
    for tier1_node in cfg.tier1_nodes:
        upstream = upstream_for_tier1[tier1_node]
        if upstream not in counts:
            raise ValueError(f"Unknown upstream node {upstream!r} for {tier1_node}")
        counts[upstream] += 1
    total_tier1_nodes = float(len(cfg.tier1_nodes))
    return {
        upstream: cfg.t2p_capacity * (count / total_tier1_nodes)
        for upstream, count in counts.items()
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
    if target == "T2P" or target.startswith("T2P-"):
        return "T2P"
    return target


# ---------------------------
# Simulation core (weekly discrete-event)
# ---------------------------


def _event_profile_for_week(event: Event, week: int) -> float:
    """
    Returns disruption intensity in [0, 1] for the event at this week.
    - During active disruption: constant magnitude.
    - During recovery: linear decay to 0.
    """
    start = event.start_week
    end = start + event.duration_weeks
    if week < start:
        return 0.0
    if start <= week < end:
        return event.magnitude
    if event.recovery_weeks <= 0:
        return 0.0
    rec_end = end + event.recovery_weeks
    if end <= week < rec_end:
        k = (week - end + 1) / float(event.recovery_weeks)
        return max(0.0, event.magnitude * (1.0 - k))
    return 0.0


def _build_profiles(events: Sequence[Event], cfg: NetworkConfig) -> Dict[str, np.ndarray]:
    horizon = cfg.horizon_weeks
    profiles: Dict[str, np.ndarray] = {
        "corridor": np.zeros(horizon),
        "S1A": np.zeros(horizon),
        "S1B": np.zeros(horizon),
        "S1C": np.zeros(horizon),
        "demand": np.zeros(horizon),
    }
    for upstream in cfg.upstream_nodes:
        profiles[upstream] = np.zeros(horizon)
    for week in range(horizon):
        for e in events:
            target = _profile_target(e.target, cfg)
            mag = _event_profile_for_week(e, week)
            if target == "demand":
                # Demand shocks accumulate additively.
                profiles["demand"][week] += mag
            else:
                # Capacity losses combine multiplicatively on residual capacity:
                # residual *= (1 - mag) across events.
                # Convert to combined loss at end.
                profiles[target][week] = 1.0 - (1.0 - profiles[target][week]) * (1.0 - mag)
    return profiles


def run_simulation(
    events: Sequence[Event],
    cfg: NetworkConfig,
    active_mode: bool = False,
    lean_tier1_inventory: bool = False,
    preload_backlog: float = 0.0,
) -> SimulationResult:
    horizon = cfg.horizon_weeks
    profiles = _build_profiles(events, cfg)
    upstream_for_tier1 = _tier1_upstream_map(cfg)
    upstream_capacity = _upstream_capacity_map(cfg)

    class_demand_share = {
        1: cfg.critical_class_share,
        2: 1.0 - cfg.critical_class_share,
    }

    tier1_inv = {
        n: (0.5 * cfg.tier1_initial_inventory if lean_tier1_inventory else cfg.tier1_initial_inventory)
        for n in cfg.tier1_nodes
    }
    fp_inv = cfg.fp_initial_inventory
    backlog_by_class = {
        1: max(0.0, preload_backlog * class_demand_share[1]),
        2: max(0.0, preload_backlog * class_demand_share[2]),
    }

    # Eq. (21)-style steady-state initialization: tau in {0,1,2} for lead time 2.
    # tau=0 denotes immediate-arrival slot; tau=1,2 are in-transit slots.
    lead_t2p = cfg.t2p_to_tier1_lead_weeks
    steady_slot = cfg.t2p_capacity / len(cfg.tier1_nodes)
    t2p_pipeline: Dict[str, List[float]] = {
        n: [steady_slot] * (lead_t2p + 1)
        for n in cfg.tier1_nodes
    }
    # Eq. R_t = S_corr^{t-1}, initialized at steady-state corridor flow.
    fp_pipeline: List[float] = [cfg.demand_per_week] * cfg.tier1_to_fp_lead_weeks

    service_levels: List[float] = []
    backlogs: List[float] = []
    fp_invs: List[float] = []
    tier1_invs: Dict[str, List[float]] = {n: [] for n in cfg.tier1_nodes}

    for t in range(horizon):
        # 1) Tier-1 pipeline arrivals and shift (Eq. 5 semantics)
        prior_pipeline: Dict[str, List[float]] = {}
        for n in cfg.tier1_nodes:
            old = list(t2p_pipeline[n])
            prior_pipeline[n] = old
            arriving = old[0]
            tier1_inv[n] += arriving

        # 2) FP receives prior-period corridor throughput
        fp_arrival = fp_pipeline.pop(0)
        fp_inv += fp_arrival
        fp_pipeline.append(0.0)

        # 3) Effective capacities
        t2p_cap = {
            upstream: upstream_capacity[upstream] * (1.0 - profiles[upstream][t])
            for upstream in cfg.upstream_nodes
        }
        corridor_cap = cfg.corridor_capacity * (1.0 - profiles["corridor"][t])
        s1_cap = {
            n: cfg.tier1_capacity * (1.0 - profiles[n][t]) for n in cfg.tier1_nodes
        }

        # 4) Demand realization and obligations
        demand_multiplier = 1.0 + profiles["demand"][t]
        class_demand = {
            1: cfg.demand_per_week * class_demand_share[1] * demand_multiplier,
            2: cfg.demand_per_week * class_demand_share[2] * demand_multiplier,
        }
        obligations = {
            k: class_demand[k] + backlog_by_class[k] for k in class_demand
        }
        total_obligation = sum(obligations.values())

        # 5) Base-stock replenishment orders O_i^t (Eq. 3)
        safety_target = cfg.tier1_base_stock_target * (0.5 if lean_tier1_inventory else 1.0)
        orders = {}
        for n in cfg.tier1_nodes:
            in_transit_future = sum(prior_pipeline[n][1:])
            orders[n] = max(0.0, safety_target - tier1_inv[n] - in_transit_future)
        dispatch = {n: 0.0 for n in cfg.tier1_nodes}
        for upstream in cfg.upstream_nodes:
            upstream_tier1_nodes = [
                node for node in cfg.tier1_nodes if upstream_for_tier1[node] == upstream
            ]
            order_total = sum(orders[node] for node in upstream_tier1_nodes)
            t2p_output = min(t2p_cap[upstream], order_total)
            if order_total > 1e-9:
                for node in upstream_tier1_nodes:
                    dispatch[node] = t2p_output * (orders[node] / order_total)

        # Complete Eq. (5) transition into t+1 pipeline state.
        for n in cfg.tier1_nodes:
            new_pipe = [0.0] * (lead_t2p + 1)
            for tau in range(lead_t2p):
                new_pipe[tau] = prior_pipeline[n][tau + 1]
            new_pipe[lead_t2p] = dispatch[n]
            t2p_pipeline[n] = new_pipe

        # 6) Tier-1 shipments and corridor factor phi^t (Eq. 6-9)
        ship_candidate = {}
        for n in cfg.tier1_nodes:
            ship_candidate[n] = min(tier1_inv[n], s1_cap[n])

        total_candidate = sum(ship_candidate.values())
        phi = min(1.0, corridor_cap / max(total_candidate, 1e-9))
        shipped_adj = {n: phi * ship_candidate[n] for n in cfg.tier1_nodes}

        shipped_total = 0.0
        for n in cfg.tier1_nodes:
            ship = shipped_adj[n]
            tier1_inv[n] -= ship
            shipped_total += ship

        fp_pipeline[-1] += shipped_total

        # 7) Fill/allocation logic (passive Eq. 18/20, active Eq. 25-29 style)
        A_t = fp_inv
        backlog_prev = sum(backlog_by_class.values())
        tau_B = cfg.active_backlog_trigger_ratio * cfg.demand_per_week
        A_eff = A_t

        if active_mode and backlog_prev >= tau_B:
            delta_r_cap = (
                cfg.active_alt_corridor_units
                if profiles["corridor"][t] >= cfg.active_corridor_trigger
                else 0.0
            )
            delta_i_cap = cfg.active_reserve_release_frac * cfg.fp_initial_inventory

            # Greedy solution of Eq. (29) under lambda_B > lambda_C
            shortfall = max(0.0, total_obligation - A_t)
            delta_r = 0.0
            delta_i = 0.0
            if cfg.active_lambda_backlog > cfg.active_lambda_cost:
                delta_r = min(delta_r_cap, shortfall)
                shortfall -= delta_r
                delta_i = min(delta_i_cap, shortfall)

            A_eff = A_t + delta_r + delta_i

            filled_by_class = {1: 0.0, 2: 0.0}
            filled_by_class[1] = min(A_eff, obligations[1])
            rem = max(0.0, A_eff - filled_by_class[1])
            denom_other = obligations[2]
            filled_by_class[2] = (obligations[2] / max(denom_other, 1e-9)) * rem if denom_other > 0 else 0.0
        else:
            filled_total_passive = min(A_eff, total_obligation)
            filled_by_class = {}
            if total_obligation > 1e-9:
                for k in obligations:
                    filled_by_class[k] = (obligations[k] / total_obligation) * filled_total_passive
            else:
                for k in obligations:
                    filled_by_class[k] = 0.0

        for k in backlog_by_class:
            backlog_by_class[k] = max(0.0, obligations[k] - filled_by_class[k])

        filled = float(sum(filled_by_class.values()))
        fp_inv = max(0.0, A_eff - filled)
        backlog = float(sum(backlog_by_class.values()))

        # paper uses baseline 900 in denominator
        sl = min(1.0, filled / cfg.demand_per_week)
        service_levels.append(float(sl))
        backlogs.append(float(backlog))
        fp_invs.append(float(fp_inv))
        for n in cfg.tier1_nodes:
            tier1_invs[n].append(float(tier1_inv[n]))

    crossed_moderate, wk_m = threshold_cross_info(service_levels, MODERATE)
    crossed_severe, wk_s = threshold_cross_info(service_levels, SEVERE)

    return SimulationResult(
        service_levels=service_levels,
        backlog_series=backlogs,
        fp_inventory_series=fp_invs,
        tier1_inventory_series=tier1_invs,
        crossed_moderate=crossed_moderate,
        crossed_severe=crossed_severe,
        first_cross_week_moderate=wk_m,
        first_cross_week_severe=wk_s,
        worst_service_level=float(min(service_levels) if service_levels else 1.0),
    )


# ---------------------------
# Scenario/Pathway libraries
# ---------------------------


def _single_event_pathway(
    pid: str, family: str, target: str, mag: float, duration: int, note: str = "", start: int = 10
) -> Pathway:
    return Pathway(pid, family, (Event(target=target, magnitude=mag, start_week=start, duration_weeks=duration),), note)


def fst_library() -> List[Pathway]:
    out: List[Pathway] = []
    start = 10

    # FST-01..27 single tier-1 failures
    mags = [0.25, 0.50, 0.75]
    durs = [2, 4, 6]
    idx = 1
    for s1 in ("S1A", "S1B", "S1C"):
        for m in mags:
            for d in durs:
                out.append(_single_event_pathway(f"FST-{idx:02d}", "Single tier-1 failure", s1, m, d, start=start))
                idx += 1

    # FST-28..35 T2P partial failures
    for m, d in [(0.30, 3), (0.40, 3), (0.50, 3), (0.60, 3), (0.60, 4), (0.70, 3), (0.70, 4), (0.80, 4)]:
        out.append(_single_event_pathway(f"FST-{idx:02d}", "T2P partial failure", "T2P", m, d, start=start))
        idx += 1

    # FST-36..41 corridor disruptions
    for m, d in [(0.20, 2), (0.40, 2), (0.60, 2), (0.80, 2), (1.00, 1), (0.60, 4)]:
        out.append(_single_event_pathway(f"FST-{idx:02d}", "Corridor disruption", "corridor", m, d, start=start))
        idx += 1

    # FST-42..45 demand spikes
    for m, d in [(0.30, 2), (0.50, 2), (0.30, 4), (0.50, 4)]:
        out.append(_single_event_pathway(f"FST-{idx:02d}", "Demand spike", "demand", m, d, start=start))
        idx += 1

    # FST-46..50 combinations
    combo_specs = [
        ("S1A + S1B simultaneous failure", [Event("S1A", 1.00, start, 2), Event("S1B", 1.00, start, 2)]),
        ("S1A + S1C simultaneous failure", [Event("S1A", 1.00, start, 2), Event("S1C", 1.00, start, 2)]),
        ("S1B + S1C simultaneous failure", [Event("S1B", 1.00, start, 2), Event("S1C", 1.00, start, 2)]),
        ("T2P + corridor concurrent", [Event("T2P", 0.60, start, 3), Event("corridor", 0.40, start, 3)]),
        ("S1A + S1B high-severity", [Event("S1A", 0.75, start, 3), Event("S1B", 0.75, start, 3)]),
    ]
    for family, events in combo_specs:
        out.append(Pathway(pathway_id=f"FST-{idx:02d}", family=family, events=tuple(events)))
        idx += 1

    return out


def fst_enhanced_library() -> List[Pathway]:
    """Mechanism-balanced 50-scenario FST baseline.

    The enhanced baseline keeps the same 50-scenario budget as the original
    FST library while shifting scenarios away from redundant tier-1-only
    cases and toward convergence, compounding, propagation, and temporal
    overlap structures raised by Reviewer 2.
    """
    out: List[Pathway] = []
    start = 10
    idx = 1

    def add(category: str, events: Sequence[Event]) -> None:
        nonlocal idx
        out.append(Pathway(pathway_id=f"FST-E-{idx:02d}", family=category, events=tuple(events)))
        idx += 1

    # 10 T2P partial failures
    for magnitude, duration in [
        (0.20, 3),
        (0.25, 4),
        (0.30, 3),
        (0.35, 5),
        (0.40, 4),
        (0.50, 3),
        (0.60, 3),
        (0.60, 4),
        (0.70, 4),
        (0.80, 4),
    ]:
        add("T2P partial failure", [Event("T2P", magnitude, start, duration)])

    # 6 corridor disruptions
    for magnitude, duration in [
        (0.40, 2),
        (0.60, 2),
        (0.80, 2),
        (1.00, 1),
        (0.60, 4),
        (0.80, 4),
    ]:
        add("Corridor disruption", [Event("corridor", magnitude, start, duration)])

    # 6 tier-1 supplier failures
    for supplier, magnitude, duration in [
        ("S1A", 0.75, 4),
        ("S1B", 0.75, 4),
        ("S1C", 0.75, 4),
        ("S1A", 1.00, 4),
        ("S1B", 1.00, 4),
        ("S1C", 1.00, 4),
    ]:
        add("Tier-1 supplier failure", [Event(supplier, magnitude, start, duration)])

    # 8 T2P + corridor combinations
    for t2p_magnitude, corridor_magnitude, duration in [
        (0.20, 0.20, 3),
        (0.25, 0.25, 3),
        (0.30, 0.30, 3),
        (0.35, 0.30, 4),
        (0.40, 0.40, 3),
        (0.50, 0.30, 4),
        (0.60, 0.40, 3),
        (0.60, 0.50, 4),
    ]:
        add(
            "T2P + corridor combination",
            [
                Event("T2P", t2p_magnitude, start, duration),
                Event("corridor", corridor_magnitude, start, duration),
            ],
        )

    # 6 T2P + tier-1 combinations
    for supplier, t2p_magnitude, supplier_magnitude, duration in [
        ("S1A", 0.25, 0.75, 3),
        ("S1B", 0.25, 0.75, 3),
        ("S1C", 0.25, 0.75, 3),
        ("S1A", 0.40, 1.00, 3),
        ("S1B", 0.40, 1.00, 3),
        ("S1C", 0.40, 1.00, 3),
    ]:
        add(
            "T2P + tier-1 combination",
            [
                Event("T2P", t2p_magnitude, start, duration),
                Event(supplier, supplier_magnitude, start, duration),
            ],
        )

    # 4 demand spike + T2P impairment scenarios
    for t2p_magnitude, demand_magnitude, duration in [
        (0.20, 0.20, 3),
        (0.30, 0.25, 3),
        (0.35, 0.30, 4),
        (0.45, 0.40, 3),
    ]:
        add(
            "Demand spike + T2P impairment",
            [
                Event("T2P", t2p_magnitude, start, duration),
                Event("demand", demand_magnitude, start, duration),
            ],
        )

    # 5 sequential T2P -> corridor disruptions
    for t2p_magnitude, corridor_magnitude, t2p_duration, corridor_duration, gap in [
        (0.30, 0.30, 3, 2, 3),
        (0.40, 0.30, 3, 2, 3),
        (0.40, 0.40, 4, 2, 3),
        (0.50, 0.40, 4, 3, 3),
        (0.60, 0.50, 3, 3, 2),
    ]:
        add(
            "Sequential T2P to corridor disruption",
            [
                Event("T2P", t2p_magnitude, start, t2p_duration, recovery_weeks=4),
                Event("corridor", corridor_magnitude, start + gap, corridor_duration, recovery_weeks=2),
            ],
        )

    # 5 sequential corridor -> T2P disruptions
    for corridor_magnitude, t2p_magnitude, corridor_duration, t2p_duration, gap in [
        (0.30, 0.30, 2, 3, 2),
        (0.40, 0.35, 2, 3, 2),
        (0.50, 0.40, 3, 3, 3),
        (0.60, 0.50, 3, 4, 3),
        (0.70, 0.60, 3, 3, 2),
    ]:
        add(
            "Sequential corridor to T2P disruption",
            [
                Event("corridor", corridor_magnitude, start, corridor_duration, recovery_weeks=4),
                Event("T2P", t2p_magnitude, start + gap, t2p_duration, recovery_weeks=2),
            ],
        )

    return out


def rst_candidate_library() -> List[Pathway]:
    out: List[Pathway] = []
    s = 10

    # Hidden convergent dependency (10)
    hcd = [
        ("RST-01", 0.30, 3), ("RST-02", 0.30, 4), ("RST-03", 0.35, 5), ("RST-04", 0.40, 3),
        ("RST-05", 0.50, 3), ("RST-06", 0.60, 3), ("RST-07", 0.20, 3), ("RST-08", 0.25, 2),
        ("RST-09", 0.30, 2), ("RST-10", 0.15, 6),
    ]
    for pid, m, d in hcd:
        out.append(_single_event_pathway(pid, "Hidden convergent dependency", "T2P", m, d, start=s))

    # Compounding moderate disturbance (15)
    cmd_specs = [
        ("RST-11", [Event("T2P", 0.25, s, 3), Event("corridor", 0.20, s, 3)]),
        ("RST-12", [Event("T2P", 0.20, s, 4), Event("corridor", 0.25, s, 4)]),
        ("RST-13", [Event("T2P", 0.30, s, 3), Event("corridor", 0.30, s, 3)]),
        ("RST-14", [Event("T2P", 0.20, s, 4), Event("demand", 0.20, s, 4), Event("corridor", 0.15, s, 4)]),
        ("RST-15", [Event("T2P", 0.30, s, 3), Event("demand", 0.25, s, 3)]),
        ("RST-16", [Event("T2P", 0.35, s, 4), Event("corridor", 0.30, s, 4), Event("demand", 0.15, s, 4)]),
        ("RST-17", [Event("T2P", 0.40, s, 3), Event("corridor", 0.20, s, 3)]),
        ("RST-18", [Event("T2P", 0.20, s, 3), Event("corridor", 0.10, s, 3)]),
        ("RST-19", [Event("T2P", 0.15, s, 4), Event("demand", 0.15, s, 4)]),
        ("RST-20", [Event("T2P", 0.25, s, 3), Event("demand", 0.10, s, 3)]),
        ("RST-21", [Event("T2P", 0.20, s, 4), Event("corridor", 0.15, s, 4)]),
        ("RST-22", [Event("T2P", 0.30, s, 2), Event("demand", 0.10, s, 2)]),
        ("RST-23", [Event("T2P", 0.20, s, 2), Event("demand", 0.20, s, 2)]),
        ("RST-24", [Event("corridor", 0.30, s, 3), Event("demand", 0.25, s, 3)]),
        ("RST-25", [Event("T2P", 0.25, s, 2), Event("demand", 0.15, s, 2), Event("corridor", 0.10, s, 2)]),
    ]
    for pid, ev in cmd_specs:
        out.append(Pathway(pid, "Compounding moderate disturbance", tuple(ev)))

    # Propagation-amplified cascade (8)
    pac_specs = [
        ("RST-26", [Event("T2P", 0.40, s, 3), Event("demand", 0.30, s, 3)]),
        ("RST-27", [Event("T2P", 0.50, s, 3), Event("demand", 0.40, s, 3)]),
        ("RST-28", [Event("T2P", 0.60, s, 2), Event("demand", 0.25, s, 2)]),
        ("RST-29", [Event("T2P", 0.35, s, 3), Event("demand", 0.20, s, 3)]),
        ("RST-30", [Event("T2P", 0.30, s, 3), Event("demand", 0.15, s, 3)]),
        ("RST-31", [Event("T2P", 0.40, s, 3)]),
        ("RST-32", [Event("corridor", 0.50, s, 3)]),
        ("RST-33", [Event("T2P", 0.30, s, 4), Event("demand", 0.20, s, 4)]),
    ]
    for pid, ev in pac_specs:
        out.append(Pathway(pid, "Propagation-amplified cascade", tuple(ev)))

    # Recovery lag and temporal overlap (9)
    rl_specs = [
        ("RST-34", [Event("T2P", 0.40, s, 4), Event("corridor", 0.50, s + 3, 2)]),
        ("RST-35", [Event("T2P", 0.50, s, 4), Event("T2P", 0.30, s + 3, 2)]),
        ("RST-36", [Event("T2P", 0.60, s, 3), Event("corridor", 0.40, s + 2, 2)]),
        ("RST-37", [Event("T2P", 0.30, s, 3), Event("corridor", 0.30, s + 5, 2)]),
        ("RST-38", [Event("T2P", 0.40, s, 3), Event("demand", 0.20, s + 3, 2)]),
        ("RST-39", [Event("T2P", 0.25, s, 4), Event("corridor", 0.20, s + 4, 2)]),
        ("RST-40", [Event("T2P", 0.50, s, 4), Event("corridor", 0.20, s + 7, 2)]),
        ("RST-41", [Event("corridor", 0.60, s, 3), Event("T2P", 0.30, s + 3, 2)]),
        ("RST-42", [Event("T2P", 0.35, s, 3), Event("demand", 0.30, s + 3, 2)]),
    ]
    for pid, ev in rl_specs:
        out.append(Pathway(pid, "Recovery lag and temporal overlap", tuple(ev)))

    # Response-sensitive mechanism probes (4). These can classify as
    # Structural or Adaptation-limited after active evaluation, but they are
    # not a separate mechanism family.
    al_specs = [
        ("RST-43", "Hidden convergent dependency", [Event("T2P", 0.20, s, 4)]),
        ("RST-44", "Hidden convergent dependency", [Event("T2P", 0.15, s, 6)]),
        ("RST-45", "Compounding moderate disturbance", [Event("corridor", 0.25, s, 4), Event("demand", 0.10, s, 4)]),
        ("RST-46", "Hidden convergent dependency", [Event("T2P", 0.20, s + 3, 3)]),  # delayed detection/escalation proxy
    ]
    for pid, family, ev in al_specs:
        out.append(Pathway(pid, family, tuple(ev)))

    # Concentrated dependency (4)
    cd_specs = [
        ("RST-47", [Event("S1A", 1.00, s, 3), Event("T2P", 0.40, s, 3)]),
        ("RST-48", [Event("S1A", 1.00, s, 4)]),
        ("RST-49", [Event("S1B", 1.00, s, 2), Event("T2P", 0.30, s, 2)]),
        ("RST-50", [Event("S1C", 1.00, s, 3), Event("corridor", 0.30, s, 3)]),
    ]
    for pid, ev in cd_specs:
        out.append(Pathway(pid, "Concentrated dependency", tuple(ev)))

    return out


# ---------------------------
# Mechanism / criticality logic
# ---------------------------


def pathway_elements(pathway: Pathway) -> Tuple[str, ...]:
    elems = []
    for e in pathway.events:
        canonical_target = _canonical_element(e.target)
        if canonical_target in {"T2P", "corridor", "S1A", "S1B", "S1C"}:
            elems.append(canonical_target)
    return tuple(sorted(set(elems)))


def pathway_severity(service_levels: Sequence[float], threshold: FailureThreshold = MODERATE) -> float:
    deficits = [max(0.0, threshold.alpha - sl) for sl in service_levels]
    return float(sum(deficits))


def pathway_plausibility(pathway: Pathway) -> float:
    # Eq. (12)-style plausibility: product of marginal densities for
    # magnitude, duration, and inter-arrival gaps.
    # P(pi) = exp(sum log f_delta + log f_tau + log f_lambda)
    if not pathway.events:
        return 1.0

    mean = 3.2
    var = 8.1
    p = mean / var
    n = mean * p / (1.0 - p)

    events = sorted(pathway.events, key=lambda e: e.start_week)
    logp = 0.0
    prev_t: Optional[int] = None

    for e in events:
        f_delta = _scaled_beta_pdf(e.magnitude)
        f_tau = float(nbinom.pmf(max(1, int(e.duration_weeks)), n, p))
        if prev_t is None:
            delta_t = max(1, int(e.start_week) + 1)
        else:
            delta_t = max(1, int(e.start_week - prev_t))
        f_lambda = float(expon(scale=18.0).pdf(delta_t))
        prev_t = int(e.start_week)

        logp += math.log(max(f_delta, 1e-12))
        logp += math.log(max(f_tau, 1e-12))
        logp += math.log(max(f_lambda, 1e-12))

    return float(math.exp(logp))


def classify_mechanism(
    events: Sequence[Event],
    cfg: NetworkConfig,
    passive_result: SimulationResult,
    active_result: SimulationResult,
    threshold: FailureThreshold = MODERATE,
) -> str:
    # Mechanism attribution aligned to paper Section 5.3.1 rules.
    passive_crossed, onset = threshold_cross_info(passive_result.service_levels, threshold)
    if not passive_crossed:
        return "Non-failure"
    t_star = int(onset or 0)

    active_set = []
    for e in events:
        start = e.start_week
        end = e.start_week + e.duration_weeks + e.recovery_weeks
        if start <= t_star <= end:
            active_set.append(e)
    if not active_set:
        active_set = list(events)

    has_t2p = any(_canonical_element(e.target) == "T2P" for e in active_set)
    has_corridor = any(e.target == "corridor" for e in active_set)
    has_demand = any(e.target == "demand" for e in active_set)

    single_cross_targets = []
    for e in active_set:
        one = run_simulation([e], cfg=cfg, active_mode=False)
        one_cross, _ = threshold_cross_info(one.service_levels, threshold)
        if one_cross:
            single_cross_targets.append(_canonical_element(e.target))

    # Compounding moderate disturbance (CMD): joint failure, no single event sufficient.
    joint = run_simulation(active_set, cfg=cfg, active_mode=False)
    joint_cross, _ = threshold_cross_info(joint.service_levels, threshold)
    if joint_cross and not single_cross_targets:
        return "Compounding moderate disturbance"

    # Propagation-amplified cascade (PAC)
    b_prev = passive_result.backlog_series[t_star - 1] if t_star > 0 else 0.0
    if has_t2p and (has_demand or b_prev >= 0.30 * cfg.demand_per_week):
        return "Propagation-amplified cascade"

    # Recovery lag and temporal overlap (RL)
    if _has_recovery_overlap(active_set):
        return "Recovery lag and temporal overlap"

    # Adaptation-limited failure is retained here as a Monte Carlo failure
    # type used for denominator diagnostics; it is not used as an RST
    # pathway-generation family.
    active_crossed, _ = threshold_cross_info(active_result.service_levels, threshold)
    if not active_crossed:
        return "Adaptation-limited failure"

    # Hidden convergent dependency (HCD)
    if has_t2p:
        depleted = 0
        for n in cfg.tier1_nodes:
            arr = passive_result.tier1_inventory_series[n]
            inv_at_t = arr[t_star] if t_star < len(arr) else arr[-1]
            if inv_at_t <= 0.10 * cfg.tier1_initial_inventory:
                depleted += 1
        non_t2p_active = any(_canonical_element(e.target) != "T2P" for e in active_set)
        if depleted >= 2 and not non_t2p_active:
            return "Hidden convergent dependency"

    # Concentrated dependency (CD): single non-convergent element can trigger failure.
    if any(t in {"S1A", "S1B", "S1C", "corridor"} for t in single_cross_targets) and not has_t2p:
        return "Concentrated dependency"

    # Default
    if has_t2p:
        return "Hidden convergent dependency"
    return "Compounding moderate disturbance"


def _has_recovery_overlap(events: Sequence[Event]) -> bool:
    # True if an event starts during another event's recovery window.
    for i, a in enumerate(events):
        rec_start = a.start_week + a.duration_weeks
        rec_end = rec_start + a.recovery_weeks
        for j, b in enumerate(events):
            if i == j:
                continue
            if rec_start < b.start_week <= rec_end:
                return True
    return False


def mechanism_guided_rst_search(
    candidates: Sequence[Pathway],
    cfg: NetworkConfig,
    budget: int = 50,
    threshold: FailureThreshold = MODERATE,
) -> Tuple[List[PathwayEval], Dict[str, float]]:
    # Algorithm 1 style: evaluate candidate pathways under passive, then re-evaluate failures under active.
    selected = list(candidates)[:budget]

    evals: List[PathwayEval] = []
    for p in selected:
        lean_flag = p.pathway_id in {"RST-33"}  # pathway-specific proxy from table text
        preload = 120.0 if p.pathway_id in {"RST-29", "RST-32"} else 0.0

        passive = run_simulation(
            p.events,
            cfg=cfg,
            active_mode=False,
            lean_tier1_inventory=lean_flag,
            preload_backlog=preload,
        )
        pass_cross, _ = threshold_cross_info(passive.service_levels, threshold)

        if pass_cross:
            active = run_simulation(
                p.events,
                cfg=cfg,
                active_mode=True,
                lean_tier1_inventory=lean_flag,
                preload_backlog=preload,
            )
            act_cross, _ = threshold_cross_info(active.service_levels, threshold)
            classification = "Structural" if act_cross else "Adaptation-limited"
            controllability = 0.10 if act_cross else 0.90
            severity = pathway_severity(passive.service_levels, threshold)
            plaus = pathway_plausibility(p)
            evals.append(
                PathwayEval(
                    pathway_id=p.pathway_id,
                    family=p.family,
                    passive_failure=True,
                    active_failure=act_cross,
                    classification=classification,
                    severity=severity,
                    plausibility=plaus,
                    controllability=controllability,
                    weight=0.0,
                    elements=pathway_elements(p),
                    note=p.note,
                )
            )
        else:
            evals.append(
                PathwayEval(
                    pathway_id=p.pathway_id,
                    family=p.family,
                    passive_failure=False,
                    active_failure=None,
                    classification="Non-failure",
                    severity=0.0,
                    plausibility=pathway_plausibility(p),
                    controllability=1.0,
                    weight=0.0,
                    elements=pathway_elements(p),
                    note=p.note,
                )
            )

    # Compute weights only over passive failures.
    fail_idx = [i for i, e in enumerate(evals) if e.passive_failure]
    sev_norm = normalize([evals[i].severity for i in fail_idx])
    pl_norm = normalize([evals[i].plausibility for i in fail_idx])
    ctrl_norm = normalize([1.0 - evals[i].controllability for i in fail_idx])
    for k, i in enumerate(fail_idx):
        evals[i].weight = 0.5 * sev_norm[k] + 0.3 * pl_norm[k] + 0.2 * ctrl_norm[k]

    # Failure-conditioned criticality
    elements = ["T2P", "corridor", "S1A", "S1B", "S1C"]
    scores = {z: 0.0 for z in elements}
    for e in evals:
        if not e.passive_failure:
            continue
        for z in e.elements:
            scores[z] += e.weight

    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return evals, scores


# ---------------------------
# Monte Carlo ground truth
# ---------------------------


def generate_random_pathway(
    rng: np.random.Generator,
    cfg: NetworkConfig,
) -> Tuple[List[Event], bool, bool, str]:
    horizon = cfg.horizon_weeks
    events: List[Event] = []

    t = int(rng.integers(0, cfg.mc_start_week_max + 1))
    upstream_capacity = _upstream_capacity_map(cfg)
    upstream_total_capacity = sum(upstream_capacity.values())
    upstream_targets = list(cfg.upstream_nodes)
    upstream_probs = [
        cfg.mc_target_probs[0] * (upstream_capacity[upstream] / upstream_total_capacity)
        for upstream in upstream_targets
    ]
    targets = upstream_targets + ["corridor", "S1A", "S1B", "S1C", "demand"]
    probs = np.array(
        upstream_probs + list(cfg.mc_target_probs[1:]),
        dtype=float,
    )
    probs = probs / probs.sum()
    n_events = 0

    while t < horizon and n_events < cfg.mc_max_events:
        target = str(rng.choice(targets, p=probs))
        mag = float(truncated_beta(rng, size=1)[0])
        dur = sample_duration_negative_binomial(rng)
        rec = int(rng.integers(1, 5))
        if target == "demand":
            # Keep demand shocks in a plausible range.
            mag = float(np.clip(mag, 0.10, 0.50))
        events.append(Event(target=target, magnitude=mag, start_week=t, duration_weeks=dur, recovery_weeks=rec))
        n_events += 1
        jump = int(max(1, round(rng.exponential(cfg.mc_interarrival_mean_weeks))))
        t += jump

    passive = run_simulation(events, cfg=cfg, active_mode=False)
    active = run_simulation(events, cfg=cfg, active_mode=True)
    m_pass, _ = threshold_cross_info(passive.service_levels, MODERATE)
    m_act, _ = threshold_cross_info(active.service_levels, MODERATE)
    family = classify_mechanism(
        events=events,
        cfg=cfg,
        passive_result=passive,
        active_result=active,
        threshold=MODERATE,
    )
    return events, m_pass, m_act, family


def monte_carlo_ground_truth(
    replications: int,
    cfg: NetworkConfig,
    seed: int = 42,
) -> pd.DataFrame:
    root = np.random.SeedSequence(seed)
    child_seeds = root.spawn(replications)
    rows = []
    for r in range(replications):
        rng = np.random.default_rng(child_seeds[r])
        events, m_pass, m_act, family = generate_random_pathway(rng, cfg)
        passive_active_classification = (
            "Non-failure"
            if not m_pass
            else ("Structural" if m_act else "Adaptation-limited")
        )
        sev_pass = run_simulation(events, cfg=cfg, active_mode=False)
        sev_active = run_simulation(events, cfg=cfg, active_mode=True)
        s_pass, _ = threshold_cross_info(sev_pass.service_levels, SEVERE)
        s_act, _ = threshold_cross_info(sev_active.service_levels, SEVERE)
        targets_in_events = {_canonical_element(e.target) for e in events}
        rows.append(
            {
                "replication": r + 1,
                "moderate_passive_failure": bool(m_pass),
                "moderate_active_failure": bool(m_act),
                "passive_active_classification": passive_active_classification,
                "severe_passive_failure": bool(s_pass),
                "severe_active_failure": bool(s_act),
                "primary_family": family,
                "event_count": len(events),
                "max_backlog_passive": float(max(sev_pass.backlog_series) if sev_pass.backlog_series else 0.0),
                "has_T2P": bool("T2P" in targets_in_events),
                "has_corridor": bool("corridor" in targets_in_events),
                "has_S1A": bool("S1A" in targets_in_events),
                "has_S1B": bool("S1B" in targets_in_events),
                "has_S1C": bool("S1C" in targets_in_events),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------
# FST baseline
# ---------------------------


def run_fst(
    scenarios: Sequence[Pathway],
    cfg: NetworkConfig,
) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        sim = run_simulation(s.events, cfg=cfg, active_mode=False)
        mod, _ = threshold_cross_info(sim.service_levels, MODERATE)
        sev, _ = threshold_cross_info(sim.service_levels, SEVERE)
        cat = s.family.lower()
        # FST classification follows visible scenario taxonomy (structural opacity),
        # not full mechanism back-attribution.
        if "t2p partial failure" in cat:
            family = "Hidden convergent dependency"
        elif "demand spike + t2p impairment" in cat:
            family = "Propagation-amplified cascade"
        elif "sequential" in cat:
            family = "Recovery lag and temporal overlap"
        elif (
            "simultaneous" in cat
            or "concurrent" in cat
            or "high-severity" in cat
            or "combination" in cat
        ):
            family = "Compounding moderate disturbance"
        else:
            family = "Concentrated dependency"
        rows.append(
            {
                "scenario_id": s.pathway_id,
                "category": s.family,
                "moderate_crossing": bool(mod),
                "severe_crossing": bool(sev),
                "assigned_family": family,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------
# Comparative study orchestration
# ---------------------------


def run_study(
    replications: int,
    output_dir: Path,
    seed: int = 42,
    budget: int = 50,
) -> Dict[str, object]:
    cfg = NetworkConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monte Carlo ground truth
    mc = monte_carlo_ground_truth(replications=replications, cfg=cfg, seed=seed)
    mc.to_csv(output_dir / "monte_carlo_ground_truth.csv", index=False)

    moderate_failures = mc[mc["moderate_passive_failure"]]
    severe_failures = mc[mc["severe_passive_failure"]]

    # FST
    fst_df = run_fst(fst_library(), cfg=cfg)
    fst_df.to_csv(output_dir / "fst_results.csv", index=False)
    fst_discovered = set(fst_df.loc[fst_df["moderate_crossing"], "assigned_family"].tolist())

    # RST
    rst_evals, crit_scores = mechanism_guided_rst_search(
        candidates=rst_candidate_library(),
        cfg=cfg,
        budget=budget,
        threshold=MODERATE,
    )
    rst_df = pd.DataFrame(
        [
            {
                "pathway_id": e.pathway_id,
                "family": e.family,
                "passive_failure": e.passive_failure,
                "active_failure": e.active_failure,
                "classification": e.classification,
                "severity": e.severity,
                "plausibility": e.plausibility,
                "controllability": e.controllability,
                "weight": e.weight,
                "elements": ",".join(e.elements),
            }
            for e in rst_evals
        ]
    )
    rst_df.to_csv(output_dir / "rst_results.csv", index=False)
    rst_discovered = set(rst_df.loc[rst_df["passive_failure"], "family"].tolist())

    # Coverage
    def _coverage(families: set, df: pd.DataFrame, col: str) -> float:
        if df.empty:
            return 0.0
        hit = (df[col].isin(families)).sum()
        return float(hit / len(df))

    def _non_adaptation_limited(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["primary_family"] != "Adaptation-limited failure"]

    def _coverage_full_denominator(families: set, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        assessable = _non_adaptation_limited(df)
        hit = (assessable["primary_family"].isin(families)).sum()
        return float(hit / len(df))

    def _coverage_non_alf_denominator(families: set, df: pd.DataFrame) -> float:
        assessable = _non_adaptation_limited(df)
        return _coverage(families, assessable, "primary_family")

    cmd_pac_rlto = {
        "Compounding moderate disturbance",
        "Propagation-amplified cascade",
        "Recovery lag and temporal overlap",
    }

    def _coverage_family_subset(families: set, df: pd.DataFrame, subset: set) -> float:
        assessable = _non_adaptation_limited(df)
        subset_df = assessable[assessable["primary_family"].isin(subset)]
        return _coverage(families, subset_df, "primary_family")

    cov_fst_mod = _coverage_full_denominator(fst_discovered, moderate_failures)
    cov_fst_sev = _coverage_full_denominator(fst_discovered, severe_failures)
    cov_rst_mod = _coverage_full_denominator(rst_discovered, moderate_failures)
    cov_rst_sev = _coverage_full_denominator(rst_discovered, severe_failures)
    cov_fst_mod_non_alf = _coverage_non_alf_denominator(fst_discovered, moderate_failures)
    cov_rst_mod_non_alf = _coverage_non_alf_denominator(rst_discovered, moderate_failures)
    cov_fst_cmd_pac_rlto = _coverage_family_subset(fst_discovered, moderate_failures, cmd_pac_rlto)
    cov_rst_cmd_pac_rlto = _coverage_family_subset(rst_discovered, moderate_failures, cmd_pac_rlto)

    # Summary objects
    summary = {
        "config": {
            "replications": replications,
            "seed": seed,
            "budget": budget,
            "horizon_weeks": cfg.horizon_weeks,
        },
        "monte_carlo": {
            "moderate_failures_count": int(len(moderate_failures)),
            "severe_failures_count": int(len(severe_failures)),
            "family_distribution_moderate": moderate_failures["primary_family"].value_counts().to_dict(),
            "classification_distribution_moderate": moderate_failures[
                "passive_active_classification"
            ].value_counts().to_dict(),
        },
        "fst": {
            "total_scenarios": int(len(fst_df)),
            "moderate_crossings": int(fst_df["moderate_crossing"].sum()),
            "severe_crossings": int(fst_df["severe_crossing"].sum()),
            "discovered_mechanism_families": sorted(fst_discovered),
            "discovered_families": sorted(fst_discovered),
            "coverage_moderate": cov_fst_mod,
            "coverage_severe": cov_fst_sev,
            "coverage_moderate_full_denominator": cov_fst_mod,
            "coverage_moderate_non_alf_denominator": cov_fst_mod_non_alf,
            "coverage_cmd_pac_rlto_moderate": cov_fst_cmd_pac_rlto,
        },
        "rst": {
            "total_candidates": int(len(rst_df)),
            "moderate_passive_failures": int(rst_df["passive_failure"].sum()),
            "moderate_active_failures": int(
                rst_df.loc[rst_df["active_failure"].notna(), "active_failure"].astype(bool).sum()
            ),
            "moderate_adaptation_limited_failures": int(
                rst_df["passive_failure"].sum()
                - rst_df.loc[rst_df["active_failure"].notna(), "active_failure"].astype(bool).sum()
            ),
            "discovered_mechanism_families": sorted(rst_discovered),
            "discovered_families": sorted(rst_discovered),
            "coverage_moderate": cov_rst_mod,
            "coverage_severe": cov_rst_sev,
            "coverage_moderate_full_denominator": cov_rst_mod,
            "coverage_moderate_non_alf_denominator": cov_rst_mod_non_alf,
            "coverage_cmd_pac_rlto_moderate": cov_rst_cmd_pac_rlto,
            "criticality_scores": crit_scores,
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    crit_df = pd.DataFrame([{"element": k, "importance": v} for k, v in crit_scores.items()]).sort_values(
        "importance", ascending=False
    )
    crit_df.to_csv(output_dir / "criticality_scores.csv", index=False)

    return summary


# ---------------------------
# CLI
# ---------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RST implementation for the IJPR reverse stress testing paper."
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=1000,
        help="Monte Carlo replications (paper uses 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (paper metadata uses 42).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=50,
        help="Pathway evaluation budget for RST (paper uses 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV/JSON outputs.",
    )
    args = parser.parse_args()

    summary = run_study(
        replications=args.replications,
        output_dir=args.output_dir,
        seed=args.seed,
        budget=args.budget,
    )

    print("Study finished.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"MC moderate failures: {summary['monte_carlo']['moderate_failures_count']}")
    print(f"MC severe failures:   {summary['monte_carlo']['severe_failures_count']}")
    print(
        "FST coverage (moderate/severe): "
        f"{summary['fst']['coverage_moderate']:.3f} / {summary['fst']['coverage_severe']:.3f}"
    )
    print(
        "RST coverage (moderate/severe): "
        f"{summary['rst']['coverage_moderate']:.3f} / {summary['rst']['coverage_severe']:.3f}"
    )


if __name__ == "__main__":
    main()
