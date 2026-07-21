#!/usr/bin/env python3
"""
Reviewer 2-6 robustness runs across seeds and a small topology variant.

Topologies:
- A: current shared-T2P single-corridor network.
- B: partial upstream diversification: S1A/S1B source from T2P-A,
  S1C sources from T2P-B, corridor unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from paper_rst_ijpr import (
    MODERATE,
    NetworkConfig,
    fst_enhanced_library,
    fst_library,
    mechanism_guided_rst_search,
    monte_carlo_ground_truth,
    rst_candidate_library,
    run_fst,
)


ALF = "Adaptation-limited failure"
CMD_PAC_RLTO = {
    "Compounding moderate disturbance",
    "Propagation-amplified cascade",
    "Recovery lag and temporal overlap",
}
ELEMENTS = ["T2P", "corridor", "S1A", "S1B", "S1C"]


def topology_config(topology: str) -> NetworkConfig:
    if topology == "A":
        return NetworkConfig()
    if topology == "B":
        return NetworkConfig(
            upstream_nodes=("T2P-A", "T2P-B"),
            tier1_upstream_nodes=("T2P-A", "T2P-A", "T2P-B"),
        )
    raise ValueError(f"Unknown topology: {topology}")


def topology_description(topology: str) -> str:
    if topology == "A":
        return "Shared-T2P single-corridor network"
    if topology == "B":
        return "Partial upstream diversification: S1A/S1B from T2P-A; S1C from T2P-B"
    raise ValueError(f"Unknown topology: {topology}")


def dense_ranks(scores: Dict[str, float]) -> Dict[str, int]:
    return (
        pd.Series(scores)
        .sort_values(ascending=False)
        .rank(ascending=False, method="dense")
        .astype(int)
        .to_dict()
    )


def coverage(discovered_families: set[str], failures: pd.DataFrame) -> float:
    if failures.empty:
        return 0.0
    return float(failures["primary_family"].isin(discovered_families).mean())


def family_subset_coverage(
    discovered_families: set[str],
    failures: pd.DataFrame,
    subset: set[str],
) -> float:
    subset_failures = failures[failures["primary_family"].isin(subset)]
    return coverage(discovered_families, subset_failures)


def eval_rows_to_dataframe(evals) -> pd.DataFrame:
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
                "weight": float(pathway_eval.weight),
                "elements": ",".join(pathway_eval.elements),
            }
            for pathway_eval in evals
        ]
    )


def run_one(topology: str, seed: int, replications: int, budget: int) -> Dict[str, object]:
    cfg = topology_config(topology)
    mc = monte_carlo_ground_truth(replications=replications, cfg=cfg, seed=seed)
    moderate_failures = mc[mc["moderate_passive_failure"]].copy()
    non_alf_failures = moderate_failures[moderate_failures["primary_family"] != ALF].copy()

    fst_original = run_fst(fst_library(), cfg=cfg)
    fst_enhanced = run_fst(fst_enhanced_library(), cfg=cfg)
    rst_evals, rst_scores = mechanism_guided_rst_search(
        candidates=rst_candidate_library(),
        cfg=cfg,
        budget=budget,
        threshold=MODERATE,
    )
    rst = eval_rows_to_dataframe(rst_evals)

    fst_original_families = set(
        fst_original.loc[fst_original["moderate_crossing"], "assigned_family"].tolist()
    )
    fst_enhanced_families = set(
        fst_enhanced.loc[fst_enhanced["moderate_crossing"], "assigned_family"].tolist()
    )
    rst_families = set(rst.loc[rst["passive_failure"], "family"].tolist())

    ranks = dense_ranks(rst_scores)
    rst_passive = int(rst["passive_failure"].sum())
    rst_structural = int(
        sum(
            False if pd.isna(active_failure) else bool(active_failure)
            for active_failure in rst["active_failure"].tolist()
        )
    )
    rst_adaptation_limited = int(
        ((rst["passive_failure"] == True) & (rst["active_failure"] == False)).sum()
    )

    return {
        "topology": topology,
        "topology_description": topology_description(topology),
        "seed": seed,
        "replications": replications,
        "mc_moderate_failures": int(mc["moderate_passive_failure"].sum()),
        "mc_severe_failures": int(mc["severe_passive_failure"].sum()),
        "fst_original_crossings": int(fst_original["moderate_crossing"].sum()),
        "fst_enhanced_crossings": int(fst_enhanced["moderate_crossing"].sum()),
        "rst_passive_pathways": rst_passive,
        "rst_structural_pathways": rst_structural,
        "rst_adaptation_limited_pathways": rst_adaptation_limited,
        "upstream_processor_criticality_rank": int(ranks.get("T2P", 0)),
        "corridor_criticality_rank": int(ranks.get("corridor", 0)),
        "rst_t2p_if": float(rst_scores.get("T2P", 0.0)),
        "rst_corridor_if": float(rst_scores.get("corridor", 0.0)),
        "fst_original_non_alf_coverage": coverage(fst_original_families, non_alf_failures),
        "fst_enhanced_non_alf_coverage": coverage(fst_enhanced_families, non_alf_failures),
        "rst_non_alf_coverage": coverage(rst_families, non_alf_failures),
        "fst_original_cmd_pac_rlto_coverage": family_subset_coverage(
            fst_original_families, non_alf_failures, CMD_PAC_RLTO
        ),
        "fst_enhanced_cmd_pac_rlto_coverage": family_subset_coverage(
            fst_enhanced_families, non_alf_failures, CMD_PAC_RLTO
        ),
        "rst_cmd_pac_rlto_coverage": family_subset_coverage(
            rst_families, non_alf_failures, CMD_PAC_RLTO
        ),
    }


def spread_table(runs: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups = [(name, group) for name, group in runs.groupby("topology")]
    groups.append(("All", runs))
    for topology, group in groups:
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "topology": topology,
                    "metric": metric,
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def generate_r26_robustness(
    output_dir: Path,
    seeds: List[int],
    replications: int,
    budget: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for topology in ["A", "B"]:
        for seed in seeds:
            print(f"Running topology {topology}, seed {seed}...")
            rows.append(run_one(topology=topology, seed=seed, replications=replications, budget=budget))

    runs = pd.DataFrame(rows)
    runs.to_csv(output_dir / "r26_robustness_runs.csv", index=False)

    metrics = [
        "mc_moderate_failures",
        "mc_severe_failures",
        "fst_original_crossings",
        "fst_enhanced_crossings",
        "rst_passive_pathways",
        "rst_structural_pathways",
        "rst_adaptation_limited_pathways",
        "upstream_processor_criticality_rank",
        "corridor_criticality_rank",
        "fst_original_non_alf_coverage",
        "fst_enhanced_non_alf_coverage",
        "rst_non_alf_coverage",
        "fst_original_cmd_pac_rlto_coverage",
        "fst_enhanced_cmd_pac_rlto_coverage",
        "rst_cmd_pac_rlto_coverage",
    ]
    spread = spread_table(runs, metrics)
    spread.to_csv(output_dir / "r26_robustness_spread.csv", index=False)

    topology_summary = (
        runs.groupby("topology", as_index=False)
        .agg(
            topology_description=("topology_description", "first"),
            seeds=("seed", lambda values: ",".join(str(int(value)) for value in values)),
            runs=("seed", "count"),
            mean_mc_moderate_failures=("mc_moderate_failures", "mean"),
            mean_mc_severe_failures=("mc_severe_failures", "mean"),
            mean_fst_original_crossings=("fst_original_crossings", "mean"),
            mean_fst_enhanced_crossings=("fst_enhanced_crossings", "mean"),
            mean_rst_passive_pathways=("rst_passive_pathways", "mean"),
            mean_rst_structural_pathways=("rst_structural_pathways", "mean"),
            mean_rst_adaptation_limited_pathways=("rst_adaptation_limited_pathways", "mean"),
            mean_rst_non_alf_coverage=("rst_non_alf_coverage", "mean"),
            mean_rst_cmd_pac_rlto_coverage=("rst_cmd_pac_rlto_coverage", "mean"),
        )
    )
    topology_summary.to_csv(output_dir / "r26_robustness_topology_summary.csv", index=False)

    manifest = {
        "reviewer_comment": "R2-6",
        "seeds": seeds,
        "replications": replications,
        "budget": budget,
        "topologies": {
            "A": topology_description("A"),
            "B": topology_description("B"),
        },
        "files": [
            "r26_robustness_runs.csv",
            "r26_robustness_spread.csv",
            "r26_robustness_topology_summary.csv",
        ],
    }
    with (output_dir / "r26_robustness_manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R2-6 seed/topology robustness checks.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_r26_robustness"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()

    generate_r26_robustness(
        output_dir=args.output_dir,
        seeds=args.seeds,
        replications=args.replications,
        budget=args.budget,
    )
    print(f"R2-6 robustness outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
