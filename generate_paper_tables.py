#!/usr/bin/env python3
"""
Generate paper-style tables (Table 4-10 + Appendix A/B/C) from the RST model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from paper_rst_ijpr import (
    MODERATE,
    SEVERE,
    FailureThreshold,
    NetworkConfig,
    Pathway,
    fst_enhanced_library,
    fst_library,
    mechanism_guided_rst_search,
    monte_carlo_ground_truth,
    run_fst,
    run_study,
)

MECHANISM_FAMILIES = [
    "Hidden convergent dependency",
    "Compounding moderate disturbance",
    "Propagation-amplified cascade",
    "Recovery lag and temporal overlap",
    "Concentrated dependency",
]

CMD_PAC_RLTO = {
    "Compounding moderate disturbance",
    "Propagation-amplified cascade",
    "Recovery lag and temporal overlap",
}

ADAPTATION_LIMITED = "Adaptation-limited"
ADAPTATION_LIMITED_FAMILY = "Adaptation-limited failure"
STRUCTURAL = "Structural"


def _event_spec(pathway: Pathway) -> str:
    parts = []
    for e in pathway.events:
        parts.append(f"{e.target}: {int(round(e.magnitude * 100))}% for {e.duration_weeks}w @t{e.start_week}")
    return " + ".join(parts)


def _fst_group(cat: str) -> str:
    c = cat.lower()
    if "single tier-1 failure" in c:
        return "Individual tier-1 failures"
    if "t2p partial failure" in c:
        return "T2P partial failures"
    if "corridor disruption" in c:
        return "Corridor disruptions"
    if "demand spike" in c:
        return "Demand spikes"
    return "Prespecified combinations"


def _family_intervention(fam: str) -> str:
    m = {
        "Concentrated dependency": "Diversification, redundancy, strategic inventory, or backup capacity.",
        "Hidden convergent dependency": "Upstream T2P qualification and alternative processor sourcing.",
        "Compounding moderate disturbance": "Slack restoration and stress-triggered allocation.",
        "Propagation-amplified cascade": "Segmentation and containment redesign.",
        "Recovery lag and temporal overlap": "Recovery sequencing and time-buffered restoration.",
    }
    return m.get(fam, "")


def _rank_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
    s = pd.Series(scores).sort_values(ascending=False)
    # Dense rank (ties share rank)
    return s.rank(ascending=False, method="dense").astype(int).to_dict()


def _compute_fst_element_scores(fst_df: pd.DataFrame, scenarios: List[Pathway]) -> Dict[str, float]:
    crossing_ids = set(fst_df.loc[fst_df["moderate_crossing"], "scenario_id"].tolist())
    score = {k: 0.0 for k in ["T2P", "corridor", "S1A", "S1B", "S1C"]}
    for s in scenarios:
        if s.pathway_id not in crossing_ids:
            continue
        for e in s.events:
            if e.target in score:
                score[e.target] += 1.0
    return score


def _compute_mc_element_frequencies(mc_df: pd.DataFrame) -> Dict[str, float]:
    m = mc_df[mc_df["moderate_passive_failure"]].copy()
    if len(m) == 0:
        return {"T2P": 0.0, "corridor": 0.0, "S1A": 0.0, "S1B": 0.0, "S1C": 0.0}
    out = {
        "T2P": float(m["has_T2P"].mean()),
        "corridor": float(m["has_corridor"].mean()),
        "S1A": float(m["has_S1A"].mean()),
        "S1B": float(m["has_S1B"].mean()),
        "S1C": float(m["has_S1C"].mean()),
    }
    return out


def _coverage(families: set, df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["primary_family"].isin(families).mean())


def _non_adaptation_limited_failures(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["primary_family"] != ADAPTATION_LIMITED_FAMILY]


def _coverage_full_denominator(families: set, df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    assessable = _non_adaptation_limited_failures(df)
    return float(assessable["primary_family"].isin(families).sum() / len(df))


def _coverage_family_subset(families: set, df: pd.DataFrame, subset: set) -> float:
    assessable = _non_adaptation_limited_failures(df)
    subset_df = assessable[assessable["primary_family"].isin(subset)]
    return _coverage(families, subset_df)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _crossing_family_count(results_df: pd.DataFrame, family: str) -> int:
    return int(((results_df["assigned_family"] == family) & (results_df["moderate_crossing"])).sum())


def generate_tables(
    output_dir: Path,
    replications: int,
    seed: int,
    budget: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = NetworkConfig()

    # Re-run study to ensure consistent base artifacts for this table build.
    summary = run_study(replications=replications, output_dir=output_dir, seed=seed, budget=budget)

    # Load core data
    mc = pd.read_csv(output_dir / "monte_carlo_ground_truth.csv")
    fst = pd.read_csv(output_dir / "fst_results.csv")
    rst = pd.read_csv(output_dir / "rst_results.csv")

    scenarios = fst_library()
    enhanced_scenarios = fst_enhanced_library()
    if len(enhanced_scenarios) != budget:
        raise ValueError(f"FST-Enhanced scenario count must equal budget {budget}; got {len(enhanced_scenarios)}")
    fst_enhanced = run_fst(enhanced_scenarios, cfg=cfg)
    fst_enhanced.to_csv(output_dir / "fst_enhanced_results.csv", index=False)

    # Table 4: FST scenario results summary
    t4 = fst.copy()
    t4["group"] = t4["category"].map(_fst_group)
    table4 = (
        t4.groupby("group", as_index=False)
        .agg(
            scenarios=("scenario_id", "count"),
            moderate_crossings=("moderate_crossing", "sum"),
            severe_crossings=("severe_crossing", "sum"),
        )
        .sort_values("group")
    )
    table4.to_csv(output_dir / "table4_fst_summary.csv", index=False)

    # Table 5: RST mechanism family by passive/active classification
    from paper_rst_ijpr import rst_candidate_library

    severe_evals, _ = mechanism_guided_rst_search(
        candidates=rst_candidate_library(), cfg=cfg, budget=budget, threshold=SEVERE
    )
    sev_df = pd.DataFrame(
        {
            "family": [e.family for e in severe_evals],
            "passive_failure": [e.passive_failure for e in severe_evals],
            "active_failure": [bool(e.active_failure) if e.active_failure is not None else False for e in severe_evals],
        }
    )

    rows = []
    for f in MECHANISM_FAMILIES:
        family_rows = rst[rst["family"] == f]
        passive_rows = family_rows[family_rows["passive_failure"] == True]
        structural_rows = passive_rows[passive_rows["active_failure"] == True]
        adaptation_rows = passive_rows[passive_rows["active_failure"] == False]
        rows.append(
            {
                "Mechanism family": f,
                "Passive failures": int(len(passive_rows)),
                "Structural after active RST": int(len(structural_rows)),
                "Adaptation-limited after active RST": int(len(adaptation_rows)),
                "Intervention implication": _family_intervention(f),
            }
        )
    table5 = pd.DataFrame(rows)
    table5.to_csv(output_dir / "table5_rst_family_counts.csv", index=False)

    # Table 6: Comparative metrics
    moderate_failures = mc[mc["moderate_passive_failure"] == True].copy()
    fst_families = set(fst.loc[fst["moderate_crossing"], "assigned_family"].tolist())
    rst_families = set(rst.loc[rst["passive_failure"], "family"].tolist())
    non_alf_failures = _non_adaptation_limited_failures(moderate_failures)
    rst_structural = int((rst["active_failure"] == True).sum())
    rst_adaptation_limited = int(((rst["passive_failure"] == True) & (rst["active_failure"] == False)).sum())
    table6 = pd.DataFrame(
        [
            {"Metric": "Budget", "FST": budget, "RST": budget},
            {"Metric": "Moderate threshold-crossing scenarios/pathways", "FST": int(fst["moderate_crossing"].sum()), "RST": int(rst["passive_failure"].sum())},
            {"Metric": "Severe threshold-crossing scenarios/pathways", "FST": int(fst["severe_crossing"].sum()), "RST": int(sev_df["passive_failure"].sum())},
            {"Metric": "Mechanism-family coverage, full denominator", "FST": _coverage_full_denominator(fst_families, moderate_failures), "RST": _coverage_full_denominator(rst_families, moderate_failures)},
            {"Metric": "Mechanism-family coverage, non-ALF denominator", "FST": _coverage(fst_families, non_alf_failures), "RST": _coverage(rst_families, non_alf_failures)},
            {"Metric": "CMD/PAC/RL-TO coverage", "FST": _coverage_family_subset(fst_families, moderate_failures, CMD_PAC_RLTO), "RST": _coverage_family_subset(rst_families, moderate_failures, CMD_PAC_RLTO)},
            {"Metric": "Passive-active classification", "FST": "Not evaluated", "RST": f"{rst_adaptation_limited} adaptation-limited; {rst_structural} structural"},
        ]
    )
    table6.to_csv(output_dir / "table6_fst_vs_rst.csv", index=False)

    enhanced_families = set(
        fst_enhanced.loc[fst_enhanced["moderate_crossing"], "assigned_family"].tolist()
    )
    fst_cmd_pac_rlto_coverage = _coverage_family_subset(fst_families, moderate_failures, CMD_PAC_RLTO)
    enhanced_cmd_pac_rlto_coverage = _coverage_family_subset(
        enhanced_families, moderate_failures, CMD_PAC_RLTO
    )
    rst_cmd_pac_rlto_coverage = _coverage_family_subset(rst_families, moderate_failures, CMD_PAC_RLTO)

    table6b = pd.DataFrame(
        [
            {
                "Metric": "Moderate crossings",
                "FST-original": int(fst["moderate_crossing"].sum()),
                "FST-enhanced-50": int(fst_enhanced["moderate_crossing"].sum()),
            },
            {
                "Metric": "Severe crossings",
                "FST-original": int(fst["severe_crossing"].sum()),
                "FST-enhanced-50": int(fst_enhanced["severe_crossing"].sum()),
            },
            {
                "Metric": "HCD pathways",
                "FST-original": _crossing_family_count(fst, "Hidden convergent dependency"),
                "FST-enhanced-50": _crossing_family_count(fst_enhanced, "Hidden convergent dependency"),
            },
            {
                "Metric": "CMD pathways",
                "FST-original": _crossing_family_count(fst, "Compounding moderate disturbance"),
                "FST-enhanced-50": _crossing_family_count(fst_enhanced, "Compounding moderate disturbance"),
            },
            {
                "Metric": "PAC pathways",
                "FST-original": _crossing_family_count(fst, "Propagation-amplified cascade"),
                "FST-enhanced-50": _crossing_family_count(fst_enhanced, "Propagation-amplified cascade"),
            },
            {
                "Metric": "RL-TO pathways",
                "FST-original": _crossing_family_count(fst, "Recovery lag and temporal overlap"),
                "FST-enhanced-50": _crossing_family_count(fst_enhanced, "Recovery lag and temporal overlap"),
            },
            {
                "Metric": "CD pathways",
                "FST-original": _crossing_family_count(fst, "Concentrated dependency"),
                "FST-enhanced-50": _crossing_family_count(fst_enhanced, "Concentrated dependency"),
            },
            {
                "Metric": "CMD/PAC/RL-TO coverage",
                "FST-original": _pct(fst_cmd_pac_rlto_coverage),
                "FST-enhanced-50": _pct(enhanced_cmd_pac_rlto_coverage),
            },
        ]
    )
    table6b.to_csv(output_dir / "table6b_fst_original_vs_enhanced.csv", index=False)

    table6_corrected = pd.DataFrame(
        [
            {"Metric": "Budget", "FST-original": budget, "FST-enhanced-50": budget, "RST": budget},
            {
                "Metric": "Moderate threshold crossings",
                "FST-original": int(fst["moderate_crossing"].sum()),
                "FST-enhanced-50": int(fst_enhanced["moderate_crossing"].sum()),
                "RST": int(rst["passive_failure"].sum()),
            },
            {
                "Metric": "Severe threshold crossings",
                "FST-original": int(fst["severe_crossing"].sum()),
                "FST-enhanced-50": int(fst_enhanced["severe_crossing"].sum()),
                "RST": int(sev_df["passive_failure"].sum()),
            },
            {
                "Metric": "Full-denominator coverage",
                "FST-original": _pct(_coverage_full_denominator(fst_families, moderate_failures)),
                "FST-enhanced-50": _pct(_coverage_full_denominator(enhanced_families, moderate_failures)),
                "RST": _pct(_coverage_full_denominator(rst_families, moderate_failures)),
            },
            {
                "Metric": "Non-ALF coverage",
                "FST-original": _pct(_coverage(fst_families, non_alf_failures)),
                "FST-enhanced-50": _pct(_coverage(enhanced_families, non_alf_failures)),
                "RST": _pct(_coverage(rst_families, non_alf_failures)),
            },
            {
                "Metric": "CMD/PAC/RL-TO coverage",
                "FST-original": _pct(fst_cmd_pac_rlto_coverage),
                "FST-enhanced-50": _pct(enhanced_cmd_pac_rlto_coverage),
                "RST": _pct(rst_cmd_pac_rlto_coverage),
            },
            {
                "Metric": "Passive-active classification",
                "FST-original": "N/A",
                "FST-enhanced-50": "N/A",
                "RST": f"{rst_adaptation_limited} ALF; {rst_structural} structural",
            },
        ]
    )
    table6_corrected.to_csv(output_dir / "table6_corrected_comparison.csv", index=False)

    table6c_rows = []
    for family in MECHANISM_FAMILIES:
        non_alf_family_count = int((non_alf_failures["primary_family"] == family).sum())
        full_denominator_count = int((moderate_failures["primary_family"] == family).sum())
        table6c_rows.append(
            {
                "Mechanism family": family,
                "MC failures, full denominator": full_denominator_count,
                "MC failures, non-ALF denominator": non_alf_family_count,
                "FST-original crossings": _crossing_family_count(fst, family),
                "FST-enhanced-50 crossings": _crossing_family_count(fst_enhanced, family),
                "RST passive pathways": int(((rst["family"] == family) & (rst["passive_failure"] == True)).sum()),
                "FST-original covered?": family in fst_families,
                "FST-enhanced-50 covered?": family in enhanced_families,
                "RST covered?": family in rst_families,
            }
        )
    table6c = pd.DataFrame(table6c_rows)
    table6c.to_csv(output_dir / "table6c_mechanism_coverage_corrected.csv", index=False)

    enhanced_extra_families = sorted(enhanced_families - rst_families)
    if enhanced_cmd_pac_rlto_coverage < rst_cmd_pac_rlto_coverage:
        enhanced_decision = "RST has stronger mechanism-search value."
    elif np.isclose(enhanced_cmd_pac_rlto_coverage, rst_cmd_pac_rlto_coverage) and enhanced_extra_families:
        enhanced_decision = (
            "RST's main value is passive-active classification; FST-enhanced also covers "
            f"{', '.join(enhanced_extra_families)}, so frame RST as complementary rather than coverage-superior."
        )
    elif np.isclose(enhanced_cmd_pac_rlto_coverage, rst_cmd_pac_rlto_coverage):
        enhanced_decision = "RST's main value is passive-active classification, not coverage."
    else:
        enhanced_decision = "RST is complementary; emphasize classification and intervention logic."

    scenario_class_counts = fst_enhanced["category"].value_counts().to_dict()
    corrected_summary = {
        "config": {
            "replications": replications,
            "seed": seed,
            "budget": budget,
        },
        "fst_enhanced_50": {
            "scenario_class_counts": scenario_class_counts,
            "moderate_crossings": int(fst_enhanced["moderate_crossing"].sum()),
            "severe_crossings": int(fst_enhanced["severe_crossing"].sum()),
            "discovered_mechanism_families": sorted(enhanced_families),
            "coverage_full_denominator": _coverage_full_denominator(enhanced_families, moderate_failures),
            "coverage_non_alf_denominator": _coverage(enhanced_families, non_alf_failures),
            "coverage_cmd_pac_rlto": enhanced_cmd_pac_rlto_coverage,
            "families_not_covered_by_rst": enhanced_extra_families,
        },
        "comparison": {
            "fst_original": {
                "coverage_full_denominator": _coverage_full_denominator(fst_families, moderate_failures),
                "coverage_non_alf_denominator": _coverage(fst_families, non_alf_failures),
                "coverage_cmd_pac_rlto": fst_cmd_pac_rlto_coverage,
            },
            "rst": {
                "coverage_full_denominator": _coverage_full_denominator(rst_families, moderate_failures),
                "coverage_non_alf_denominator": _coverage(rst_families, non_alf_failures),
                "coverage_cmd_pac_rlto": rst_cmd_pac_rlto_coverage,
                "adaptation_limited_pathways": rst_adaptation_limited,
                "structural_pathways": rst_structural,
            },
        },
        "decision_rule_result": enhanced_decision,
    }
    with (output_dir / "summary_corrected.json").open("w", encoding="utf-8") as f:
        json.dump(corrected_summary, f, indent=2)

    # Table 7: Failure-conditioned criticality comparison
    rst_scores = summary["rst"]["criticality_scores"]
    rst_rank = _rank_from_scores(rst_scores)

    fst_scores = _compute_fst_element_scores(fst, scenarios)
    fst_rank = _rank_from_scores(fst_scores)

    mc_freq = _compute_mc_element_frequencies(mc)
    mc_rank = _rank_from_scores(mc_freq)

    elements = ["T2P", "corridor", "S1A", "S1B", "S1C"]
    table7_rows = []
    for e in elements:
        table7_rows.append(
            {
                "element": e,
                "RST_IF": float(rst_scores.get(e, 0.0)),
                "RST_rank": int(rst_rank.get(e, 0)),
                "FST_rank": int(fst_rank.get(e, 0)),
                "MC_frequency": float(mc_freq.get(e, 0.0)),
                "MC_rank": int(mc_rank.get(e, 0)),
            }
        )
    table7 = pd.DataFrame(table7_rows)

    # Kendall tau_b
    rst_vec = [rst_scores[e] for e in elements]
    fst_vec = [fst_scores[e] for e in elements]
    mc_vec = [mc_freq[e] for e in elements]
    tau_rst, p_rst = kendalltau(rst_vec, mc_vec)
    tau_fst, p_fst = kendalltau(fst_vec, mc_vec)
    table7.to_csv(output_dir / "table7_criticality_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"comparison": "RST vs MC", "kendall_tau_b": float(tau_rst), "p_value": float(p_rst)},
            {"comparison": "FST vs MC", "kendall_tau_b": float(tau_fst), "p_value": float(p_fst)},
        ]
    ).to_csv(output_dir / "table7_kendall_tau.csv", index=False)

    # Table 8: Threshold sensitivity
    thresholds = [
        FailureThreshold("SL<90% >=1 period", 0.90, 1),
        FailureThreshold("SL<90% >=2 periods", 0.90, 2),
        FailureThreshold("SL<85% >=3 periods", 0.85, 3),
        FailureThreshold("SL<80% >=4 periods", 0.80, 4),
    ]
    t8_rows = []
    from paper_rst_ijpr import rst_candidate_library

    for th in thresholds:
        evals, scores = mechanism_guided_rst_search(rst_candidate_library(), cfg=cfg, budget=budget, threshold=th)
        pcount = sum(1 for e in evals if e.passive_failure)
        fam_counts = pd.Series([e.family for e in evals if e.passive_failure]).value_counts()
        dom = fam_counts.index[0] if len(fam_counts) else "None"
        t8_rows.append(
            {
                "threshold": th.name,
                "passive_pathways": int(pcount),
                "T2P_IF": float(scores.get("T2P", 0.0)),
                "dominant_mechanism": dom,
                "intervention_implication": _family_intervention(dom),
            }
        )
    pd.DataFrame(t8_rows).to_csv(output_dir / "table8_threshold_sensitivity.csv", index=False)

    # Table 9: Passive vs active pathway-level diagnosis (moderate)
    rst_mod = rst[rst["passive_failure"] == True].copy().sort_values("weight", ascending=False)
    t9 = rst_mod.copy()
    t9["passive_status"] = "Failure"
    t9["active_status"] = np.where(t9["active_failure"] == True, "Failure", "Safe")
    t9["classification"] = np.where(t9["active_failure"] == True, STRUCTURAL, ADAPTATION_LIMITED)
    t9["intervention"] = t9["family"].map(_family_intervention)
    t9 = t9[["pathway_id", "family", "passive_status", "active_status", "classification", "weight", "intervention"]]
    t9.to_csv(output_dir / "table9_passive_active_pathways.csv", index=False)

    # Table 10: Mechanism families and passive-active classification summary
    fam_mc = non_alf_failures["primary_family"].value_counts(normalize=True)
    fam_rst_pass = rst[rst["passive_failure"] == True]["family"].value_counts()
    fam_rst_act = rst[rst["active_failure"] == True]["family"].value_counts()
    fam_rst_adaptation = rst[(rst["passive_failure"] == True) & (rst["active_failure"] == False)]["family"].value_counts()
    t10_rows = []
    for f in MECHANISM_FAMILIES:
        t10_rows.append(
            {
                "mechanism_family": f,
                "mc_primary_share_non_alf_moderate": float(fam_mc.get(f, 0.0)),
                "rst_moderate_passive_count": int(fam_rst_pass.get(f, 0)),
                "rst_structural_count": int(fam_rst_act.get(f, 0)),
                "rst_adaptation_limited_count": int(fam_rst_adaptation.get(f, 0)),
                "intervention_logic": _family_intervention(f),
            }
        )
    pd.DataFrame(t10_rows).to_csv(output_dir / "table10_mechanism_summary.csv", index=False)

    # Appendix A: full FST scenario list
    a_rows = []
    for s in scenarios:
        row = fst.loc[fst["scenario_id"] == s.pathway_id].iloc[0]
        a_rows.append(
            {
                "scenario_id": s.pathway_id,
                "category": s.family,
                "specification": _event_spec(s),
                "moderate_crossing": bool(row["moderate_crossing"]),
                "severe_crossing": bool(row["severe_crossing"]),
            }
        )
    pd.DataFrame(a_rows).to_csv(output_dir / "appendix_a_fst_full_list.csv", index=False)

    # Appendix B: full RST candidate list
    b = rst.copy()
    b["classification"] = np.where(
        b["passive_failure"] == True,
        np.where(b["active_failure"] == True, "Structural", "Adaptation-limited"),
        "Non-failure",
    )
    b.to_csv(output_dir / "appendix_b_rst_full_list.csv", index=False)

    # Appendix C: metadata
    import platform
    import scipy

    c_rows = [
        {"item": "Programming language", "value": platform.python_version()},
        {"item": "NumPy", "value": np.__version__},
        {"item": "SciPy", "value": scipy.__version__},
        {"item": "pandas", "value": pd.__version__},
        {"item": "Monte Carlo replications", "value": replications},
        {"item": "Random seed", "value": seed},
        {"item": "Budget", "value": budget},
        {"item": "Horizon", "value": cfg.horizon_weeks},
        {"item": "mc_max_events", "value": cfg.mc_max_events},
        {"item": "mc_interarrival_mean_weeks", "value": cfg.mc_interarrival_mean_weeks},
    ]
    pd.DataFrame(c_rows).to_csv(output_dir / "appendix_c_metadata.csv", index=False)

    # One consolidated workbook (optional; CSV outputs are always generated).
    workbook_written = False
    try:
        import openpyxl  # noqa: F401

        with pd.ExcelWriter(output_dir / "paper_tables.xlsx", engine="openpyxl") as xw:
            table4.to_excel(xw, sheet_name="Table4_FST", index=False)
            table5.to_excel(xw, sheet_name="Table5_RST_Family", index=False)
            table6.to_excel(xw, sheet_name="Table6_Compare", index=False)
            table7.to_excel(xw, sheet_name="Table7_Criticality", index=False)
            pd.read_csv(output_dir / "table7_kendall_tau.csv").to_excel(xw, sheet_name="Table7_Tau", index=False)
            pd.read_csv(output_dir / "table8_threshold_sensitivity.csv").to_excel(xw, sheet_name="Table8_Threshold", index=False)
            pd.read_csv(output_dir / "table9_passive_active_pathways.csv").to_excel(xw, sheet_name="Table9_PassiveActive", index=False)
            pd.read_csv(output_dir / "table10_mechanism_summary.csv").to_excel(xw, sheet_name="Table10_Mechanisms", index=False)
            pd.read_csv(output_dir / "appendix_a_fst_full_list.csv").to_excel(xw, sheet_name="AppendixA_FST", index=False)
            pd.read_csv(output_dir / "appendix_b_rst_full_list.csv").to_excel(xw, sheet_name="AppendixB_RST", index=False)
            pd.read_csv(output_dir / "appendix_c_metadata.csv").to_excel(xw, sheet_name="AppendixC_Meta", index=False)
        workbook_written = True
    except Exception:
        workbook_written = False

    with (output_dir / "tables_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(output_dir.resolve()),
                "replications": replications,
                "seed": seed,
                "budget": budget,
                "workbook_written": workbook_written,
                "files": sorted([p.name for p in output_dir.glob("*") if p.is_file()]),
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style tables from RST outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_tables"))
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()

    generate_tables(
        output_dir=args.output_dir,
        replications=args.replications,
        seed=args.seed,
        budget=args.budget,
    )
    print(f"Tables generated in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
