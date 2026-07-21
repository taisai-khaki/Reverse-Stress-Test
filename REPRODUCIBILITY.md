# Reviewer Reproducibility Guide

This package contains the submitted-result audit, the revised Sections 3 and 4
computational model, fixed-seed Monte Carlo outputs, deterministic FST/RST
outputs, sensitivity checks, topology robustness results, and regression tests.

## Environment

- Python 3.14.3 was used for the archived run.
- Exact Python package versions are pinned in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Validation Gate

```powershell
python -m unittest discover -s tests -v
```

The package currently contains 14 regression tests covering the system operator,
pipeline and inventory conservation, response bounds, mechanism attribution,
adaptation-limited taxonomy, severe RST count, response-sensitivity settings,
and exact multilabel-signature matching.

## Full Revised Analysis

```powershell
python generate_revised_analysis.py `
  --output-dir 02_revised_analysis_outputs `
  --seeds 42 314 1618 2026 2718 `
  --replications-per-seed 1000 `
  --workers 1
```

The driver validates cached seed files by row count and resumes them by default.
Use `--no-resume` to regenerate every Monte Carlo replication from the fixed
seeds.

## Key Archived Results

- Monte Carlo moderate failures: 2,455 / 5,000.
- Monte Carlo severe failures: 1,445 / 5,000.
- Baseline FST crossings: 33 moderate and 11 severe.
- Mechanism-balanced FST crossings: 46 moderate and 35 severe.
- RST pathways: 48 passive, 39 severe, 41 structural, and 7 adaptation-limited.
- Exact multilabel-signature coverage, full denominator:
  - Baseline FST: 79.3%.
  - Mechanism-balanced FST: 85.3%.
  - RST: 90.2%.

Exact signature coverage requires equality of the complete mechanism-label set.
Sharing only one label does not count as an exact match.

## Output Map

- `00_reproduce_submitted_results/`: audit of the originally submitted results.
- `01_revised_simulation_outputs/`: earlier revised Tables A-F retained for traceability.
- `02_revised_analysis_outputs/`: final revised analysis, raw fixed-seed runs, and review workbook.
- `outputs_tables/`: corrected legacy comparison tables and appendices.
- `outputs_r26_robustness/`: earlier reviewer R2-6 robustness package retained for traceability.

The authoritative revised summary is
`02_revised_analysis_outputs/analysis_summary.json`. The workbook
`02_revised_analysis_outputs/revised_analysis_review.xlsx` provides a formatted
review of the same CSV and JSON outputs.
