# Statistical Reproducibility

This directory contains pseudonymous inputs and recorded results for the fixed-model analysis in the CMIG manuscript.

## Inputs and Recorded Outputs

| Path | Contents |
| --- | --- |
| `case_mapping.json` | 1,318-scan mapping with cohort, source, class, split, order, and patient-group identifiers |
| `data_partition_manifest.json` | Internal and external study assignments |
| `selected_predictions.json` | Selected prediction paths, file hashes, and analysis settings |
| `predictions/` | 28 prediction CSV files for 14 models in two test cohorts |
| `main_statistics.py` | Patient-cluster bootstrap and paired comparisons |
| `main_results.csv`, `metrics_long.csv` | Recorded metric estimates and confidence intervals |
| `paired_comparisons.csv` | 52 recorded ACC/AUC comparisons |
| `selected_model_subgroups.csv`, `selected_model_error_profile.csv` | Descriptive subgroup and error summaries |
| `timing_measurements_anonymized.csv`, `timing_results.csv` | 420 measurements and their summaries |
| `reproduction_validation.json` | Historical formal-analysis numerical validation |
| `code_provenance.json`, `ucsf_version_provenance.json` | Source and dataset provenance records |
| `SHA256SUMS.csv` | File sizes and SHA-256 values, relative to this directory |

## Installation

From the repository root:

```bash
python -m pip install -r requirements-statistics.txt
```

The statistical dependency file pins NumPy 1.26.4. See [Installation](../docs/installation.md).

## Formal Analysis

From the repository root:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out reproduced_results
```

The same command from this directory is:

```bash
python main_statistics.py --manifest selected_predictions.json --out ../reproduced_results
```

Defaults are 20,000 class-stratified patient-cluster bootstrap replicates, seed `20260920`, and one Holm family of 52 comparisons. Expected outputs include 28 main-table rows and 52 paired-comparison rows. Values are fractions from 0 to 1; paired differences also include percentage-point fields.

## Program Check

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out smoke_results --smoke --bootstrap 100
```

Run this command from the repository root. Its reduced-resample confidence intervals and p values are program-check outputs; use the formal command for reported statistical results.

The script checks prediction hashes, case sets, labels, patient groups, scores, and the `score > 0.5` classification rule. [Manifest schema](statistics_manifest_schema.md) documents these inputs and the generated outputs.

## Scope

The analysis recomputes fixed-model statistics from distributed predictions. MRI training and prediction regeneration use separate weights, images, local resources, and evaluation tools described in [Installation](../docs/installation.md) and the CMIG Supplementary material.

Data-provider entry points and study counts are in [Data access](../docs/data_access.md). Version identities and the repaired statistical distribution are in [Version history](../docs/version_history.md). Scientific JSON and CSV files retain the preceding release bytes. The directory checksum list covers its current distribution, including revised documentation, and excludes itself.
