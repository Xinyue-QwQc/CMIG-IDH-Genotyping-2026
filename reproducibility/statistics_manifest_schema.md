# Statistical Manifest Schema

The distributed `selected_predictions.json` provides the complete 14-model, two-cohort analysis. Use that file for the CMIG reproduction commands. The following fragment illustrates fields using actual distributed paths; it is an excerpt rather than a complete formal manifest.

```json
{
  "schema_version": 1,
  "mapping_path": "case_mapping.json",
  "seed": 20260920,
  "bootstrap": 20000,
  "ours_model": "Ours",
  "planned_models": [
    {
      "model": "Ours",
      "display_name": "Ours (FinalVersion 2026-06-01)"
    }
  ],
  "cohorts": [
    {
      "id": "Internal",
      "mapping_dataset": "Internal",
      "mapping_split": "Test"
    },
    {
      "id": "External",
      "mapping_dataset": "External",
      "mapping_split": "Test"
    }
  ],
  "predictions": [
    {
      "id": "Ours_Internal",
      "model": "Ours",
      "cohort": "Internal",
      "status": "selected",
      "path": "predictions/Ours_Internal.csv",
      "sha256": "08348b3daa11b0fc11d45484c18e021b598b20be3cb2c52e2d38494b39454aaa"
    }
  ]
}
```

## Manifest Fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Input schema version; currently 1 |
| `mapping_path` | Case mapping path, resolved relative to the manifest file |
| `seed`, `bootstrap` | Formal defaults: 20260920 and 20000 |
| `ours_model` | Model used as the reference in paired comparisons |
| `planned_models` | Unique model identifiers and display names |
| `cohorts` | Cohort identifiers and mapping dataset/split selectors |
| `predictions` | One entry per model/cohort, with a unique prediction ID |

Each selected prediction entry supplies `id`, `model`, `cohort`, `status`, `path`, and `sha256`. Paths may be absolute or relative to the manifest directory. The current release uses paths within this repository. Other provenance fields are retained in `manifest_used.json`.

## Case Mapping and Predictions

`case_mapping.json` contains a `cases` dictionary. Each study record has its source, class label, cohort, split, order, and `patient_id` or `patient_group`. These identifiers are study pseudonyms. The script determines the complete expected case set for each test cohort from that mapping and resamples patient groups jointly across their scans.

The default prediction columns are `ID`, `idh_truth`, `pred`, and `pred_class`. IDH-mutant is class 1. Scores must be finite values in [0, 1]; the saved class equals `pred > 0.5`, with a score of exactly 0.5 assigned to wild-type. Per-file `columns` or `score_column` settings support alternative column names. Optional source and patient-group columns are checked against the mapping.

The script checks file SHA-256, complete case sets, duplicate IDs, class labels, patient-group consistency, and the saved classification rule. Selected but invalid files are reported in `input_validation.csv`. Entries outside the selected status retain their availability reason in the output.

## Commands

From the repository root:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out reproduced_results
```

For a reduced-resample program check:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out smoke_results --smoke --bootstrap 100
```

Formal analysis requires 14 models, two cohorts, 20,000 replicates, seed 20260920, and 52 planned comparisons. Program-check results use a separate output directory and describe execution with reduced resampling.

## Outputs

| Output | Contents |
| --- | --- |
| `metrics_long.csv` | Nine metrics per model/cohort, confidence intervals, valid replicate counts, and warnings |
| `main_results.csv`, `main_table_*.md` | Wide result tables and readable cohort summaries |
| `paired_comparisons.csv` | Ours-minus-baseline ACC/AUC differences, intervals, raw and Holm-adjusted p values |
| `input_validation.csv` | File hashes, case and patient counts, confusion matrices, and availability status |
| `patient_index_*.csv` | Ordered scan-to-patient-group mappings |
| `bootstrap_patient_weights_*.npz` | Resampling multiplicities for patient groups |
| `bootstrap_metrics.npz` | Bootstrap metric draws |
| `analysis_summary.json`, `manifest_used.json` | Protocol settings, environment information, validation summary, and input identities |

Proportions are on the 0–1 scale. Paired differences additionally include percentage-point fields. Undefined metrics and unavailable comparisons are reported explicitly. Holm correction retains one planned family of 52 tests.

The command returns exit code 2 for supplied prediction files that fail validation. A structurally invalid manifest raises an error. A manifest with missing predictions can produce partial output with exit code 0, so inspect validation status as well as the process exit code. The distributed CMIG manifest is complete.
