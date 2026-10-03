# CMIG IDH Genotyping 2026

Code and pseudonymous fixed-model evaluation inputs for the manuscript **Clinical Knowledge Guided Foundation Model Customization on Multimodal MRI for Glioma IDH Genotyping**.

## Release contents

- `source/` contains the 185-file evaluation source snapshot used for the archived September 2026 analysis.
- `reproducibility/` contains pseudonymous manifests, predictions, recorded metrics, timing measurements, and the fixed-model bootstrap analysis script.
- `metadata/` contains provenance, version verification, schema, and exclusion notes.

## Version status

This distribution repairs the statistics entry point from historical commit `8a5fff4e5cff07829aa31fab766a241f0e1bd3ba`. The 185 `source/` files, `reproducibility/main_statistics.py`, and `reproducibility/selected_predictions.json` retain the historical commit bytes. Prediction values and statistical methods are unchanged. Use this repository together with the manuscript Supplementary package.

The evaluated `evaluation_code_snapshot.zip` is supplied in the manuscript at `Supplementary/Reproducibility/evaluation_code_snapshot.zip`, with SHA-256 `baa4256228d0a06f871097ad7a14750f3aebc281c7e352af676749e77ad1230a`. That Supplementary ZIP contains 190 files: 185 source files and five evaluation/timing tools. The five tools and the separately supplied 3DINO YAML configuration files are obtained from the Supplementary package; they are outside this local repository copy.

The 28 `reproducibility/predictions/*.csv` files retain the current Supplementary originals byte-for-byte, including CRLF line endings, and match the unchanged prediction manifest hashes. `.gitattributes` marks these paths `-text` to preserve those bytes. `reproducibility/SHA256SUMS.csv` lists actual files relative to `reproducibility/`, excludes itself, and uses LF.

## Statistical recomputation

Python 3 and NumPy are required. From this repository root, run:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out reproduced_results
```

The default formal analysis uses 20,000 patient-cluster bootstrap replicates, seed `20260920`, and 52 paired comparisons with Holm correction. A short entry check uses:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out smoke_results --smoke --bootstrap 2
```

The smoke output is a program check and does not replace the reported formal analysis.

## Scope and access

This repository does not contain MRI volumes, original case identifiers, clinical source records, model checkpoints, or third-party pretrained weights. The source code retains machine-specific paths from the evaluated snapshot; local assets and path/resource preparation are required for training or MRI inference. Dataset access remains subject to the terms of TCIA, BraTS, UPENN-GBM, IVY-GAP, and UCSF-PDGM. The available inputs directly support fixed-prediction statistical recomputation; MRI prediction regeneration requires the assets and permissions described in the Supplementary package.

The paper records the selected checkpoint hash and the evaluated source snapshot in `metadata/code_provenance.json`; the checkpoint itself is outside this copy. Historical provenance documents are retained from the base commit.

The historical release was assembled on 2026-09-29. Distribution integrity and entry-point documentation were corrected on 2026-10-02.
