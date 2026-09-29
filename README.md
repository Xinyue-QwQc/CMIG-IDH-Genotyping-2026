# CMIG IDH Genotyping 2026

Code and pseudonymous fixed-model evaluation inputs for the manuscript **Clinical Knowledge Guided Foundation Model Customization on Multimodal MRI for Glioma IDH Genotyping**.

## Release contents

- `source/` contains the 185-file evaluation source snapshot used for the archived September 2026 analysis.
- `reproducibility/` contains pseudonymous manifests, predictions, recorded metrics, timing measurements, and the fixed-model bootstrap analysis script.
- `metadata/` contains provenance, version verification, schema, and exclusion notes.

## Version status

The source snapshot was downloaded from the shared research server and compared byte-for-byte with the evaluation archive bundled with the manuscript. The archive SHA-256 is `baa4256228d0a06f871097ad7a14750f3aebc281c7e352af676749e77ad1230a` and the archive contains 190 entries, including directories; 185 source files are included here.

The main evaluation used fixed predictions, 20,000 patient-cluster bootstrap replicates, seed `20260920`, and 52 paired comparisons with Holm correction. To reproduce the statistical tables from the pseudonymous inputs:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out reproduced_results
```

A shorter check is available with `--smoke --bootstrap 100`. It is a validation run and does not replace the reported 20,000-replicate analysis.

## Scope and access

This repository does not contain MRI volumes, original case identifiers, clinical source records, model checkpoints, or third-party pretrained weights. The source code retains machine-specific paths from the evaluated snapshot; adapt them to local assets before attempting training or inference. Dataset access remains subject to the terms of TCIA, BraTS, UPENN-GBM, IVY-GAP, and UCSF-PDGM.

The paper records the selected checkpoint hash and the exact source snapshot in `metadata/code_provenance.json`; the checkpoint itself is intentionally not distributed.

Release assembled on 2026-09-29 from the verified server snapshot.
