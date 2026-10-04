# Metadata

This directory identifies the CMIG evaluated source, selected checkpoint, dataset version, and historical environment records.

| Path | Purpose |
| --- | --- |
| `code_provenance.json` | Recorded evaluation and training-metadata identities |
| `selected_checkpoint.sha256` | Evaluated study checkpoint identifier |
| `ucsf_version_provenance.json` | UCSF-PDGM Version 3 package evidence |
| `reproduction_validation.json` | Historical statistical validation |
| `VERSION_VERIFICATION.md` | Historical source and current documentation relationship |
| `EXCLUDED_ASSETS.md` | Distributed assets and separately obtained resources |
| `environment_snapshots/` | Original dependency exports and normalization record |

The JSON provenance records retain their historical bytes and source context. Current user instructions are in [Installation](../docs/installation.md), [Data access](../docs/data_access.md), and [Statistical reproducibility](../reproducibility/README.md). The canonical [manifest schema](../reproducibility/statistics_manifest_schema.md) describes the distributed statistical inputs.
