# CMIG Model Source

This directory contains the MRI model, training, data preparation, and development implementation accompanying the CMIG manuscript.

| Path | Contents |
| --- | --- |
| `models/` | Foundation-model components, task-specific modules, text encoders, and comparison models |
| `data/` | Data preparation and loading |
| `utils/` | Arguments, losses, and utilities |
| `dinov2/` | 3DINO-related implementation |
| `train_net*.py`, `test_net*.py` | Historical training and evaluation entry points |
| `requirements.txt`, `text_requirements.txt` | Normalized references for two separate environments |

See [Installation](../docs/installation.md) for resources and dependency preparation, [Data access](../docs/data_access.md) for providers, and [Version verification](../metadata/VERSION_VERIFICATION.md) for source identity. The runnable fixed-prediction statistical entry point is in [reproducibility/](../reproducibility/README.md).

The historical Python implementation retains its local path assumptions. Preparing MRI training or inference requires path configuration and study assets described in the CMIG Supplementary material. The five fixed-model evaluation/timing tools are supplied in its source archive.
