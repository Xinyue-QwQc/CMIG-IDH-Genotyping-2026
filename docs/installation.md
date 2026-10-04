# Installation

## Statistical Analysis

Python 3.9 or later and NumPy are sufficient for the statistical entry point. The dependency file pins NumPy 1.26.4, the version recorded in the model environment reference.

```bash
python -m venv .venv
```

Activate the environment for your shell:

```bash
# Linux and macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install the statistical dependencies from the repository root:

```bash
python -m pip install -r requirements-statistics.txt
```

Run the commands in [Reproducibility](../reproducibility/README.md).

## MRI Model Environment

The archived model environment records Python 3.9, PyTorch 2.2.1, torchvision 0.17.1, MONAI 1.3.0, and Transformers 4.38.2. From a suitable Linux/CUDA environment, the normalized dependency reference is installed with:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` includes `source/requirements.txt`. The normalized model and text references replace 67 server-local file URLs with package names. Five package versions are recoverable from recorded wheel filenames; the remaining 62 references have unrecorded exact versions and therefore remain unpinned. These files describe environment preparation rather than a complete historical environment lock. The GPU model environment has not been newly installed or evaluated for this documentation release.

Original environment exports are preserved byte-for-byte in [environment snapshots](../metadata/environment_snapshots/README.md). [normalization.json](../metadata/environment_snapshots/normalization.json) records each replacement and its version basis.

`source/text_requirements.txt` records a separate environment using PyTorch 2.4.0 with CUDA 11.8 and Transformers 5.4.0. Use it as a separate environment reference. The primary CMIG text encoder is BioLinkBERT-large under the model environment above. The two dependency exports describe different environments.

## Pretrained Resources and Study Assets

| Resource | Location or preparation |
| --- | --- |
| SAM-Med3D resources | [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) |
| BioLinkBERT-large and tokenizer | [Hugging Face](https://huggingface.co/michiyasunaga/BioLinkBERT-large) |
| Evaluated study checkpoint | Identified by `metadata/selected_checkpoint.sha256`; obtained separately |
| Fixed description embedding | Identified in the CMIG Supplementary `prompt_encoding.json`; obtained with the study assets |
| MRI volumes and clinical annotations | [Dataset providers](data_access.md) |
| Five evaluation/timing tools | CMIG Supplementary `Reproducibility/evaluation_code_snapshot.zip` |
| Two 3DINO configurations | CMIG Supplementary `Reproducibility/additional_configuration/` |

The source archive in the Supplementary has 190 files: 185 historical project files and five evaluation/timing tools. The tools are `inference_models.py`, `inference_baselines.py`, `inference_runner.py`, `timing_models.py`, and `timing_runner.py`. Place the separately supplied 3DINO YAML files at their matching `dinov2/configs` paths after extracting that archive.

Training and MRI inference use imports relative to `source/` and require local dataset paths, pretrained assets, and study weights. Configure those paths for the intended environment and follow the CMIG evaluation protocol. This release documents the evaluated implementation and statistical reproduction; it provides a directly runnable entry point for statistics from the distributed predictions.
