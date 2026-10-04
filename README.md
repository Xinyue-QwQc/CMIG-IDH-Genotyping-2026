# Clinical Knowledge Guided Foundation Model Customization on Multimodal MRI for Glioma IDH Genotyping

This repository provides the code and statistical analysis inputs for the CMIG manuscript **Clinical Knowledge Guided Foundation Model Customization on Multimodal MRI for Glioma IDH Genotyping**.

## Data Access

The study uses four publicly accessible MRI datasets. Obtain images and clinical annotations from the dataset providers under their access and citation terms.

1. [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html). The study uses the TCIA/TCGA portion with available IDH, age, and sex annotations. The [BraTS2020 mirror](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) and [MTTU-Net](https://github.com/miacsu/MTTU-Net) provide additional imaging and label resources.
2. [IvyGAP](https://www.cancerimagingarchive.net/collection/ivygap/). Related processed images and annotations are described in [IvyGAP-Radiomics](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70222827).
3. [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/).
4. [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/). The CMIG evaluation uses the Version 3 structural-image package, comprising 501 scans from 495 patient groups.

Dataset versions, study subsets, and distributed analysis inputs are described in [Data access and study cohorts](docs/data_access.md). MRI images and original clinical tables are obtained from their providers; this repository distributes study pseudonyms, predictions, and derived summaries.

## Prerequisites

### Python environment preparation

For statistical recomputation, prepare a Python environment and install NumPy:

```bash
conda create -n cmig-idh python=3.9
conda activate cmig-idh
pip install -r requirements-statistics.txt
```

For MRI model development, `requirements.txt` points to the normalized model dependency reference in `source/requirements.txt`. Environment preparation, separate text dependencies, and required local resources are explained in [Installation](docs/installation.md).

### Checkpoint preparation for the foundation models

The [SAM-Med3D repository](https://github.com/uni-medical/SAM-Med3D) provides the visual foundation-model resources. The [BioLinkBERT-large repository](https://huggingface.co/michiyasunaga/BioLinkBERT-large) provides the text encoder and tokenizer.

The evaluated CMIG model also uses a study-specific checkpoint and a fixed description embedding. Their identifiers and the evaluation protocol are documented in the manuscript Supplementary material. [Installation](docs/installation.md) lists the resources needed to regenerate MRI predictions.

## Statistical Reproduction

From the repository root, run:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out reproduced_results
```

The formal analysis uses 20,000 patient-cluster bootstrap replicates, seed `20260920`, and 52 paired comparisons with Holm correction. The inputs cover 14 models and two test cohorts. A shorter program check uses:

```bash
python reproducibility/main_statistics.py --manifest reproducibility/selected_predictions.json --out smoke_results --smoke --bootstrap 100
```

See [Reproducibility](reproducibility/README.md) for inputs, outputs, and interpretation.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── NOTICE.md
├── requirements.txt
├── requirements-statistics.txt
├── docs/                 # Dataset, installation, and version guides
├── source/               # MRI model, training, and development code
├── reproducibility/      # Predictions, statistical analysis, and recorded results
└── metadata/             # Source identities and original environment records
```

See the [source guide](source/README.md) and [metadata guide](metadata/README.md) for directory details. Historical source and distribution versions are documented in [Version history](docs/version_history.md). Citation information is provided in [CITATION.cff](CITATION.cff); reuse terms are described in [NOTICE.md](NOTICE.md).

## Acknowledgement

We thank the investigators and data providers responsible for BraTS2020, TCIA/TCGA, IvyGAP, UPENN-GBM, and UCSF-PDGM. We acknowledge the support of the High-Performance Computing Center of Central South University.

We also thank the authors of the following projects:

- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)
- [BioLinkBERT](https://huggingface.co/michiyasunaga/BioLinkBERT-large)
- [MTTU-Net](https://github.com/miacsu/MTTU-Net)
