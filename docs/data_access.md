# Data Access and Study Cohorts

## Dataset Providers

| Dataset | Provider entry | Study use | Collection identifier |
| --- | --- | --- | --- |
| BraTS2020 | [BraTS2020 data](https://www.med.upenn.edu/cbica/brats2020/data.html) | 218 scans from the TCIA/TCGA portion | BraTS2020; TCGA annotations |
| IvyGAP | [TCIA IvyGAP](https://www.cancerimagingarchive.net/collection/ivygap/) | 34 internal scans | [10.7937/K9/TCIA.2016.XLwaN6nL](https://doi.org/10.7937/K9/TCIA.2016.XLwaN6nL) |
| UPENN-GBM | [TCIA UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | 565 internal scans | [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49) |
| UCSF-PDGM | [TCIA UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) | 501 external scans; Version 3 structural images | [10.7937/TCIA.BDGF-8V37](https://doi.org/10.7937/TCIA.BDGF-8V37) |

Follow each provider's instructions for image access, clinical annotations, reuse, and citation. BraTS2020 describes registration and a data usage agreement. Its TCIA/TCGA mapping supports linking the selected images to molecular annotations. [MTTU-Net](https://github.com/miacsu/MTTU-Net) provides an additional IDH-related label resource.

[IvyGAP-Radiomics](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70222827) is a related analysis collection containing processed images, segmentations, and radiomic resources. Its collection identity differs from the original IvyGAP collection; select study inputs according to the CMIG Supplementary protocol.

## CMIG Study Partitions

The internal cohort contains 817 scans: 576 training, 78 validation, and 163 internal test scans. Each scan record belongs to one subset. Repeated examinations from some UPENN-GBM patients appear across subsets, so the internal partition and performance are described at the scan level.

The UCSF-PDGM external cohort contains 501 scans from 495 patient groups, including six follow-up examinations. The Version 3 package was updated on January 11, 2023. Its study designation is supported by the source directory and matching scan identifiers; the original download receipt and a bytewise comparison against later image versions were not retained. See [UCSF version provenance](../metadata/ucsf_version_provenance.json).

The MRI inputs are T1, T1CE, T2, and FLAIR, stacked in that order. The study uses a 1 mm isotropic grid of 240 × 240 × 155 voxels. Preprocessing, crops, demographic inputs, and checkpoint selection are described in the CMIG manuscript and its Supplementary material.

## Distributed Analysis Inputs

[reproducibility/](../reproducibility/README.md) contains pseudonymous study partitions and patient-group mappings, 28 model/cohort prediction tables, statistical-analysis code, recorded summary tables, and 420 timing measurements. The analysis uses the released molecular IDH annotations as reference labels.

MRI volumes, original clinical tables, and model weights are obtained separately. The original dataset identifiers are replaced with study pseudonyms in the distributed analysis manifests and predictions. Historical model source may retain provider-issued example filenames and local filesystem paths.

The five evaluation/timing tools and two additional 3DINO configuration files are supplied in the CMIG manuscript Supplementary package. The manuscript Supplementary also provides the full text-input and encoding protocol. Their release locations are described in [Installation](installation.md).
