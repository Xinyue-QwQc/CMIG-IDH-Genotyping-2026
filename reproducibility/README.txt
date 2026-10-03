Supplementary material for Clinical Knowledge Guided Foundation Model Customization on Multimodal MRI for Glioma IDH Genotyping

Data files and captions
- data_partition_manifest.json — ordered train/validation/internal-test assignments (817 internal scans) and external evaluation (501 scans); original case and patient IDs replaced with study pseudonyms. The internal split is scan-level and retains cross-partition follow-up links.
- main_results.csv — 28 model/cohort metric rows; estimates and 95% patient-cluster bootstrap intervals are fractions from 0 to 1.
- paired_comparisons.csv — 52 fixed-model AUC/ACC paired comparisons; differences and CIs are in fractional and percentage-point units; adjusted p values use one Holm family of 52.
- timing_results.csv — model-specific summary of 30 measurement values per model under the fixed single-GPU protocol.
- selected_model_subgroups.csv — selected-model per-source and patient-overlap subgroup summaries (descriptive only).
- selected_model_error_profile.csv — selected-model true/false positive/negative group counts, age and sex summaries; no causal claims.
- SHA256SUMS.csv — size and SHA-256 for the deliverable files in this folder.
- case_mapping.json, selected_predictions.json, predictions/, main_statistics.py, and timing_measurements_anonymized.csv — pseudonymous analysis inputs and code in this directory. The evaluated source ZIP and five tools are supplied separately in the manuscript Supplementary package at Supplementary/Reproducibility/evaluation_code_snapshot.zip.

Statistical recomputation
From this reproducibility/ folder, run:
  python main_statistics.py --manifest selected_predictions.json --out reproduced_results
Python 3 and NumPy are required. A short program check adds --smoke --bootstrap 2; formal analysis uses the default 20,000 replicates. This corrected statistical distribution retains the historical evaluation code and prediction values and is used with the manuscript Supplementary package.

Data providers and persistent source collection identifiers
- BraTS 2020 access: https://www.med.upenn.edu/cbica/brats2020/data.html (TCIA/TCGA portion selected for this study).
- UPENN-GBM, DOI 10.7937/TCIA.709X-DN49: https://www.cancerimagingarchive.net/collection/upenn-gbm/
- IvyGAP, DOI 10.7937/K9/TCIA.2016.XLwaN6nL: https://www.cancerimagingarchive.net/collection/ivygap/
- UCSF-PDGM Version 3 image package (updated 2023-01-11), DOI 10.7937/TCIA.BDGF-8V37: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/ ; provenance counts are in ucsf_version_provenance.json.

Limits and interpretation
The research outputs in this folder contain only evaluation predictions, derived summaries and analysis source. MRI images, clinical source tables and evaluated/training checkpoints remain with their providers or the study team. The 501 study scans match the on-server Version 3 image package, including six follow-up examinations renamed in the Version 3 update. The present TCIA collection page cites Version 5; subsequent updates described there affect metadata/access tables and diffusion files. The original download receipt is unavailable, so the package designation rests on the source directory and its matching scan identifiers. Public use of a repository separate from the manuscript supplementary files requires institutional review of source-data and weight-sharing terms.
