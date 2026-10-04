# Version History

## Repository Documentation

The October 3, 2026 documentation update organizes the repository around the CMIG manuscript. It adds dataset and pretrained-model entry points, directory guides, citation metadata, portable dependency references, and statistical instructions. All 182 Python files under `source/`, the statistical script, predictions, manifests, and recorded scientific results retain their preceding release bytes.

The two historical dependency exports remain available under `metadata/environment_snapshots/`. The three revised files in `source/` are its README and two dependency references.

## Statistical Distribution

[8b06ce7](https://github.com/Xinyue-QwQc/CMIG-IDH-Genotyping-2026/commit/8b06ce7292b0399beb9726e6b7a5ce120614c35b) is the October 2 distribution repair. It restores the 28 prediction CSV files to the Supplementary bytes so that the prediction-manifest hashes validate. It preserves scientific values, methods, and model implementation. Use this release or a later documentation revision for statistical recomputation.

## Historical Evaluated Source

[8a5fff4](https://github.com/Xinyue-QwQc/CMIG-IDH-Genotyping-2026/commit/8a5fff4e5cff07829aa31fab766a241f0e1bd3ba) is the initial CMIG public-source release referenced by the Supplementary. Its 185-file source tree represents the historical project snapshot. The later distribution repair provides the compatible statistical inputs.

The CMIG evaluated source revision is `ad87f2f8a9530c9ccd39d37d1391e027bbc48d67`. The Supplementary evaluation archive has SHA-256 `baa4256228d0a06f871097ad7a14750f3aebc281c7e352af676749e77ad1230a` and includes five additional tools. Source, training-metadata, and checkpoint identifiers have separate provenance roles; see [Version verification](../metadata/VERSION_VERIFICATION.md).

## Earlier Implementation

[cyxuccc/PaperCode](https://github.com/cyxuccc/PaperCode) contains an earlier implementation. Its observed revision is `85c692eebfac95fb440704e213ce3cfb3bc20e08`. The CMIG evaluated implementation and analysis inputs are distributed through this repository and the CMIG Supplementary package.
