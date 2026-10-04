# Version Verification

## CMIG Evaluation Identifiers

| Identifier | Value |
| --- | --- |
| Evaluated source revision | `ad87f2f8a9530c9ccd39d37d1391e027bbc48d67` |
| Supplementary source archive SHA-256 | `baa4256228d0a06f871097ad7a14750f3aebc281c7e352af676749e77ad1230a` |
| Archive files | 190: 185 historical source files and five tools |
| Selected checkpoint SHA-256 | `45b256995362643db098fecc948612a7db2e3edd31e1762185dd94821016b6f4` |
| Training metadata revision | `6408f531d8998653d8e874395157626cf1ed5a87` |
| Initial CMIG public source | `8a5fff4e5cff07829aa31fab766a241f0e1bd3ba` |
| Statistical distribution repair | `8b06ce7292b0399beb9726e6b7a5ce120614c35b` |

The initial publication compared the 185 project files against the evaluated source archive. The source archive additionally contains five evaluation/timing tools supplied in the CMIG Supplementary. Its archive identity remains the original evaluated identifier.

## Documentation Release

The October 3 repository update preserves all 182 Python files under `source/`, `reproducibility/main_statistics.py`, prediction CSVs, input manifests, and recorded scientific results. `source/README.md` and the two dependency references are revised for repository use. Original dependency-file bytes are retained in `environment_snapshots/`.

The documentation update therefore changes three non-Python files in the 185-file source directory. Historical archive checksums refer to the archived evaluation package. The current statistical checksum list covers its distributed directory after documentation updates.

Source revision, training-metadata revision, checkpoint hash, and public Git commit identify different artifacts. Their roles are recorded separately in `code_provenance.json`. See [Version history](../docs/version_history.md) for the relationship between releases.
