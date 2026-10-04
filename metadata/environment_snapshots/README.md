# Environment Snapshots

| File | Historical source |
| --- | --- |
| `requirements_original.txt` | Original `source/requirements.txt` from commit `8b06ce7` |
| `text_requirements_original.txt` | Original `source/text_requirements.txt` from commit `8b06ce7` |
| `normalization.json` | Replacement rules and package-version evidence |

The original exports are preserved byte-for-byte as provenance. They contain server-local package paths. The normalized files under `source/` replace those paths with public package names, pinning versions recoverable from wheel filenames. Exact versions absent from the records remain unpinned.

Use [Installation](../../docs/installation.md) for the current dependency references and the distinction between statistical, model, and separate text environments.
