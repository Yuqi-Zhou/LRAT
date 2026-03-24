# Third-Party Notices

This repository contains original LRAT code together with vendored third-party components used to support retrieval, training, and evaluation workflows.

## Vendored Components

### FlagEmbedding

- Path: `FlagEmbedding/`
- Upstream project: `FlagOpen/FlagEmbedding`
- Upstream license: MIT
- Local status in this repository:
  - this is a modified local copy
  - the current version includes user-side changes on top of upstream code
  - those local changes were themselves developed based on earlier external modifications

### Tevatron

- Path: `tevatron/`
- Upstream project: `texttron/tevatron`
- Upstream license: Apache License 2.0
- Local status in this repository:
  - this is a vendored upstream dependency
  - it is included here to support dense retrieval workflows and related utilities

## Notes

- Please preserve upstream license files when redistributing this repository or any vendored subdirectories.
- If you publish checkpoints, datasets, or code extracted from this repository, double-check the corresponding upstream license obligations.
- The top-level repository license applies to LRAT's own code and documentation, while vendored subdirectories retain their upstream licenses unless explicitly stated otherwise.
