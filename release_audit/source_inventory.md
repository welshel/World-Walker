# Source inventory audit

Audited 838 files across pipeline_v1 and P_PIPE1; remaining UNKNOWN entries: 0.

| Category | Count |
|---|---:|
| CACHE | 30 |
| EXPERIMENT_OUTPUT | 756 |
| LOG | 7 |
| PRIVATE_DATA | 11 |
| PROJECT_ASSET_SMALL | 2 |
| PROJECT_CONFIG | 3 |
| PROJECT_DOC | 1 |
| PROJECT_SOURCE | 30 |
| UNKNOWN | 0 |

Copied project-owned Python source files: 22; extracted small prompt assets: 2. Excluded bytes: 514.89 MiB.

The P_PIPE1 experiment directory was inventoried but not copied. Only the two FLUX endpoint prompt constants were extracted into configs/prompts/. Camera-v5 was reviewed and excluded because it is bound to the garden experiment, its reference trajectory, and a local renderer. Agent2Topdown contributed generated run artifacts, not source code.

See source_inventory.json for per-file classifications and exclusion reasons.
