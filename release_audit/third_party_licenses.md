# Third-party provenance and license audit

No third-party repository source or model weights are redistributed by this repository. The TripoSplat inference wrapper and Agent2Topdown backend are project integration code. The inspected external checkout licenses are recorded below; model terms are separate from code licenses.

| Component | Upstream/reference | Observed revision | License evidence | Status |
|---|---|---|---|---|
| TripoSplat code | https://github.com/VAST-AI-Research/TripoSplat | `d8db9e018b413dd9c4a9fe22463781bf98e8e68d` reference commit; installed snapshot has no Git metadata | Inspected snapshot `LICENSE` begins `MIT License` | MIT observed for code; verify model/checkpoint terms separately |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio | `c458cb42ab1ee838bff85c6546e14bb01c3571e9` local HEAD | Inspected `LICENSE` is Apache License 2.0 | Apache-2.0 observed |
| FixAnything code | URL not verified from the inspected checkout | `744023e987920af352b880f60cc8841c96a0a105` local HEAD | Inspected `LICENSE` is Apache License 2.0 | Apache-2.0 observed; upstream URL remains unverified |
| FLUX.2 Klein 4B weights | `black-forest-labs/FLUX.2-klein-4B`, ModelScope local weights | Model revision not pinned in this audit | Model license terms not independently reviewed here | UNKNOWN |
| Wan2.1 weights | Wan-AI/Wan2.1 model family | Model revision not pinned in this audit | Model license terms not independently reviewed here | UNKNOWN |
| SeedVR2 weights/code | `bytedance-community/SeedVR2-3B` model reference | Not used in the E2E; repository commit unavailable | License text not inspected | UNKNOWN |
| Agent2Topdown API | DeepSeek API service called by the bundled project backend | API model configured per user environment | Service/API terms not inspected here | UNKNOWN |
| Python dependencies | PyTorch, PyYAML, Pillow, ImageIO, NumPy, SciPy, OpenCV, Requests, gsplat | Runtime-selected versions | Package license metadata not included in this audit | NOT_VERIFIED |

The project-level license remains unselected. No MIT, Apache, or other root license is claimed; see `LICENSE_SELECTION_REQUIRED.md`.

`SOURCE_RECOVERED_FROM`: project-owned Agent2Topdown and camera/rendering behavior was recovered from the previously audited P_PIPE0/P_PIPE1 and P_CAM1 source trees. Those host paths are intentionally not published in this portable audit file.
