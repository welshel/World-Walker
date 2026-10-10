# P_RELEASE2 Final Report

FINAL_STATUS = P_RELEASE2_PASS_GITHUB_PUBLICATION_READY

## Release

PROJECT_NAME = World-Walker
RELEASE_ROOT = ict_project/release/natural_language_3d_video_pipeline_release1
GIT_COMMIT = 60c59d07b912fceae5cfe18c115d52f6050a329c
GIT_COMMIT_SCOPE = Release code, documentation, teaser, and verification artifacts. This report is committed in a follow-up audit-only commit.
README_POLISHED = YES
GITHUB_REMOTE_TARGET = https://github.com/welshel/World-Walker.git
SERVER_GITHUB_REMOTE_CONFIGURED = NO
GITHUB_LOGIN_OR_PUSH_PERFORMED = NO
READY_TO_PUSH = YES

## Teaser

TEASER_CREATED = YES
TEASER_PATH = docs/assets/teaser.webp
TEASER_SIZE_BYTES = 271800
TEASER_SOURCE_EXPERIMENT = P_PIPE0_garden_end_to_end_smoke
TEASER_SOURCE_REPORT_STATUS = P_PIPE1_PASS_IMAGE_TO_FIXANYTHING
TEASER_SOURCE_FIXANYTHING = SUCCESS, native 61 frames at 24 FPS
TEASER_SOURCE_ALIGNMENT = global000 GOOD; global060 GOOD
TEASER_SOURCE_FRAME_INDICES = 0, 15, 30, 45, 60
TEASER_COLUMNS = Proxy; World-Walker
TEASER_VISUAL_REVIEW = PASS; aspect ratio preserved, no image content edits, and sampled rows were visually checked.
TEASER_MANIFEST = docs/assets/teaser_manifest.json
FRESH_CLONE_E2E_CLAIMED = NO

## Anchor alignment policy

ANCHOR_ALIGNMENT_METRIC_UNCHANGED = TRUE
ANCHOR_FAIL_OVERRIDE_IMPLEMENTED = TRUE
ANCHOR_FAIL_OVERRIDE_DEFAULT = FALSE
ANCHOR_GOOD_POLICY = Continue
ANCHOR_PARTIAL_POLICY = Warn and continue
ANCHOR_FAIL_POLICY = Block by default; continue only with the explicit CLI flag or config option
ANCHOR_OVERRIDE_WARNING = Printed with the required wording when an explicit override is used
ANCHOR_ALIGNMENT_OVERRIDE_USED_RECORDED = Stage manifests and final report
ANCHOR_OVERRIDE_RESUME_FINGERPRINT = Included for flux_anchor, fixanything, and report; legacy-cache fallback is disabled for these stages
ANCHOR_OVERRIDE_TESTS = PASS

## Runtime and tests

SEEDVR2_DEFAULT_FALSE = TRUE
UNIT_TESTS = PASS (37)
COMPILE_TEST = PASS
CLI_TEST = PASS; World-Walker name and override flag present
DRY_RUN_TEST = PASS; default override off, explicit override on, no model execution
CONFIG_TEST = PASS; all three example configs use World-Walker, default override off, and SeedVR2 off
README_RELATIVE_LINKS = PASS
FLUX_ALIGNMENT_METRIC_DIFF = NONE

## Scans

TRACKED_SECRET_SCAN = PASS; .env is ignored and not tracked; .env.example contains placeholders only
PRIVATE_PATH_SCAN = PASS
LARGE_FILE_SCAN = PASS; no tracked file exceeds 5 MiB
MEDIA_WEIGHT_SCAN = PASS; no video, PLY, or model weight files included
README_FENCE_SCAN = PASS
DOCS_ASSETS_TOTAL_BYTES = 272545

## Repository size

TRACKED_FILE_COUNT = 109
TRACKED_CONTENT_SIZE_BYTES = 908755 (about 0.87 MiB)
GIT_DIRECTORY_SIZE = about 1.1 MiB

## Audit files

- release_audit/P_RELEASE2_verification/unittest.log
- release_audit/P_RELEASE2_verification/compile.log
- release_audit/P_RELEASE2_verification/config_check.json
- release_audit/P_RELEASE2_verification/dryrun_default.json
- release_audit/P_RELEASE2_verification/dryrun_override.json
- release_audit/P_RELEASE2_verification/static_scan.json

## License and credential notes

PROJECT_LICENSE_READY = NO
The project-level license remains unselected. No MIT, Apache, or other project license is claimed. Choose a license before presenting the repository as a licensed public release; third-party code and model terms remain separate.

The previously exposed DeepSeek API key must be rotated. It was not reused or included in this release.
