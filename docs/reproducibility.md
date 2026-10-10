# Reproducibility and resume

Completed stages save manifests with configuration fingerprints, input hashes, output hashes, and status. Use --resume to reuse results only when those values still match, and --force-stage STAGE to rerun a named stage. Native frame count is 61, temporal resampling is false, and clean indices are [0, 60]. SeedVR2 defaults off and its toggle does not invalidate upstream stage fingerprints. The anchor alignment override defaults off and participates in the FLUX anchor, FixAnything, and report stage fingerprints.

Model weights and external repositories are not distributed. Record upstream revisions, model hashes, runtime versions, and the local config with each run.
