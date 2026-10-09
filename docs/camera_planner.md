# Camera planner contract

The camera stage expects a selected world-to-camera trajectory and an audit report. The configured target is 61 native frames, collision checks, and no temporal resampling. The validated garden style uses elevated eye level and mild downward pitch. The planner must estimate scene up and a safe region, check camera-center, near-frustum, and forward-ray clearances, and fail closed when safety is uncertain.

The 61 poses are planned directly; they are not generated at another length and normalized later. This keeps clean endpoint indices [0, 60] aligned. The planner executable is external in this release and must produce the configured artifacts. Large or unusual scenes may need trajectory tuning.
