# Troubleshooting

- Provide exactly one of --scene-prompt or --input-topdown.
- Start with --dry-run; it validates inputs/config and prints the planned stages without model execution.
- Run scripts/check_environment.py and configure all external paths needed by selected stages.
- In text mode, inspect the FLUX top-down candidates and complete visual review before TripoSplat.
- Treat camera safety failures as a stop; do not bypass clearance or native-frame checks.
- Keep private manifests and logs out of public commits.
