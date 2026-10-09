"""Stable adapter API for the optional SeedVR2 stage."""
from pipeline.integrations.seedvr2_adapter import (
    build_command, collect_outputs, model_identity, run, validate_environment, video_info,
)

__all__ = ["build_command", "collect_outputs", "model_identity", "run", "validate_environment", "video_info"]
