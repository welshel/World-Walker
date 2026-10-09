"""Strict, dependency-free validation for the supported garden scene schema."""

from copy import deepcopy
from typing import Any

from .schema import BACKGROUND_TYPES, CANVAS, OBJECT_SHAPES, OBJECT_TYPES, REGION_TYPES, SCATTER_TYPES, SCENE_TYPE


class SceneValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("Scene validation failed:\n- " + "\n- ".join(self.errors))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_point(point, path, errors):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        errors.append(f"{path} must be a two-number [x, y] point")
        return False
    ok = True
    for axis, value in zip(("x", "y"), point):
        if not _is_number(value) or not 0 <= value <= 1:
            errors.append(f"{path}.{axis} must be a number in [0, 1]")
            ok = False
    return ok


def _check_polygon(polygon, path, errors):
    if not isinstance(polygon, list) or len(polygon) < 3:
        errors.append(f"{path} must be a polygon with at least 3 points")
        return
    for index, point in enumerate(polygon):
        _check_point(point, f"{path}[{index}]", errors)


def validate_scene(scene: Any) -> dict:
    errors = []
    if not isinstance(scene, dict):
        raise SceneValidationError(["top-level JSON value must be an object"])

    required = ("scene_type", "canvas", "background", "regions", "objects", "scatters", "style_description")
    for key in required:
        if key not in scene:
            errors.append(f"missing required field: {key}")
    if errors:
        raise SceneValidationError(errors)

    if scene["scene_type"] != SCENE_TYPE:
        errors.append('scene_type must be "garden"')
    canvas = scene["canvas"]
    if not isinstance(canvas, dict) or canvas != CANVAS:
        errors.append("canvas must be exactly {\"width\": 1024, \"height\": 1024}")
    background = scene["background"]
    if not isinstance(background, dict) or background.get("type") not in BACKGROUND_TYPES:
        errors.append(f"background.type must be one of {sorted(BACKGROUND_TYPES)}")

    ids = set()
    for collection_name in ("regions", "objects", "scatters"):
        collection = scene[collection_name]
        if not isinstance(collection, list):
            errors.append(f"{collection_name} must be an array")
            continue
        for index, item in enumerate(collection):
            path = f"{collection_name}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{path} must be an object")
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                errors.append(f"{path}.id must be a non-empty string")
            elif item_id in ids:
                errors.append(f"duplicate id: {item_id}")
            else:
                ids.add(item_id)

            if collection_name == "regions":
                if item.get("type") not in REGION_TYPES:
                    errors.append(f"{path}.type must be one of {sorted(REGION_TYPES)}")
                _check_polygon(item.get("polygon"), f"{path}.polygon", errors)
            elif collection_name == "objects":
                if item.get("type") not in OBJECT_TYPES:
                    errors.append(f"{path}.type must be one of {sorted(OBJECT_TYPES)}")
                shape = item.get("shape")
                if shape not in OBJECT_SHAPES:
                    errors.append(f"{path}.shape must be one of {sorted(OBJECT_SHAPES)}")
                elif shape == "rectangle":
                    bbox = item.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(v) for v in bbox):
                        errors.append(f"{path}.bbox must be [x1, y1, x2, y2]")
                    else:
                        x1, y1, x2, y2 = bbox
                        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                            errors.append(f"{path}.bbox must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
                    rotation = item.get("rotation", 0)
                    if not _is_number(rotation):
                        errors.append(f"{path}.rotation must be a number when provided")
                elif shape == "circle":
                    _check_point(item.get("center"), f"{path}.center", errors)
                    radius = item.get("radius")
                    if not _is_number(radius) or radius <= 0:
                        errors.append(f"{path}.radius must be a positive number")
            else:
                if item.get("type") not in SCATTER_TYPES:
                    errors.append(f"{path}.type must be one of {sorted(SCATTER_TYPES)}")
                _check_polygon(item.get("region"), f"{path}.region", errors)
                count = item.get("count")
                if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                    errors.append(f"{path}.count must be a positive integer")

    description = scene["style_description"]
    if not isinstance(description, str) or not description.strip():
        errors.append("style_description must be a non-empty string")
    if errors:
        raise SceneValidationError(errors)
    return deepcopy(scene)
