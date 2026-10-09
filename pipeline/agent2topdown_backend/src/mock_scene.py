"""Deterministic built-in garden used when no API key is configured."""


def build_mock_scene():
    return {
        "scene_type": "garden",
        "canvas": {"width": 1024, "height": 1024},
        "background": {"type": "grass"},
        "regions": [
            {"id": "path_main", "type": "stone_path", "polygon": [[0.46, 0.0], [0.54, 0.0], [0.54, 1.0], [0.46, 1.0]]},
            {"id": "flowerbed_west", "type": "flowerbed", "polygon": [[0.12, 0.34], [0.37, 0.34], [0.37, 0.62], [0.12, 0.62]]},
            {"id": "flowerbed_east", "type": "flowerbed", "polygon": [[0.63, 0.34], [0.88, 0.34], [0.88, 0.62], [0.63, 0.62]]},
            {"id": "pond_south", "type": "pond", "polygon": [[0.38, 0.79], [0.62, 0.79], [0.62, 0.92], [0.38, 0.92]]},
        ],
        "objects": [
            {"id": "greenhouse", "type": "greenhouse", "shape": "rectangle", "bbox": [0.31, 0.07, 0.69, 0.23], "rotation": 0},
            {"id": "fountain", "type": "fountain", "shape": "circle", "center": [0.5, 0.49], "radius": 0.07},
            {"id": "bench_west", "type": "bench", "shape": "rectangle", "bbox": [0.25, 0.68, 0.37, 0.72], "rotation": 0},
            {"id": "bench_east", "type": "bench", "shape": "rectangle", "bbox": [0.63, 0.68, 0.75, 0.72], "rotation": 0},
        ],
        "scatters": [
            {"id": "trees_west", "type": "tree", "region": [[0.04, 0.1], [0.16, 0.1], [0.16, 0.96], [0.04, 0.96]], "count": 9},
            {"id": "trees_east", "type": "tree", "region": [[0.84, 0.1], [0.96, 0.1], [0.96, 0.96], [0.84, 0.96]], "count": 9},
            {"id": "flower_accents", "type": "flower_cluster", "region": [[0.13, 0.36], [0.86, 0.36], [0.86, 0.61], [0.13, 0.61]], "count": 24},
        ],
        "style_description": "A neat ornamental garden with a glass greenhouse to the north, a circular fountain at the center, balanced flower beds, mature trees along both edges, and calm stone paths.",
    }
