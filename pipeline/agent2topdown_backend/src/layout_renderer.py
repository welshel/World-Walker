"""Deterministic Pillow renderer for a readable 1024x1024 top-down plan."""

import random
import math
from pathlib import Path

from PIL import Image, ImageDraw


SIZE = 1024
COLORS = {
    "grass": "#B9D99A",
    "stone_path": "#C9C8C2",
    "flowerbed": "#E894A9",
    "pond": "#75CBE2",
    "soil": "#9B704E",
    "greenhouse": "#9BD9DC",
    "fountain": "#399BCC",
    "bench": "#8B5B3D",
    "gazebo": "#E7D5AA",
    "statue": "#D5D9D6",
    "tree": "#286442",
    "shrub": "#4F8B55",
}
OUTLINE = "#35433A"


def _xy(point):
    return round(point[0] * (SIZE - 1)), round(point[1] * (SIZE - 1))


def _polygon_points(points):
    return [_xy(point) for point in points]


def _contains(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _scatter_points(polygon, count, rng):
    xs, ys = zip(*polygon)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    points = []
    attempts = 0
    while len(points) < count and attempts < count * 200:
        candidate = (rng.uniform(left, right), rng.uniform(top, bottom))
        if _contains(candidate, polygon):
            points.append(candidate)
        attempts += 1
    if len(points) != count:
        raise ValueError("Unable to place scatter points inside its region; check polygon geometry")
    return points


def render_layout(scene, output_path, labels=False, seed=17):
    output_path = Path(output_path)
    image = Image.new("RGB", (SIZE, SIZE), COLORS[scene["background"]["type"]])
    draw = ImageDraw.Draw(image)
    for region in scene["regions"]:
        points = _polygon_points(region["polygon"])
        color = COLORS[region["type"]]
        draw.polygon(points, fill=color, outline=OUTLINE)
        if labels:
            center = (sum(p[0] for p in points) // len(points), sum(p[1] for p in points) // len(points))
            draw.text(center, region["type"], fill="#1F2B22", anchor="mm")

    for item in scene["objects"]:
        color = COLORS[item["type"]]
        if item["shape"] == "rectangle":
            x1, y1, x2, y2 = item["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            angle = item.get("rotation", 0)
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            radians = math.radians(angle)
            rotated = []
            for x, y in corners:
                dx, dy = x - cx, y - cy
                rotated.append((cx + dx * math.cos(radians) - dy * math.sin(radians), cy + dx * math.sin(radians) + dy * math.cos(radians)))
            draw.polygon(_polygon_points(rotated), fill=color, outline=OUTLINE)
        else:
            cx, cy = _xy(item["center"])
            radius = round(item["radius"] * (SIZE - 1))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=OUTLINE, width=3)
        if labels:
            anchor = _xy(item.get("center", [(item["bbox"][0] + item["bbox"][2]) / 2, (item["bbox"][1] + item["bbox"][3]) / 2]))
            draw.text(anchor, item["type"], fill="#17251B", anchor="mm")

    rng = random.Random(seed)
    for scatter in scene["scatters"]:
        for point in _scatter_points(scatter["region"], scatter["count"], rng):
            x, y = _xy(point)
            if scatter["type"] == "tree":
                radius, fill = 15, COLORS["tree"]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="#183E2B", width=3)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#77A86A")
            elif scatter["type"] == "shrub":
                radius, fill = 10, COLORS["shrub"]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=OUTLINE, width=2)
            else:
                for dx, dy, fill in ((-5, 0, "#F5D45C"), (5, 0, "#E56C83"), (0, -5, "#F4F0D0")):
                    draw.ellipse((x + dx - 3, y + dy - 3, x + dx + 3, y + dy + 3), fill=fill, outline="#8A564C")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path
