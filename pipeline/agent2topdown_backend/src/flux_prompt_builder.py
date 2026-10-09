"""Create a copy-ready FLUX prompt anchored to the generated layout."""


def _direction(x, y):
    vertical = "north" if y < 0.34 else "south" if y > 0.66 else "central"
    horizontal = "west" if x < 0.38 else "east" if x > 0.62 else "center"
    if vertical == "central":
        return {"west": "west side", "east": "east side", "center": "center"}[horizontal]
    if horizontal == "center":
        return f"{vertical} side"
    return f"{vertical}-{horizontal} area"


def _polygon_center(polygon):
    return sum(p[0] for p in polygon) / len(polygon), sum(p[1] for p in polygon) / len(polygon)


def summarize_layout(scene):
    facts = []
    for region in scene["regions"]:
        x, y = _polygon_center(region["polygon"])
        facts.append(f"{region['type'].replace('_', ' ')} region {region['id']} at the {_direction(x, y)}")
    for item in scene["objects"]:
        if item["shape"] == "circle":
            x, y = item["center"]
            geometry = "circular"
        else:
            x1, y1, x2, y2 = item["bbox"]
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            geometry = "rectangular"
        facts.append(f"{geometry} {item['type']} {item['id']} at the {_direction(x, y)}")
    for item in scene["scatters"]:
        x, y = _polygon_center(item["region"])
        facts.append(f"{item['count']} {item['type'].replace('_', ' ')} elements distributed within the {_direction(x, y)} area")
    return "; ".join(facts) if facts else "open grass lawn with a clear, balanced garden plan"


def build_flux_prompt(scene):
    return f"""Use the provided layout image as the authoritative spatial layout reference.

Create a photorealistic garden image by refining the supplied plan. Preserve the top-down viewpoint. Preserve the positions, shapes, sizes, and relative arrangement of all major regions and objects exactly as shown in the layout. Do not redesign the scene. Do not move, add, remove, or replace major objects. Refine the layout into a photorealistic top-down garden scene while keeping its spatial structure stable.

Style and materials: {scene['style_description'].strip()}

Layout details: {summarize_layout(scene)}.

Keep the entire garden visible from directly overhead, with clear boundaries between paths, planting beds, water, lawn, and landmarks. Use natural materials, realistic foliage and lighting, and polished photographic detail without changing the plan."""
