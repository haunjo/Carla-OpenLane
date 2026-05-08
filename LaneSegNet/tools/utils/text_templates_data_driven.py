#!/usr/bin/env python3
"""
Data-Driven Text Templates for Lane Topology Reasoning

Based on actual Carla-OLV2 dataset inspection, focusing on:
1. Curvature: straight, left-turn, right-turn
2. Intersection: within intersection, connected to intersection, not connected

Total: 3 × 3 = 9 templates
Template ID = curvature_idx * 3 + intersection_idx
"""

# Attribute definitions
CURVATURE = {
    0: "straight",
    1: "left-turn",
    2: "right-turn"
}

INTERSECTION_TYPE = {
    0: "regular",           # Not connected to intersection
    1: "connector",         # Connected to intersection (entry/exit)
    2: "intersection"       # Within intersection
}

# Template descriptions
CURVATURE_DESC = {
    "straight": "straight lane",
    "left-turn": "left-turning lane",
    "right-turn": "right-turning lane"
}

INTERSECTION_DESC = {
    "regular": "on regular road",
    "connector": "connecting to intersection",
    "intersection": "within intersection area"
}

def generate_template(curvature_idx, intersection_idx):
    """Generate compositional text template."""
    curvature = CURVATURE[curvature_idx]
    intersection = INTERSECTION_TYPE[intersection_idx]

    curv_desc = CURVATURE_DESC[curvature]
    int_desc = INTERSECTION_DESC[intersection]

    # Natural language composition
    template = f"{curv_desc} {int_desc}"
    return template

def get_template_id(curvature_idx, intersection_idx):
    """Convert attribute indices to template ID."""
    return curvature_idx * 3 + intersection_idx

def parse_template_id(template_id):
    """Convert template ID to attribute indices."""
    curvature_idx = template_id // 3
    intersection_idx = template_id % 3
    return curvature_idx, intersection_idx

# Generate all 9 templates
TEXT_TEMPLATES = []
TEMPLATE_METADATA = []

for curv_idx in range(3):  # straight, left, right
    for int_idx in range(3):  # regular, connector, intersection
        template_id = get_template_id(curv_idx, int_idx)
        template_text = generate_template(curv_idx, int_idx)

        TEXT_TEMPLATES.append(template_text)
        TEMPLATE_METADATA.append({
            "id": template_id,
            "curvature": CURVATURE[curv_idx],
            "intersection_type": INTERSECTION_TYPE[int_idx],
            "text": template_text
        })

# Verify
assert len(TEXT_TEMPLATES) == 9
assert len(TEMPLATE_METADATA) == 9

# Attribute extraction helpers
def classify_curvature(centerline_points, threshold_left=15, threshold_right=-15):
    """
    Classify lane curvature from centerline points.

    Args:
        centerline_points: Nx3 array of 3D points
        threshold_left: Angle threshold for left turn (degrees)
        threshold_right: Angle threshold for right turn (degrees)

    Returns:
        curvature_idx: 0 (straight), 1 (left-turn), 2 (right-turn)
    """
    import numpy as np

    if len(centerline_points) < 3:
        return 0  # Default to straight

    # Compute overall direction change
    start = centerline_points[0][:2]
    middle = centerline_points[len(centerline_points)//2][:2]
    end = centerline_points[-1][:2]

    # Vectors
    vec1 = middle - start
    vec2 = end - middle

    # Angle (cross product for sign, dot product for magnitude)
    cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    angle = np.arctan2(cross, np.dot(vec1, vec2)) * 180 / np.pi

    if angle > threshold_left:
        return 1  # Left turn
    elif angle < threshold_right:
        return 2  # Right turn
    else:
        return 0  # Straight

def classify_intersection_type(lane_dict, lanelet_map=None, routing_graph=None):
    """
    Classify lane's relationship to intersection.

    Args:
        lane_dict: Lane dictionary with 'is_intersection_or_connector' field
        lanelet_map: Optional Lanelet2 map for detailed analysis
        routing_graph: Optional routing graph for connectivity

    Returns:
        intersection_idx: 0 (regular), 1 (connector), 2 (intersection)
    """
    # OpenLane-V2 provides 'is_intersection_or_connector' flag
    is_intersection = lane_dict.get('is_intersection_or_connector', False)

    if not is_intersection:
        return 0  # Regular road

    # If lanelet_map available, can distinguish connector vs within
    # For now, use heuristics:
    # - Short lanes (< 20m) near intersection → connector
    # - Longer lanes within intersection → intersection

    if lanelet_map is not None:
        # TODO: Use lanelet_map for precise classification
        pass

    # Simple heuristic: check lane length
    centerline = lane_dict.get('centerline', [])
    if len(centerline) < 2:
        return 1  # Default to connector

    import numpy as np
    length = 0
    for i in range(len(centerline) - 1):
        p1 = centerline[i][:2]
        p2 = centerline[i+1][:2]
        length += np.linalg.norm(p2 - p1)

    if length < 15:  # Short → connector
        return 1
    else:  # Long → within intersection
        return 2

if __name__ == "__main__":
    print("=" * 80)
    print("9 Data-Driven Text Templates")
    print("=" * 80)

    print(f"\nTotal templates: {len(TEXT_TEMPLATES)}")

    # Print by category
    for curv_idx, curv_name in CURVATURE.items():
        print(f"\n{'='*80}")
        print(f"{curv_name.upper()} LANES")
        print(f"{'='*80}")

        for int_idx, int_name in INTERSECTION_TYPE.items():
            template_id = get_template_id(curv_idx, int_idx)
            template_text = TEXT_TEMPLATES[template_id]
            print(f"  [{template_id}] {int_name:12s}: {template_text}")

    # Examples
    print("\n" + "="*80)
    print("EXAMPLES")
    print("="*80)

    examples = [
        (0, 0, "Most common: Straight lane on regular road"),
        (0, 1, "Highway entry: Straight lane connecting to intersection"),
        (0, 2, "Intersection through: Straight lane within intersection"),
        (1, 2, "Left turn in intersection"),
        (2, 2, "Right turn in intersection"),
        (1, 1, "Approach: Left-turning lane connecting to intersection"),
    ]

    for curv, inter, description in examples:
        tid = get_template_id(curv, inter)
        text = TEXT_TEMPLATES[tid]
        print(f"\n{description}:")
        print(f"  ID: {tid}")
        print(f"  Curvature: {CURVATURE[curv]}, Intersection: {INTERSECTION_TYPE[inter]}")
        print(f"  Template: \"{text}\"")

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Curvature types: {len(CURVATURE)}")
    print(f"Intersection types: {len(INTERSECTION_TYPE)}")
    print(f"Total combinations: {len(CURVATURE)} × {len(INTERSECTION_TYPE)} = {len(TEXT_TEMPLATES)}")

    print("\n" + "="*80)
    print("ALL 9 TEMPLATES")
    print("="*80)
    for i, template in enumerate(TEXT_TEMPLATES):
        curv_idx, int_idx = parse_template_id(i)
        print(f"[{i}] {template}")
        print(f"    → curvature={CURVATURE[curv_idx]}, intersection={INTERSECTION_TYPE[int_idx]}")
