#!/usr/bin/env python3
"""
Compositional Text Templates for Lane Topology Reasoning

Schema:
  - Curvature: straight (0), left-curving (1), right-curving (2)
  - Topology role: regular (0), fork (1), merge (2), terminal (3)
  - Context: regular-road (0), intersection (1), highway (2)

Total: 3 × 4 × 3 = 36 templates
Template ID = curvature_idx * 12 + role_idx * 3 + context_idx
"""

# Attribute definitions
CURVATURE = {
    0: "straight",
    1: "left-curving",
    2: "right-curving"
}

TOPOLOGY_ROLE = {
    0: "regular",
    1: "fork",
    2: "merge",
    3: "terminal"
}

CONTEXT = {
    0: "regular-road",
    1: "intersection",
    2: "highway"
}

# Role descriptions for template generation
ROLE_DESCRIPTIONS = {
    "regular": "in continuous path",
    "fork": "at diverging junction",
    "merge": "at converging junction",
    "terminal": "at termination point"
}

CONTEXT_DESCRIPTIONS = {
    "regular-road": "on regular road",
    "intersection": "within intersection",
    "highway": "on highway"
}

def generate_template(curvature_idx, role_idx, context_idx):
    """Generate compositional text template."""
    curvature = CURVATURE[curvature_idx]
    role = TOPOLOGY_ROLE[role_idx]
    context = CONTEXT[context_idx]

    role_desc = ROLE_DESCRIPTIONS[role]
    context_desc = CONTEXT_DESCRIPTIONS[context]

    template = f"{curvature} lane segment {role_desc} {context_desc}"
    return template

def get_template_id(curvature_idx, role_idx, context_idx):
    """Convert attribute indices to template ID."""
    return curvature_idx * 12 + role_idx * 3 + context_idx

def parse_template_id(template_id):
    """Convert template ID to attribute indices."""
    curvature_idx = template_id // 12
    role_idx = (template_id % 12) // 3
    context_idx = template_id % 3
    return curvature_idx, role_idx, context_idx

# Generate all 36 templates
TEXT_TEMPLATES = []
TEMPLATE_METADATA = []

for curv_idx in range(3):  # straight, left, right
    for role_idx in range(4):  # regular, fork, merge, terminal
        for ctx_idx in range(3):  # regular-road, intersection, highway
            template_id = get_template_id(curv_idx, role_idx, ctx_idx)
            template_text = generate_template(curv_idx, role_idx, ctx_idx)

            TEXT_TEMPLATES.append(template_text)
            TEMPLATE_METADATA.append({
                "id": template_id,
                "curvature": CURVATURE[curv_idx],
                "topology_role": TOPOLOGY_ROLE[role_idx],
                "context": CONTEXT[ctx_idx],
                "text": template_text
            })

# Verify
assert len(TEXT_TEMPLATES) == 36
assert len(TEMPLATE_METADATA) == 36

if __name__ == "__main__":
    print("=" * 80)
    print("36 Compositional Text Templates")
    print("=" * 80)

    print(f"\nTotal templates: {len(TEXT_TEMPLATES)}")

    # Print by category
    for curv_idx, curv_name in CURVATURE.items():
        print(f"\n{'='*80}")
        print(f"{curv_name.upper()} LANES")
        print(f"{'='*80}")

        for role_idx, role_name in TOPOLOGY_ROLE.items():
            print(f"\n  {role_name.upper()}:")
            for ctx_idx, ctx_name in CONTEXT.items():
                template_id = get_template_id(curv_idx, role_idx, ctx_idx)
                template_text = TEXT_TEMPLATES[template_id]
                print(f"    [{template_id:2d}] {template_text}")

    # Examples
    print("\n" + "="*80)
    print("EXAMPLES")
    print("="*80)

    examples = [
        (0, 0, 0, "Straight lane on regular road, regular connection"),
        (0, 1, 1, "Straight lane at fork within intersection"),
        (1, 0, 1, "Left-curving lane in intersection, continuous path"),
        (2, 2, 0, "Right-curving lane merging on regular road"),
        (0, 3, 2, "Straight lane terminating on highway"),
    ]

    for curv, role, ctx, description in examples:
        tid = get_template_id(curv, role, ctx)
        text = TEXT_TEMPLATES[tid]
        print(f"\n{description}:")
        print(f"  ID: {tid}")
        print(f"  Template: \"{text}\"")

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Curvature types: {len(CURVATURE)}")
    print(f"Topology roles: {len(TOPOLOGY_ROLE)}")
    print(f"Context types: {len(CONTEXT)}")
    print(f"Total combinations: {len(CURVATURE)} × {len(TOPOLOGY_ROLE)} × {len(CONTEXT)} = {len(TEXT_TEMPLATES)}")
