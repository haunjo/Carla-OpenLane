#!/usr/bin/env python3
"""
Verify CLIP embedding separation for 36 compositional text templates.
"""
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from text_templates_compositional import TEXT_TEMPLATES, TEMPLATE_METADATA, parse_template_id

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

print("="*80)
print("CLIP Embedding Verification for 36 Templates")
print("="*80)

# Compute embeddings
print(f"\nEncoding {len(TEXT_TEMPLATES)} templates...")
texts = clip.tokenize(TEXT_TEMPLATES).to(device)
with torch.no_grad():
    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarity = (text_features @ text_features.T).cpu().numpy()
print("Done!")

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

off_diagonal = similarity[np.triu_indices_from(similarity, k=1)]
print(f"Min similarity:  {off_diagonal.min():.4f}")
print(f"Max similarity:  {off_diagonal.max():.4f}")
print(f"Mean similarity: {off_diagonal.mean():.4f}")
print(f"Std similarity:  {off_diagonal.std():.4f}")

# Histogram
print("\nDistribution of pairwise similarities:")
bins = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
hist, _ = np.histogram(off_diagonal, bins=bins)
for i in range(len(bins)-1):
    pct = hist[i] / len(off_diagonal) * 100
    status = "✅" if bins[i+1] <= 0.7 else ("⚠️" if bins[i+1] <= 0.85 else "🚨")
    print(f"  {status} [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:4d} pairs ({pct:5.1f}%)")

# Analysis by attribute
print("\n" + "="*80)
print("WITHIN-ATTRIBUTE SIMILARITY (Should be HIGH)")
print("="*80)

def get_within_attribute_similarity(attr_name):
    """Compute similarity within same attribute group."""
    sims = []

    if attr_name == "curvature":
        # Same curvature (e.g., all straight lanes)
        for curv_idx in range(3):
            indices = [tid for tid in range(36) if parse_template_id(tid)[0] == curv_idx]
            for i in indices:
                for j in indices:
                    if i < j:
                        sims.append(similarity[i, j])

    elif attr_name == "topology_role":
        # Same topology role (e.g., all fork lanes)
        for role_idx in range(4):
            indices = [tid for tid in range(36) if parse_template_id(tid)[1] == role_idx]
            for i in indices:
                for j in indices:
                    if i < j:
                        sims.append(similarity[i, j])

    elif attr_name == "context":
        # Same context (e.g., all intersection lanes)
        for ctx_idx in range(3):
            indices = [tid for tid in range(36) if parse_template_id(tid)[2] == ctx_idx]
            for i in indices:
                for j in indices:
                    if i < j:
                        sims.append(similarity[i, j])

    return np.array(sims)

# Within-attribute (should be high - they share attribute)
for attr in ["curvature", "topology_role", "context"]:
    sims = get_within_attribute_similarity(attr)
    print(f"\n{attr.upper()} (same {attr}):")
    print(f"  Mean: {sims.mean():.4f}, Std: {sims.std():.4f}")
    print(f"  Range: [{sims.min():.4f}, {sims.max():.4f}]")

# Between-attribute analysis
print("\n" + "="*80)
print("CRITICAL DISTINCTIONS (Should be LOW)")
print("="*80)

# Curvature distinction
print("\nCurvature distinctions:")
curv_pairs = [
    (0, 1, "straight vs left-curving"),
    (0, 2, "straight vs right-curving"),
    (1, 2, "left-curving vs right-curving")
]

for c1, c2, desc in curv_pairs:
    indices1 = [tid for tid in range(36) if parse_template_id(tid)[0] == c1]
    indices2 = [tid for tid in range(36) if parse_template_id(tid)[0] == c2]

    sims = []
    for i in indices1:
        for j in indices2:
            sims.append(similarity[i, j])

    sims = np.array(sims)
    status = "✅" if sims.mean() < 0.7 else "⚠️"
    print(f"  {status} {desc:35s}: mean={sims.mean():.4f}, std={sims.std():.4f}")

# Topology role distinction
print("\nTopology role distinctions:")
role_pairs = [
    (0, 1, "regular vs fork"),
    (0, 2, "regular vs merge"),
    (1, 2, "fork vs merge"),
    (0, 3, "regular vs terminal")
]

for r1, r2, desc in role_pairs:
    indices1 = [tid for tid in range(36) if parse_template_id(tid)[1] == r1]
    indices2 = [tid for tid in range(36) if parse_template_id(tid)[1] == r2]

    sims = []
    for i in indices1:
        for j in indices2:
            sims.append(similarity[i, j])

    sims = np.array(sims)
    status = "✅" if sims.mean() < 0.7 else "⚠️"
    print(f"  {status} {desc:35s}: mean={sims.mean():.4f}, std={sims.std():.4f}")

# Context distinction
print("\nContext distinctions:")
ctx_pairs = [
    (0, 1, "regular-road vs intersection"),
    (0, 2, "regular-road vs highway"),
    (1, 2, "intersection vs highway")
]

for ct1, ct2, desc in ctx_pairs:
    indices1 = [tid for tid in range(36) if parse_template_id(tid)[2] == ct1]
    indices2 = [tid for tid in range(36) if parse_template_id(tid)[2] == ct2]

    sims = []
    for i in indices1:
        for j in indices2:
            sims.append(similarity[i, j])

    sims = np.array(sims)
    status = "✅" if sims.mean() < 0.7 else "⚠️"
    print(f"  {status} {desc:35s}: mean={sims.mean():.4f}, std={sims.std():.4f}")

# Visualization: Heatmap for selected templates
print("\n" + "="*80)
print("Generating visualizations...")
print("="*80)

# Select representative subset (12 templates)
representative_ids = [
    0,   # straight, regular, regular-road
    1,   # straight, regular, intersection
    3,   # straight, fork, regular-road
    6,   # straight, merge, regular-road
    9,   # straight, terminal, regular-road
    12,  # left-curving, regular, regular-road
    15,  # left-curving, fork, regular-road
    18,  # left-curving, merge, regular-road
    24,  # right-curving, regular, regular-road
    27,  # right-curving, fork, regular-road
    30,  # right-curving, merge, regular-road
    33,  # right-curving, terminal, regular-road
]

representative_labels = [
    "S-Reg-RR",    # 0
    "S-Reg-Int",   # 1
    "S-Fork-RR",   # 3
    "S-Merg-RR",   # 6
    "S-Term-RR",   # 9
    "L-Reg-RR",    # 12
    "L-Fork-RR",   # 15
    "L-Merg-RR",   # 18
    "R-Reg-RR",    # 24
    "R-Fork-RR",   # 27
    "R-Merg-RR",   # 30
    "R-Term-RR",   # 33
]

# Extract subset similarity
subset_sim = similarity[np.ix_(representative_ids, representative_ids)]

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(subset_sim, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=representative_labels, yticklabels=representative_labels,
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cosine Similarity'},
            square=True)
ax.set_title('CLIP Embedding Similarity: Representative 12/36 Templates\n(S=Straight, L=Left, R=Right, Reg=Regular, Fork/Merg/Term, RR=Regular-Road, Int=Intersection)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
output_path = 'text_template_36_similarity.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Full heatmap (36x36)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(similarity, cmap='RdYlGn_r', vmin=0, vmax=1, ax=ax,
            cbar_kws={'label': 'Cosine Similarity'}, square=True)
ax.set_title('Full 36x36 Template Similarity Matrix', fontsize=12, fontweight='bold')
ax.set_xlabel('Template ID')
ax.set_ylabel('Template ID')
plt.tight_layout()
output_path_full = 'text_template_36_similarity_full.png'
plt.savefig(output_path_full, dpi=150, bbox_inches='tight')
print(f"Full heatmap saved to: {output_path_full}")

# Final assessment
print("\n" + "="*80)
print("ASSESSMENT")
print("="*80)

assessment_score = 0

# Criterion 1: Mean off-diagonal similarity (lower is better)
if off_diagonal.mean() < 0.7:
    print("✅ Mean similarity < 0.7: GOOD")
    assessment_score += 3
elif off_diagonal.mean() < 0.85:
    print("⚠️  Mean similarity < 0.85: ACCEPTABLE")
    assessment_score += 2
else:
    print("🚨 Mean similarity >= 0.85: TOO HIGH")
    assessment_score += 1

# Criterion 2: Topology role distinction (most important)
role_sims = []
for r1, r2, _ in role_pairs:
    indices1 = [tid for tid in range(36) if parse_template_id(tid)[1] == r1]
    indices2 = [tid for tid in range(36) if parse_template_id(tid)[1] == r2]
    sims = [similarity[i, j] for i in indices1 for j in indices2]
    role_sims.extend(sims)
role_sims = np.array(role_sims)

if role_sims.mean() < 0.65:
    print("✅ Topology role distinction < 0.65: EXCELLENT")
    assessment_score += 3
elif role_sims.mean() < 0.75:
    print("⚠️  Topology role distinction < 0.75: GOOD")
    assessment_score += 2
else:
    print("🚨 Topology role distinction >= 0.75: WEAK")
    assessment_score += 1

# Criterion 3: Max similarity (should not be too high)
if off_diagonal.max() < 0.90:
    print("✅ Max similarity < 0.90: GOOD")
    assessment_score += 2
elif off_diagonal.max() < 0.95:
    print("⚠️  Max similarity < 0.95: ACCEPTABLE")
    assessment_score += 1
else:
    print("🚨 Max similarity >= 0.95: ALMOST IDENTICAL")

print(f"\nOverall score: {assessment_score}/8")

if assessment_score >= 7:
    print("✅ VERDICT: Templates are well-separated. Proceed with implementation!")
elif assessment_score >= 5:
    print("⚠️  VERDICT: Templates are acceptable but could be improved.")
else:
    print("🚨 VERDICT: Templates need refinement.")

print("\n" + "="*80)
print("DONE")
print("="*80)
