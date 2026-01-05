#!/usr/bin/env python3
"""
Verify CLIP embedding separation for 9 data-driven text templates.
"""
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from text_templates_data_driven import TEXT_TEMPLATES, CURVATURE, INTERSECTION_TYPE, parse_template_id

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

print("="*80)
print("CLIP Embedding Verification for 9 Data-Driven Templates")
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

# Critical distinctions
print("\n" + "="*80)
print("CRITICAL DISTINCTIONS")
print("="*80)

# Curvature distinction (most important for topology)
print("\nCurvature distinctions (same intersection):")
for int_idx in range(3):
    int_name = INTERSECTION_TYPE[int_idx]
    print(f"\n  Within {int_name}:")

    # straight vs left
    id1 = 0 * 3 + int_idx
    id2 = 1 * 3 + int_idx
    sim = similarity[id1, id2]
    status = "✅" if sim < 0.7 else ("⚠️" if sim < 0.85 else "🚨")
    print(f"    {status} straight vs left-turn:  {sim:.4f}")

    # straight vs right
    id1 = 0 * 3 + int_idx
    id2 = 2 * 3 + int_idx
    sim = similarity[id1, id2]
    status = "✅" if sim < 0.7 else ("⚠️" if sim < 0.85 else "🚨")
    print(f"    {status} straight vs right-turn: {sim:.4f}")

    # left vs right
    id1 = 1 * 3 + int_idx
    id2 = 2 * 3 + int_idx
    sim = similarity[id1, id2]
    status = "✅" if sim < 0.7 else ("⚠️" if sim < 0.85 else "🚨")
    print(f"    {status} left-turn vs right-turn:{sim:.4f}")

# Intersection distinction (same curvature)
print("\nIntersection distinctions (same curvature):")
for curv_idx in range(3):
    curv_name = CURVATURE[curv_idx]
    print(f"\n  For {curv_name} lanes:")

    # regular vs connector
    id1 = curv_idx * 3 + 0
    id2 = curv_idx * 3 + 1
    sim = similarity[id1, id2]
    status = "✅" if sim < 0.7 else ("⚠️" if sim < 0.85 else "🚨")
    print(f"    {status} regular vs connector:    {sim:.4f}")

    # regular vs intersection
    id1 = curv_idx * 3 + 0
    id2 = curv_idx * 3 + 2
    sim = similarity[id1, id2]
    status = "✅" if sim < 0.7 else ("⚠️" if sim < 0.85 else "🚨")
    print(f"    {status} regular vs intersection: {sim:.4f}")

    # connector vs intersection
    id1 = curv_idx * 3 + 1
    id2 = curv_idx * 3 + 2
    sim = similarity[id1, id2]
    status = "✅" if sim < 0.7 else ("⚠️" if sim < 0.85 else "🚨")
    print(f"    {status} connector vs intersection:{sim:.4f}")

# Visualization
print("\n" + "="*80)
print("Generating visualization...")
print("="*80)

labels = [
    "S-Reg", "S-Con", "S-Int",  # Straight
    "L-Reg", "L-Con", "L-Int",  # Left
    "R-Reg", "R-Con", "R-Int"   # Right
]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(similarity, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cosine Similarity'},
            square=True, linewidths=0.5)
ax.set_title('CLIP Embedding Similarity: 9 Data-Driven Templates\n(S=Straight, L=Left, R=Right, Reg=Regular, Con=Connector, Int=Intersection)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
output_path = 'text_template_9_similarity.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Assessment
print("\n" + "="*80)
print("ASSESSMENT")
print("="*80)

assessment_score = 0

# Criterion 1: Mean off-diagonal similarity
if off_diagonal.mean() < 0.7:
    print("✅ Mean similarity < 0.7: EXCELLENT")
    assessment_score += 3
elif off_diagonal.mean() < 0.85:
    print("⚠️  Mean similarity < 0.85: GOOD")
    assessment_score += 2
else:
    print("🚨 Mean similarity >= 0.85: TOO HIGH")
    assessment_score += 1

# Criterion 2: Curvature distinction (critical for topology)
curv_sims = []
for int_idx in range(3):
    for c1 in range(3):
        for c2 in range(c1+1, 3):
            id1 = c1 * 3 + int_idx
            id2 = c2 * 3 + int_idx
            curv_sims.append(similarity[id1, id2])
curv_sims = np.array(curv_sims)

if curv_sims.mean() < 0.70:
    print("✅ Curvature distinction < 0.70: EXCELLENT")
    assessment_score += 3
elif curv_sims.mean() < 0.80:
    print("⚠️  Curvature distinction < 0.80: GOOD")
    assessment_score += 2
else:
    print("🚨 Curvature distinction >= 0.80: WEAK")
    assessment_score += 1

# Criterion 3: Max similarity
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
    verdict = "✅ EXCELLENT"
    recommendation = "Templates are well-separated. Proceed with implementation!"
elif assessment_score >= 5:
    verdict = "⚠️  GOOD"
    recommendation = "Templates are acceptable. Consider adding learnable temperature."
else:
    verdict = "🚨 NEEDS IMPROVEMENT"
    recommendation = "Templates need refinement or use learnable projection."

print(f"{verdict}: {recommendation}")

# Temperature scaling simulation
print("\n" + "="*80)
print("TEMPERATURE SCALING SIMULATION")
print("="*80)

print("\nEffect of temperature on softmax distribution:")
print("(Example: 3 similar templates with similarities [0.85, 0.88, 0.87])")

sims = np.array([0.85, 0.88, 0.87])
temperatures = [1.0, 0.1, 0.05, 0.01]

for temp in temperatures:
    logits = sims / temp
    probs = np.exp(logits) / np.sum(np.exp(logits))
    print(f"\n  Temperature = {temp:.2f}:")
    print(f"    Probabilities: [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")
    print(f"    Max prob: {probs.max():.3f}, Entropy: {-np.sum(probs * np.log(probs + 1e-10)):.3f}")

print("\n" + "="*80)
print("DONE")
print("="*80)
