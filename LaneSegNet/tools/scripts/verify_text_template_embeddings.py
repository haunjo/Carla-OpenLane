#!/usr/bin/env python3
"""
Verify CLIP text template embeddings are well-separated in embedding space.
"""
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

# Text templates (Version 1: Action-based)
templates_v1 = [
    "lane continuing straight to next segment",      # 0
    "lane turning left to next segment",             # 1
    "lane turning right to next segment",            # 2
    "lane diverging into multiple branches",         # 3
    "lane converging from multiple sources",         # 4
    "lane terminating without continuation"          # 5
]

# Alternative Version 2: Structure-based
templates_v2 = [
    "straight lane segment connecting forward",           # 0
    "curved lane segment connecting left turn",           # 1
    "curved lane segment connecting right turn",          # 2
    "branching lane segment splitting into paths",        # 3
    "merging lane segment joining from paths",            # 4
    "terminal lane segment ending at boundary"            # 5
]

# Simple baseline for comparison
templates_baseline = [
    "straight lane",
    "left turn lane",
    "right turn lane",
    "fork lane",
    "merge lane",
    "end lane"
]

def compute_similarity_matrix(templates, name):
    """Compute pairwise cosine similarity matrix."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"{'='*80}")

    # Tokenize and encode
    texts = clip.tokenize(templates).to(device)
    with torch.no_grad():
        text_features = model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute similarity matrix
    similarity = (text_features @ text_features.T).cpu().numpy()

    # Print similarity matrix
    print(f"\nSimilarity Matrix:")
    print("=" * 80)

    # Header
    labels = ["straight", "left", "right", "fork", "merge", "term"]
    header = "         " + "  ".join([f"{l:>8s}" for l in labels])
    print(header)
    print("-" * 80)

    for i, label in enumerate(labels):
        row_str = f"{label:>8s} "
        for j in range(len(templates)):
            if i == j:
                row_str += f"  {similarity[i,j]:>8.2f}"  # Diagonal (should be 1.0)
            else:
                # Color code: <0.5 good, 0.5-0.7 ok, >0.7 too similar
                sim_val = similarity[i,j]
                if sim_val < 0.5:
                    row_str += f"  \033[92m{sim_val:>8.2f}\033[0m"  # Green
                elif sim_val < 0.7:
                    row_str += f"  \033[93m{sim_val:>8.2f}\033[0m"  # Yellow
                else:
                    row_str += f"  \033[91m{sim_val:>8.2f}\033[0m"  # Red
        print(row_str)

    # Statistics
    print("\n" + "=" * 80)
    print("Statistics (excluding diagonal):")
    off_diagonal = similarity[np.triu_indices_from(similarity, k=1)]
    print(f"  Min similarity:  {off_diagonal.min():.4f}")
    print(f"  Max similarity:  {off_diagonal.max():.4f}")
    print(f"  Mean similarity: {off_diagonal.mean():.4f}")
    print(f"  Std similarity:  {off_diagonal.std():.4f}")

    # Critical pairs
    print("\n" + "=" * 80)
    print("Critical Pair Analysis:")
    print("-" * 80)

    critical_pairs = [
        (0, 1, "straight vs left"),
        (0, 2, "straight vs right"),
        (1, 2, "left vs right"),
        (3, 4, "fork vs merge"),
        (3, 0, "fork vs straight"),
        (4, 0, "merge vs straight"),
        (5, 0, "term vs straight"),
    ]

    for i, j, desc in critical_pairs:
        sim_val = similarity[i, j]
        status = "✅ Good" if sim_val < 0.7 else "⚠️  High"
        print(f"  {desc:25s}: {sim_val:.4f}  {status}")

    return similarity

# Compute for all versions
sim_v1 = compute_similarity_matrix(templates_v1, "Version 1: Action-based")
sim_v2 = compute_similarity_matrix(templates_v2, "Version 2: Structure-based")
sim_baseline = compute_similarity_matrix(templates_baseline, "Baseline: Simple")

# Visualization
print("\n" + "="*80)
print("Generating visualization...")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

labels = ["straight", "left", "right", "fork", "merge", "term"]

# Version 1
sns.heatmap(sim_v1, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Cosine Similarity'})
axes[0].set_title('Version 1: Action-based', fontsize=12, fontweight='bold')

# Version 2
sns.heatmap(sim_v2, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Cosine Similarity'})
axes[1].set_title('Version 2: Structure-based', fontsize=12, fontweight='bold')

# Baseline
sns.heatmap(sim_baseline, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, ax=axes[2], cbar_kws={'label': 'Cosine Similarity'})
axes[2].set_title('Baseline: Simple', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path = 'text_template_similarity_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Compare off-diagonal stats
off_diag_v1 = sim_v1[np.triu_indices_from(sim_v1, k=1)]
off_diag_v2 = sim_v2[np.triu_indices_from(sim_v2, k=1)]
off_diag_baseline = sim_baseline[np.triu_indices_from(sim_baseline, k=1)]

scores = {
    "Version 1 (Action-based)": {
        "mean_sim": off_diag_v1.mean(),
        "max_sim": off_diag_v1.max(),
        "score": (1 - off_diag_v1.mean()) * 100 - off_diag_v1.max() * 50
    },
    "Version 2 (Structure-based)": {
        "mean_sim": off_diag_v2.mean(),
        "max_sim": off_diag_v2.max(),
        "score": (1 - off_diag_v2.mean()) * 100 - off_diag_v2.max() * 50
    },
    "Baseline (Simple)": {
        "mean_sim": off_diag_baseline.mean(),
        "max_sim": off_diag_baseline.max(),
        "score": (1 - off_diag_baseline.mean()) * 100 - off_diag_baseline.max() * 50
    }
}

# Sort by score (lower mean similarity and max similarity is better)
sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)

print("\nRanking (better separation = higher score):")
print("-" * 80)
for rank, (name, stats) in enumerate(sorted_scores, 1):
    print(f"{rank}. {name}")
    print(f"   Mean similarity: {stats['mean_sim']:.4f}")
    print(f"   Max similarity:  {stats['max_sim']:.4f}")
    print(f"   Score:           {stats['score']:.2f}")
    print()

print("="*80)
print(f"✅ BEST: {sorted_scores[0][0]}")
print("="*80)

# Print best templates
if sorted_scores[0][0] == "Version 1 (Action-based)":
    best_templates = templates_v1
elif sorted_scores[0][0] == "Version 2 (Structure-based)":
    best_templates = templates_v2
else:
    best_templates = templates_baseline

print("\nRecommended Text Templates:")
print("-" * 80)
for i, template in enumerate(best_templates):
    print(f"{i}. \"{template}\"")

print("\n" + "="*80)
print("DONE")
print("="*80)
