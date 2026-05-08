#!/usr/bin/env python
"""
Visualize Domain Gap using UMAP or t-SNE

This script creates 2D projections of BEV features from different domains.

Usage:
    # UMAP
    python tools/visualize_domain_gap.py \
        --features features_carla.pkl features_real.pkl \
        --out domain_gap_umap.png \
        --method umap \
        --n-samples 500

    # t-SNE
    python tools/visualize_domain_gap.py \
        --features features_carla.pkl features_real.pkl \
        --out domain_gap_tsne.png \
        --method tsne \
        --n-samples 500

    # Both (side-by-side comparison)
    python tools/visualize_domain_gap.py \
        --features features_carla.pkl features_real.pkl \
        --out domain_gap_comparison.png \
        --method both \
        --n-samples 500
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize domain gap')
    parser.add_argument('--features', nargs='+', required=True,
                       help='pickle files with extracted features')
    parser.add_argument('--out', default='domain_gap.png',
                       help='output figure path')
    parser.add_argument('--method', type=str, default='tsne',
                       choices=['umap', 'tsne', 'both'],
                       help='dimensionality reduction method')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='number of samples per domain (for balancing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')

    # UMAP parameters
    parser.add_argument('--umap-neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                       help='UMAP min_dist parameter')

    # t-SNE parameters
    parser.add_argument('--tsne-perplexity', type=float, default=30.0,
                       help='t-SNE perplexity parameter (5-50)')
    parser.add_argument('--tsne-n-iter', type=int, default=1000,
                       help='t-SNE number of iterations')
    parser.add_argument('--tsne-learning-rate', type=float, default=200.0,
                       help='t-SNE learning rate')

    # Visualization style
    parser.add_argument('--style', type=str, default='publication',
                       choices=['scatter', 'density', 'contour', 'publication'],
                       help='visualization style')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 10],
                       help='figure size (width height)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='output DPI')
    parser.add_argument('--no-title', action='store_true',
                       help='remove title from figure')
    parser.add_argument('--annotate-clusters', action='store_true',
                       help='add text annotations near clusters')

    args = parser.parse_args()
    return args


def load_features(feature_files, n_samples_per_domain=None, seed=42):
    """Load and optionally subsample features from multiple files"""
    all_features = []
    all_domains = []
    domain_labels = []

    np.random.seed(seed)

    for fpath in feature_files:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        features = data['features']  # [N, C]
        domain = data['domain']

        # Subsample if requested
        if n_samples_per_domain is not None and features.shape[0] > n_samples_per_domain:
            indices = np.random.choice(features.shape[0], n_samples_per_domain, replace=False)
            features = features[indices]

        all_features.append(features)
        all_domains.extend([domain] * features.shape[0])
        domain_labels.append(domain)

        print(f"Loaded {features.shape[0]} samples from {domain} domain")

    # Concatenate
    all_features = np.concatenate(all_features, axis=0)  # [N_total, C]
    all_domains = np.array(all_domains)

    print(f"\nTotal samples: {all_features.shape[0]}")
    print(f"Feature dim: {all_features.shape[1]}")
    print(f"Domains: {set(all_domains)}")

    return all_features, all_domains, domain_labels


def apply_umap(features, args):
    """Apply UMAP dimensionality reduction"""
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")

    print(f"\nApplying UMAP...")
    print(f"  n_neighbors: {args.umap_neighbors}")
    print(f"  min_dist: {args.umap_min_dist}")

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        random_state=args.seed,
        verbose=True
    )
    embedding = reducer.fit_transform(features)
    print(f"  Embedding shape: {embedding.shape}")
    return embedding


def apply_tsne(features, args):
    """Apply t-SNE dimensionality reduction"""
    print(f"\nApplying t-SNE...")
    print(f"  perplexity: {args.tsne_perplexity}")
    print(f"  n_iter: {args.tsne_n_iter}")
    print(f"  learning_rate: {args.tsne_learning_rate}")

    reducer = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_n_iter,
        learning_rate=args.tsne_learning_rate,
        random_state=args.seed,
        verbose=1
    )
    embedding = reducer.fit_transform(features)
    print(f"  Embedding shape: {embedding.shape}")
    return embedding


def plot_embedding(embedding, domains, ax, title, domain_colors, domain_labels_map, style='publication',
                  show_title=True, annotate_clusters=False):
    """Plot a 2D embedding with domain coloring

    Args:
        style: 'scatter', 'density', 'contour', 'publication'
        show_title: whether to show title
        annotate_clusters: whether to add text annotations near clusters
    """

    if style == 'scatter':
        # Simple scatter plot
        for domain in set(domains):
            mask = domains == domain
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=domain_colors.get(domain, '#95A5A6'),
                label=domain_labels_map.get(domain, domain.capitalize()),
                alpha=0.4,
                s=30,
                edgecolors='none'
            )

    elif style == 'density':
        # Density-based coloring with larger points
        for domain in set(domains):
            mask = domains == domain
            points = embedding[mask]

            # Compute density
            try:
                xy = points.T
                z = gaussian_kde(xy)(xy)
            except:
                z = np.ones(points.shape[0])

            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=z,
                cmap='viridis' if 'synthetic' in domain else 'plasma' if 'real' in domain else 'cividis',
                label=domain_labels_map.get(domain, domain.capitalize()),
                alpha=0.6,
                s=40,
                edgecolors='white',
                linewidths=0.3
            )

    elif style == 'contour':
        # High-quality contour plot with organic-shaped backgrounds
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.colors as mcolors
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon

        # Plot backgrounds and contours for each domain
        for domain in sorted(set(domains)):
            mask = domains == domain
            points = embedding[mask]

            # Plot KDE contours
            try:
                x = points[:, 0]
                y = points[:, 1]

                # High-resolution grid for smooth contours
                xi = np.linspace(x.min() - 2, x.max() + 2, 300)
                yi = np.linspace(y.min() - 2, y.max() + 2, 300)
                Xi, Yi = np.meshgrid(xi, yi)

                # KDE with bandwidth adjustment for smoother contours
                kde = gaussian_kde(points.T, bw_method=0.15)
                zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

                # Get base color
                base_color = domain_colors.get(domain, '#95A5A6')

                # Use threshold to create organic-shaped background
                # Only fill areas with density above threshold (removes rectangular edges)
                threshold = zi.max() * 0.05  # Only show areas with >5% of max density

                # Filled contours with threshold (organic shape)
                levels_fill = np.linspace(threshold, zi.max(), 10)
                contourf = ax.contourf(
                    Xi, Yi, zi,
                    levels=levels_fill,
                    colors=[base_color],
                    alpha=0.15,
                    antialiased=True,
                    extend='neither'  # Don't extend beyond specified levels
                )

                # Contour lines (prominent)
                levels_line = np.linspace(zi.max() * 0.15, zi.max(), 5)
                contour = ax.contour(
                    Xi, Yi, zi,
                    levels=levels_line,
                    colors=[base_color],
                    alpha=0.7,
                    linewidths=2.5,
                    linestyles='solid',
                    antialiased=True
                )

            except Exception as e:
                print(f"Warning: Could not create contours for {domain}: {e}")

        # Scatter overlay on top (smaller, more transparent for ECCV style)
        for domain in sorted(set(domains)):
            mask = domains == domain
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=domain_colors.get(domain, '#95A5A6'),
                label=domain_labels_map.get(domain, domain.capitalize()),
                alpha=0.4,
                s=18,
                edgecolors='white',
                linewidths=0.3,
                zorder=10,  # Ensure points are on top
                rasterized=True
            )

    elif style == 'publication':
        # ECCV-style: smaller points, lower alpha, clean appearance
        # Plot in reverse order so first domain appears on top
        for domain in reversed(sorted(set(domains))):
            mask = domains == domain
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=domain_colors.get(domain, '#95A5A6'),
                label=domain_labels_map.get(domain, domain.capitalize()),
                alpha=0.45,
                s=30,
                edgecolors='white',
                linewidths=0.4,
                rasterized=True  # For better PDF rendering
            )

    # Remove axes and ticks - completely clean
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Set white background
    ax.set_facecolor('white')

    # ECCV-style legend: clean and readable
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        framealpha=0.98,
        edgecolor='#BBBBBB',
        facecolor='white',
        fontsize=13,
        markerscale=2.2,
        fancybox=False,
        shadow=False,
        labelspacing=0.8,
        borderpad=0.9,
        handletextpad=0.6,
        borderaxespad=0.5
    )
    legend.get_frame().set_linewidth(1.0)

    # Optional cluster annotations
    if annotate_clusters:
        # Add small text labels near cluster centers
        for domain in set(domains):
            mask = domains == domain
            points = embedding[mask]
            # Compute centroid
            centroid = points.mean(axis=0)

            # Short label (just domain name)
            short_label = domain.capitalize()

            ax.text(
                centroid[0], centroid[1],
                short_label,
                fontsize=11,
                fontweight='500',
                color=domain_colors.get(domain, '#95A5A6'),
                alpha=0.6,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.8)
            )

    # Title (optional)
    if show_title:
        ax.set_title(title, fontsize=16, fontweight='500', pad=20, family='sans-serif')


def main():
    args = parse_args()

    # Set publication-quality matplotlib settings
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13,
        'figure.dpi': args.dpi,
        'savefig.dpi': args.dpi,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,  # TrueType fonts for PDF
        'ps.fonttype': 42,
    })

    print(f"\n{'='*80}")
    print(f"Domain Gap Visualization ({args.method.upper()}, {args.style} style)")
    print(f"{'='*80}\n")

    # Load features
    features, domains, domain_labels = load_features(
        args.features,
        n_samples_per_domain=args.n_samples,
        seed=args.seed
    )

    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Define publication-quality colors (colorblind-friendly palette)
    domain_colors = {
        'synthetic': '#D62728',    # Strong red
        'real': '#1F77B4',          # Strong blue
        'adapted': '#2CA02C',       # Strong green
        'finetuned': '#2CA02C'      # Alias for adapted
    }

    domain_labels_map = {
        'synthetic': 'Simulation (CARLA)',
        'real': 'Real (before fine-tuning)',
        'adapted': 'Real (after fine-tuning)',
        'finetuned': 'Real (after fine-tuning)'
    }

    # Apply dimensionality reduction and plot
    if args.method == 'umap':
        embedding = apply_umap(features_normalized, args)

        fig, ax = plt.subplots(figsize=tuple(args.figsize), facecolor='white')
        plot_embedding(embedding, domains, ax,
                      'Feature-Space Comparison Between Simulation and Real Domains',
                      domain_colors, domain_labels_map, style=args.style,
                      show_title=not args.no_title,
                      annotate_clusters=args.annotate_clusters)

    elif args.method == 'tsne':
        embedding = apply_tsne(features_normalized, args)

        fig, ax = plt.subplots(figsize=tuple(args.figsize), facecolor='white')
        plot_embedding(embedding, domains, ax,
                      'Feature-Space Comparison Between Simulation and Real Domains',
                      domain_colors, domain_labels_map, style=args.style,
                      show_title=not args.no_title,
                      annotate_clusters=args.annotate_clusters)

    elif args.method == 'both':
        embedding_umap = apply_umap(features_normalized, args)
        embedding_tsne = apply_tsne(features_normalized, args)

        fig, axes = plt.subplots(1, 2, figsize=(args.figsize[0]*1.8, args.figsize[1]), facecolor='white')

        plot_embedding(embedding_umap, domains, axes[0],
                      'UMAP Projection',
                      domain_colors, domain_labels_map, style=args.style,
                      show_title=not args.no_title,
                      annotate_clusters=args.annotate_clusters)

        plot_embedding(embedding_tsne, domains, axes[1],
                      't-SNE Projection',
                      domain_colors, domain_labels_map, style=args.style,
                      show_title=not args.no_title,
                      annotate_clusters=args.annotate_clusters)

    plt.tight_layout()

    # Save
    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\n{'='*80}")
    print(f"Saved to: {args.out}")
    print(f"  DPI: {args.dpi}")
    print(f"  Style: {args.style}")
    print(f"  Size: {args.figsize[0]}x{args.figsize[1]} inches")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
