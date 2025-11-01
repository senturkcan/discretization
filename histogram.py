import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Load dataset
df = pd.read_csv("wdbc.data", delimiter=",")

# Encode target variable (M/B to 0/1)
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, 1])

# Get column 2 data
col_idx = 2
X = df.iloc[:, [col_idx]].values
X_1d = X.flatten()

# Calculate initial mutual information
initial_mi = mutual_info_classif(X, y, random_state=42)[0]

print(f"Column {col_idx} statistics:")
print(f"Min: {X_1d.min():.4f}, Max: {X_1d.max():.4f}")
print(f"Mean: {X_1d.mean():.4f}, Std: {X_1d.std():.4f}")
print(f"Initial MI: {initial_mi:.4f}")

# Test different k-means binning configurations
bin_sizes_to_test = [4, 8, 16, 32, 64, 100, 140, 160, 180, 200]

# Create subplots - need more space for 10 plots + original
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

# Plot original data
counts_orig, _, patches_orig = axes[0].hist(X_1d, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
axes[0].set_title(f'Original Data (Column {col_idx})\nMI: {initial_mi:.4f}')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Count')
axes[0].grid(True, alpha=0.3)

# Test each k-means binning configuration
for i, n_bins in enumerate(bin_sizes_to_test, 1):
    # Skip if we run out of subplot spaces
    if i >= len(axes):
        break

    # Create k-means discretizer
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                   strategy="kmeans", random_state=42)

    # Fit and transform
    binned_X = discretizer.fit_transform(X)

    # Calculate MI for binned data
    binned_mi = mutual_info_classif(binned_X, y, random_state=42)[0]

    # Get bin edges
    bin_edges = discretizer.bin_edges_[0]

    # Plot histogram using the actual bin edges from KBinsDiscretizer
    counts, _, patches = axes[i].hist(X_1d, bins=bin_edges, alpha=0.7, color='lightblue',
                                      edgecolor='black', density=False)

    # Add vertical lines for bin boundaries with numerical labels
    for j, edge in enumerate(bin_edges):
        axes[i].axvline(x=edge, color='red', linestyle='--', linewidth=0.01, alpha=0.8)
        # Add text labels for bin boundaries (only for smaller bin counts to avoid clutter)
        if n_bins <= 32:
            axes[i].text(edge, axes[i].get_ylim()[1] * 0.9, f'{edge:.2f}',
                         rotation=90, ha='right', va='top', fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

    # Color each bin region with different colors
    for j, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Set3(j % 12))
        patch.set_alpha(0.7)

    # Set title and labels
    title = f'K-means - {n_bins} bins\nMI: {binned_mi:.4f} (Δ: {binned_mi - initial_mi:+.4f})'

    axes[i].set_title(title)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].grid(True, alpha=0.3)

    # Print bin information
    print(f"\nK-means - {n_bins} bins:")
    print(f"MI: {binned_mi:.4f} (improvement: {binned_mi - initial_mi:+.4f})")
    print("Bin edges:")
    for j, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        count = np.sum((X_1d >= start) & (X_1d < end if j < len(bin_edges) - 2 else X_1d <= end))
        print(f"  Bin {j + 1}: [{start:.4f}, {end:.4f}{'(' if j < len(bin_edges) - 2 else ']'} - {count} samples")

# Hide any unused subplots
for j in range(len(bin_sizes_to_test) + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# Save the figure
plt.savefig(f'column_{col_idx}_kmeans_binning_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved as 'column_{col_idx}_kmeans_binning_analysis.png'")

plt.show()

# Show detailed comparison
print(f"\n{'=' * 60}")
print("SUMMARY:")
print(f"Original MI: {initial_mi:.4f}")
print("Best improvements")
print("Colored regions show how the data is divided into discrete bins")