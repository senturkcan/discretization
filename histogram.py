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
bin_sizes_to_test = [4, 8, 16, 32, 64]

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Plot original data
axes[0].hist(X_1d, bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
axes[0].set_title(f'Original Data (Column {col_idx})\nMI: {initial_mi:.4f}')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].grid(True, alpha=0.3)

# Test each k-means binning configuration
for i, n_bins in enumerate(bin_sizes_to_test, 1):
    # Create k-means discretizer
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                   strategy="kmeans", random_state=42)

    # Fit and transform
    binned_X = discretizer.fit_transform(X)

    # Calculate MI for binned data
    binned_mi = mutual_info_classif(binned_X, y, random_state=42)[0]

    # Get bin edges
    bin_edges = discretizer.bin_edges_[0]

    # Plot histogram with bin boundaries
    axes[i].hist(X_1d, bins=30, alpha=0.5, color='lightblue', edgecolor='black',
                 density=True, label='Original Data')

    # Add vertical lines for bin boundaries with numerical labels
    for j, edge in enumerate(bin_edges):
        axes[i].axvline(x=edge, color='red', linestyle='--', linewidth=2, alpha=0.8)
        # Add text labels for bin boundaries
        axes[i].text(edge, axes[i].get_ylim()[1] * 0.9, f'{edge:.3f}',
                     rotation=90, ha='right', va='top', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Color each bin region
    for j in range(len(bin_edges) - 1):
        axes[i].axvspan(bin_edges[j], bin_edges[j + 1], alpha=0.2,
                        color=plt.cm.Set3(j % 12), label=f'Bin {j + 1}' if j < 5 else "")

    # Set title and labels
    title = f'K-means - {n_bins} bins\nMI: {binned_mi:.4f} (Δ: {binned_mi - initial_mi:+.4f})'

    axes[i].set_title(title)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].grid(True, alpha=0.3)

    # Print bin information
    print(f"\nK-means - {n_bins} bins:")
    print(f"MI: {binned_mi:.4f} (improvement: {binned_mi - initial_mi:+.4f})")
    print("Bin edges:")
    for j, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        count = np.sum((X_1d >= start) & (X_1d < end if j < len(bin_edges) - 2 else X_1d <= end))
        print(f"  Bin {j + 1}: [{start:.4f}, {end:.4f}{'(' if j < len(bin_edges) - 2 else ']'} - {count} samples")

# Save the figure
plt.savefig(f'column_{col_idx}_kmeans_binning_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved as 'column_{col_idx}_kmeans_binning_analysis.png'")

plt.show()

# Show detailed comparison
print(f"\n{'=' * 60}")
print("SUMMARY:")
print(f"Original MI: {initial_mi:.4f}")
print("Best improvements shown above - red dashed lines show bin boundaries")
print("Colored regions show how the data is divided into discrete bins")