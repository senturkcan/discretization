import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
import warnings
from math import log2
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# Load the dataset
def load_wdbc_data(filename):

    try:
        # WDBC dataset doesn't have headers this is for creating headers
        column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
        data = pd.read_csv(filename, names=column_names, header=None)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None


# Basic mutual information calculation
def calculate_basic_mi(X, y_encoded):
    unique_x = np.unique(X)
    unique_y = np.unique(y_encoded)
    n_total = len(X)

    #marginals
    p_x = {x_val: np.sum(X == x_val) / n_total for x_val in unique_x}
    p_y = {y_val: np.sum(y_encoded == y_val) / n_total for y_val in unique_y}

    mi = 0.0
    for x_val in unique_x:
        for y_val in unique_y:
            p_xy = np.sum((X == x_val) & (y_encoded == y_val)) / n_total
            if p_xy > 0:
                mi += p_xy * log2(p_xy / (p_x[x_val] * p_y[y_val]))
    return mi


def analyze_mutual_information(data):

    # Get features
    X = data.iloc[:, 2:]  # Features
    y = data.iloc[:, 1]  # Target

    # y is M/B we are converting that to 1 and 0
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # MI from regression
    mi_regression = mutual_info_regression(X, y_encoded)

    # Tested bin sizes
    bin_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

    results = []

    # For each feature
    for i, column in enumerate(X.columns):
        feature = X[column].values
        mi_binned = []

        # Try different bin sizes
        for n_bins in bin_sizes:
            # Binning the columns
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            feature_binned = discretizer.fit_transform(feature.reshape(-1, 1)).flatten()

            # Calculate MI using basic formula for binned columns
            mi_val = calculate_basic_mi(feature_binned, y)
            mi_binned.append(mi_val)

        results.append({
            'column_index': i + 2,
            'column_name': column,
            'mi_regression': mi_regression[i],
            'mi_binned': mi_binned
        })

    return results, bin_sizes


def create_comparison_plots(results, bin_sizes, cols_per_row=4):
    """Histogram for features. Comparing the binned mi with a line with dots and regression mi with a red line."""

    n_features = len(results)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(20, 5 * n_rows))
    fig.suptitle('Mutual Information Comparison: Regression MI vs Binned MI', fontsize=16, y=0.98)

    # Flatten axes for indexing
    if n_rows == 1:
        axes = [axes] if cols_per_row == 1 else axes
    else:
        axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]

        # Plot MI regression as horizontal line
        ax.axhline(y=result['mi_regression'], color='red', linestyle='-',
                   linewidth=2, label=f'MI Regression: {result["mi_regression"]:.4f}')

        # Plot MI with different bin sizes as points and line
        ax.plot(bin_sizes, result['mi_binned'], 'bo-', linewidth=2,
                markersize=6, label='MI with Binning')

        # Add points for each bin size
        for j, (bin_size, mi_val) in enumerate(zip(bin_sizes, result['mi_binned'])):
            ax.annotate(f'{mi_val:.3f}',
                        (bin_size, mi_val),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center', fontsize=8)

        ax.set_xlabel('Number of Bins')
        ax.set_ylabel('Mutual Information')
        ax.set_title(f'Column {result["column_index"]}: {result["column_name"][:15]}...', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks(bin_sizes)

    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()



def print_and_analyze_closest_bins(results, bin_sizes):
    """Find and print bin size closest to regression MI for each feature"""
    closest_bins = []

    for result in results:
        mi_reg = result['mi_regression']
        mi_binned = result['mi_binned']

        # Find smallest difference
        differences = [abs(mi_bin - mi_reg) for mi_bin in mi_binned]
        best_idx = np.argmin(differences)

        info = {
            'column_index': result['column_index'],
            'column_name': result['column_name'],
            'mi_regression': mi_reg,
            'closest_bin_size': bin_sizes[best_idx],
            'closest_mi_value': mi_binned[best_idx],
            'difference': differences[best_idx]
        }

        closest_bins.append(info)

        # Print directly inside the loop
        print(f"Column {info['column_index']:2d}: Best bin size = {info['closest_bin_size']:2d}")

    return closest_bins


def main():
    # Load data
    data = load_wdbc_data('wdbc.data')
    if data is None:
        return

    # Calculate MI
    results, bin_sizes = analyze_mutual_information(data)

    # Find and print closest bin sizes
    closest_bins = print_and_analyze_closest_bins(results, bin_sizes)

    create_comparison_plots(results, bin_sizes)

if __name__ == "__main__":
    main()