import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score
import seaborn as sns
from math import log2
import warnings

warnings.filterwarnings('ignore')


# Load the dataset
def load_wdbc_data(filename):
    """Load WDBC dataset"""
    try:
        # WDBC dataset doesn't have headers, so we'll create them
        column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
        data = pd.read_csv(filename, names=column_names, header=None)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found. Please make sure the file exists.")
        return None


# Basic mutual information calculation
def calculate_basic_mi(X, y):
    """Calculate mutual information using basic formula"""
    # Create contingency table
    unique_x = np.unique(X)
    unique_y = np.unique(y)

    mi = 0.0
    n_total = len(X)

    for x_val in unique_x:
        for y_val in unique_y:
            # Joint probability
            p_xy = np.sum((X == x_val) & (y == y_val)) / n_total

            if p_xy > 0:
                # Marginal probabilities
                p_x = np.sum(X == x_val) / n_total
                p_y = np.sum(y == y_val) / n_total

                # MI contribution
                mi += p_xy * log2(p_xy / (p_x * p_y))

    return mi


def analyze_mutual_information(data):
    """Perform comprehensive mutual information analysis"""

    # Prepare data
    X = data.iloc[:, 2:]  # Features (columns 2 onwards)
    y_categorical = data.iloc[:, 1]  # Target (column 1)

    # Encode target variable for regression MI
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_categorical)

    # Calculate MI using mutual_info_regression
    mi_regression = mutual_info_regression(X, y_encoded, random_state=42)

    # Bin sizes to test
    bin_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12,14,16,18,20,24,28,32,36,40,44]

    # Store results
    results = []

    # Calculate MI for each feature with different binning strategies
    for i, column in enumerate(X.columns):
        feature_data = X[column].values

        # MI with different bin sizes
        mi_binned = []

        for n_bins in bin_sizes:
            # Discretize the feature
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
            feature_binned = discretizer.fit_transform(feature_data.reshape(-1, 1)).flatten()

            # Calculate basic MI
            mi_basic = mutual_info_score(feature_binned, y_categorical)
            mi_binned.append(mi_basic)

        results.append({
            'column_index': i + 2,  # Actual column index (starting from 2)
            'column_name': column,
            'mi_regression': mi_regression[i],
            'mi_binned': mi_binned
        })

    return results, bin_sizes


def create_comparison_plots(results, bin_sizes, cols_per_row=4):
    """Create comparison plots for all features"""

    n_features = len(results)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(20, 5 * n_rows))
    fig.suptitle('Mutual Information Comparison: Regression MI vs Binned MI', fontsize=16, y=0.98)

    # Flatten axes for easier indexing
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


def create_summary_statistics(results, bin_sizes):
    """Create summary statistics"""
    print("=" * 80)
    print("MUTUAL INFORMATION ANALYSIS SUMMARY")
    print("=" * 80)

    # Overall statistics
    mi_regression_values = [r['mi_regression'] for r in results]
    print(f"Dataset shape: {len(results)} features")
    print(f"MI Regression - Mean: {np.mean(mi_regression_values):.4f}, Std: {np.std(mi_regression_values):.4f}")
    print(f"MI Regression - Min: {np.min(mi_regression_values):.4f}, Max: {np.max(mi_regression_values):.4f}")

    print("\n" + "=" * 80)
    print("TOP 10 FEATURES BY MI REGRESSION")
    print("=" * 80)

    # Sort by MI regression value
    sorted_results = sorted(results, key=lambda x: x['mi_regression'], reverse=True)

    for i, result in enumerate(sorted_results[:10]):
        print(
            f"{i + 1:2d}. Column {result['column_index']:2d} ({result['column_name'][:20]:<20}): {result['mi_regression']:.4f}")

    print("\n" + "=" * 80)
    print("BINNING ANALYSIS SUMMARY")
    print("=" * 80)

    # Analyze how binning affects MI
    for bin_size in bin_sizes:
        bin_mi_values = [r['mi_binned'][bin_sizes.index(bin_size)] for r in results]
        correlation = np.corrcoef([r['mi_regression'] for r in results], bin_mi_values)[0, 1]
        print(f"Bins={bin_size:2d}: Mean MI={np.mean(bin_mi_values):.4f}, "
              f"Correlation with MI_regression={correlation:.4f}")


def main():
    """Main execution function"""
    print("Loading WDBC dataset...")

    # Load data
    data = load_wdbc_data('wdbc.data')

    if data is None:
        print("Could not load dataset. Please check the file path.")
        return

    print(f"Dataset loaded successfully! Shape: {data.shape}")
    print(f"Target variable distribution:")
    print(data.iloc[:, 1].value_counts())

    # Perform analysis
    print("\nCalculating mutual information...")
    results, bin_sizes = analyze_mutual_information(data)

    # Create visualizations
    print("Creating comparison plots...")
    create_comparison_plots(results, bin_sizes)

    # Show summary statistics
    create_summary_statistics(results, bin_sizes)

    # Additional analysis: Best bin size for each feature
    print("\n" + "=" * 80)
    print("OPTIMAL BIN SIZE ANALYSIS")
    print("=" * 80)

    for result in results[:5]:  # Show first 5 features
        best_bin_idx = np.argmax(result['mi_binned'])
        best_bin_size = bin_sizes[best_bin_idx]
        best_mi = result['mi_binned'][best_bin_idx]

        print(f"Column {result['column_index']:2d}: Best bin size = {best_bin_size}, "
              f"MI = {best_mi:.4f} (vs Regression MI = {result['mi_regression']:.4f})")


if __name__ == "__main__":
    main()