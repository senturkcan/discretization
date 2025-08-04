import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# Load the dataset
def load_parkinsons_data(filename):
    try:
        # Parkinsons dataset already has headers
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None


# Basic mutual information calculation
def calculate_basic_mi(X, y_encoded):
    unique_x = np.unique(X)
    unique_y = np.unique(y_encoded)
    n_total = len(X)

    # marginals
    p_x = {x_val: np.sum(X == x_val) / n_total for x_val in unique_x}
    p_y = {y_val: np.sum(y_encoded == y_val) / n_total for y_val in unique_y}

    mi = 0.0
    for x_val in unique_x:
        for y_val in unique_y:
            p_xy = np.sum((X == x_val) & (y_encoded == y_val)) / n_total
            if p_xy > 0:
                # to avoid division by zero
                marginal_product = p_x[x_val] * p_y[y_val]
                if marginal_product > 0:
                    mi += p_xy * np.log2(p_xy / marginal_product)
    return mi


def analyze_mutual_information(data):
    # Remove ID column (column 0) and target column (column 17)
    feature_columns = [col for i, col in enumerate(data.columns) if i != 0 and i != 17]
    X = data[feature_columns]  # Features (excluding ID and target)
    y = data.iloc[:, 17]  # Target (column 17)

    # Target is already 0 and 1, so no need for label encoding
    y_encoded = y.values

    # Print debug information
    print(f"Target column name: {data.columns[17]}")
    print(f"Target unique values: {np.unique(y_encoded)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns[:5]}...")  # Show first 5 feature names

    # MI from regression
    mi_regression = mutual_info_regression(X, y_encoded)

    # Tested bin sizes
    bin_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20,22, 24,26, 28,30, 32,34, 40, 46, 52, 64, 76,88,100]

    results = []

    # For each feature
    for i, column in enumerate(X.columns):
        feature = X[column].values
        mi_binned = []

        # Check number of unique values for this feature
        n_unique = len(np.unique(feature))

        # Try different bin sizes
        for n_bins in bin_sizes:
            try:
                # Only calculate MI if we have enough unique values
                if n_unique < n_bins:
                    # Skip this bin size and use NaN to maintain array structure
                    mi_binned.append(np.nan)
                else:
                    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
                    feature_binned = discretizer.fit_transform(feature.reshape(-1, 1)).flatten()

                    # Calculate MI using basic formula for binned columns
                    mi_val = calculate_basic_mi(feature_binned, y_encoded)
                    mi_binned.append(mi_val)
            except ValueError as e:
                # Handle cases where binning fails
                print(f"Warning: Binning failed for feature {column} with {n_bins} bins: {e}")
                mi_binned.append(np.nan)

        results.append({
            'column_index': data.columns.get_loc(column),
            'column_name': column,
            'mi_regression': mi_regression[i],
            'mi_binned': mi_binned,
            'n_unique': n_unique
        })

    return results, bin_sizes


def create_comparison_plots(results, bin_sizes, cols_per_row=4):
    """Histogram for features. Comparing the binned mi with a line with dots and regression mi with a red line."""

    n_features = len(results)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(24, 7 * n_rows))
    fig.suptitle('Regression MI vs Binned MI - Parkinsons Dataset', fontsize=14, y=0.99)

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

        # Filter out NaN values for plotting
        valid_indices = [j for j, val in enumerate(result['mi_binned']) if not np.isnan(val)]
        valid_bin_sizes = [bin_sizes[j] for j in valid_indices]
        valid_mi_values = [result['mi_binned'][j] for j in valid_indices]

        # Plot MI with different bin sizes as points and line (only valid values)
        if valid_mi_values:  # Only plot if we have valid values
            ax.plot(valid_bin_sizes, valid_mi_values, 'bo-', linewidth=1.5,
                    markersize=4, label='MI with Binning')

            # Add annotations for valid points only with smaller font and better positioning
            for bin_size, mi_val in zip(valid_bin_sizes, valid_mi_values):
                ax.annotate(f'{mi_val:.3f}',
                            (bin_size, mi_val),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha='center', fontsize=6)

        ax.set_xlabel('# of Bins', fontsize=8)
        ax.set_ylabel('MI', fontsize=8)
        ax.set_title(f'Col {result["column_index"]}: {result["column_name"][:12]}...\n(n_unique={result["n_unique"]})',
                     fontsize=8, pad=35)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_xticks(bin_sizes)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='x', rotation=45)

    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


def print_and_analyze_closest_bins(results, bin_sizes):
    """Find and print bin size closest to regression MI for each feature"""
    closest_bins = []

    print("\nClosest bin sizes for each feature:")
    print("=" * 50)

    for result in results:
        mi_reg = result['mi_regression']
        mi_binned = result['mi_binned']

        # Filter out NaN values
        valid_mi_values = [val for val in mi_binned if not np.isnan(val)]
        valid_bin_indices = [i for i, val in enumerate(mi_binned) if not np.isnan(val)]

        if not valid_mi_values:
            # If no valid values, skip this feature
            print(f"Column {result['column_index']:2d} ({result['column_name'][:20]:20s}): "
                  f"No valid bin sizes (n_unique={result['n_unique']})")
            continue

        # Find smallest difference among valid values
        differences = [abs(mi_bin - mi_reg) for mi_bin in valid_mi_values]
        best_idx = np.argmin(differences)
        best_bin_idx = valid_bin_indices[best_idx]

        info = {
            'column_index': result['column_index'],
            'column_name': result['column_name'],
            'mi_regression': mi_reg,
            'closest_bin_size': bin_sizes[best_bin_idx],
            'closest_mi_value': valid_mi_values[best_idx],
            'difference': differences[best_idx],
            'n_unique': result['n_unique']
        }

        closest_bins.append(info)

        # Print with more detailed information
        print(f"Column {info['column_index']:2d} ({info['column_name'][:20]:20s}): "
              f"Best bin size = {info['closest_bin_size']:2d}, "
              f"MI_reg = {info['mi_regression']:.4f}, "
              f"MI_bin = {info['closest_mi_value']:.4f}, "
              f"Diff = {info['difference']:.4f}, "
              f"n_unique = {info['n_unique']}")

    return closest_bins


def main():
    # Load data
    filename = 'parkinsons_plus.data'
    data = load_parkinsons_data(filename)
    if data is None:
        return

    print(f"Data loaded successfully. Shape: {data.shape}")
    print(f"Column names: {list(data.columns)}")
    print(f"Target distribution: {data.iloc[:, 17].value_counts()}")

    # Calculate MI
    results, bin_sizes = analyze_mutual_information(data)

    # Find and print closest bin sizes
    closest_bins = print_and_analyze_closest_bins(results, bin_sizes)

    create_comparison_plots(results, bin_sizes)


if __name__ == "__main__":
    main()