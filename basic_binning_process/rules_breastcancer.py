import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')


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


def analyze_bin_methods(data):
    """Analyze different binning methods for each column"""

    # Get all column names except ID (column 0) and target (column 1 for WDBC)
    all_columns = data.columns.tolist()
    feature_columns = [col for i, col in enumerate(all_columns) if i != 0 and i != 1]

    # Binning methods to test
    bin_methods = ['fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']

    # Store results
    results = {}

    print("Analyzing binning methods for each column...")
    print("=" * 60)

    for column in feature_columns:
        column_index = all_columns.index(column)
        feature_data = data[column].values

        # Remove any NaN values
        feature_data = feature_data[~np.isnan(feature_data)]

        results[column_index] = {}

        print(f"Column {column_index:2d} ({column[:20]:<20}): ", end="")

        for method in bin_methods:
            try:
                # Get bin edges using numpy histogram
                bin_edges = np.histogram_bin_edges(feature_data, bins=method)
                n_bins = len(bin_edges) - 1  # Number of bins is edges - 1
                results[column_index][method] = n_bins
                print(f"{method}:{n_bins:2d} ", end="")
            except Exception as e:
                # Some methods might fail for certain data distributions
                results[column_index][method] = 'Error'
                print(f"{method}:Err ", end="")

        print()  # New line after each column

    return results, bin_methods, feature_columns


def create_results_table(results, bin_methods, feature_columns, data):
    """Create a comprehensive results table"""

    # Get column indices
    all_columns = data.columns.tolist()
    column_indices = [all_columns.index(col) for col in feature_columns]

    # Create DataFrame for better visualization
    table_data = []

    for col_idx in sorted(column_indices):
        row = [col_idx]  # Start with column number
        for method in bin_methods:
            row.append(results[col_idx][method])
        table_data.append(row)

    # Create DataFrame
    df_results = pd.DataFrame(table_data, columns=['Column'] + bin_methods)

    return df_results


def display_results_table(df_results):
    """Display results in a nicely formatted table"""

    print("\n" + "=" * 80)
    print("HISTOGRAM BIN SELECTION RESULTS")
    print("=" * 80)
    print("Number of bins selected by different methods for each column")
    print("-" * 80)

    # Display using tabulate for better formatting
    print(tabulate(df_results, headers=df_results.columns, tablefmt='grid', showindex=False))


def create_heatmap(df_results):
    """Create a heatmap visualization of the results"""

    # Prepare data for heatmap (exclude column numbers)
    heatmap_data = df_results.set_index('Column')

    # Convert 'Error' values to NaN for better visualization
    heatmap_data = heatmap_data.replace('Error', np.nan)

    # Convert to numeric
    heatmap_data = heatmap_data.astype(float)

    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(heatmap_data,
                annot=True,
                fmt='.0f',
                cmap='plasma',
                cbar_kws={'label': 'Number of Bins'},
                linewidths=0.5)

    plt.title('Number of Bins by Method and Column', fontsize=16, pad=20)
    plt.xlabel('Binning Method', fontsize=12)
    plt.ylabel('Column Number', fontsize=12)
    plt.tight_layout()
    plt.show()


def analyze_method_statistics(df_results):
    """Analyze statistics for each binning method"""

    print("\n" + "=" * 80)
    print("BINNING METHOD STATISTICS")
    print("=" * 80)

    # Exclude column numbers and convert errors to NaN
    method_data = df_results.drop('Column', axis=1).replace('Error', np.nan).astype(float)

    print(f"{'Method':<10} {'Mean':<8} {'Std':<8} {'Min':<6} {'Max':<6} {'Median':<8}")
    print("-" * 50)

    for method in method_data.columns:
        values = method_data[method].dropna()
        if len(values) > 0:
            print(f"{method:<10} {values.mean():<8.2f} {values.std():<8.2f} "
                  f"{values.min():<6.0f} {values.max():<6.0f} {values.median():<8.2f}")
        else:
            print(f"{method:<10} {'N/A':<8} {'N/A':<8} {'N/A':<6} {'N/A':<6} {'N/A':<8}")


def compare_methods_by_column(df_results):
    """Compare methods for each column"""

    print("\n" + "=" * 80)
    print("METHOD COMPARISON BY COLUMN")
    print("=" * 80)
    print("Showing min, max, and range of bins for each column across all methods")
    print("-" * 80)

    method_cols = [col for col in df_results.columns if col != 'Column']

    print(
        f"{'Column':<8} {'Min Bins':<10} {'Max Bins':<10} {'Range':<8} {'Most Conservative':<15} {'Most Aggressive':<15}")
    print("-" * 80)

    for _, row in df_results.iterrows():
        col_num = row['Column']
        values = []
        method_values = {}

        for method in method_cols:
            if row[method] != 'Error' and pd.notna(row[method]):
                values.append(row[method])
                method_values[method] = row[method]

        if values:
            min_bins = min(values)
            max_bins = max(values)
            range_bins = max_bins - min_bins

            # Find methods with min and max bins
            min_methods = [method for method, val in method_values.items() if val == min_bins]
            max_methods = [method for method, val in method_values.items() if val == max_bins]

            print(f"{col_num:<8} {min_bins:<10.0f} {max_bins:<10.0f} {range_bins:<8.0f} "
                  f"{','.join(min_methods):<15} {','.join(max_methods):<15}")


def main():
    """Main execution function"""
    print("Loading WDBC dataset...")

    # Load data
    data = load_wdbc_data('wdbc.data')

    if data is None:
        print("Could not load dataset. Please check the file path.")
        return

    print(f"Dataset loaded successfully! Shape: {data.shape}")
    print(f"Target variable (column 1 - Diagnosis) distribution:")
    print(data.iloc[:, 1].value_counts())

    # Analyze binning methods
    results, bin_methods, feature_columns = analyze_bin_methods(data)

    # Create results table
    df_results = create_results_table(results, bin_methods, feature_columns, data)

    # Display results
    display_results_table(df_results)

    # Create visualizations
    print("\nCreating heatmap visualization...")
    create_heatmap(df_results)

    # Analyze statistics
    analyze_method_statistics(df_results)

    # Compare methods by column
    compare_methods_by_column(df_results)

    # Save results to CSV
    df_results.to_csv('wdbc_bin_analysis_results.csv', index=False)
    print(f"\nResults saved to 'wdbc_bin_analysis_results.csv'")

    return df_results


if __name__ == "__main__":
    results_df = main()