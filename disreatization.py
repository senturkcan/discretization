#Guideline:
#select dataset
#find mutual information
#apply KBinsDiscretizer from scikitlearn
#find the new mutual information
#make it dynamic so it can be applied for a lot of variations of binning

#improvments:
#make it for all columns
#make it for all possiable bin sizes (it will requaire checking best combinations automatically)
#import
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import os
#mutual_info_classif for discrete target variable
#sklearn.feature_selection.mutual_info_regression for continuous target variable

#read dataset
file_path = "wdbc.data"
df = pd.read_csv(file_path, delimiter=",")

#selecting features and calculate mi
feature_columns = [2]  # adjust as needed
X = df.iloc[:, feature_columns]
y = df.iloc[:, 1]

initial_mi_scores = mutual_info_classif(X=X, y=y)
print(f"the initial mi score is {initial_mi_scores}")

#binning test area
#possiable values
possible_bin_sizes = [4, 8, 16, 32, 64, 100, 140, 160, 180, 200]
possible_strategies = ["uniform", "quantile", "kmeans"]
quantile_methods = [
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
    "interpolated_inverted_cdf", "hazen", "weibull", "linear",
    "median_unbiased", "normal_unbiased"
]


# Create directory for results if it doesn't exist
results_dir = "binning_results"
os.makedirs(results_dir, exist_ok=True)

# Process each column from 2 to 31
for column_idx in range(2, 32):  # columns 2 to 31
    print(f"\nProcessing Column {column_idx}...")

    # Select current feature
    X = df.iloc[:, [column_idx]]

    # Calculate initial mutual information
    initial_mi_scores = mutual_info_classif(X=X, y=y)
    initial_mi = initial_mi_scores[0]
    print(f"Column {column_idx} - Initial MI score: {initial_mi:.6f}")

    # Create results storage for current column
    results = []

    # Test all combinations
    for n_bins in possible_bin_sizes:
        for strategy in possible_strategies:
            if strategy == "quantile":
                # Test all quantile methods for quantile strategy
                for method in quantile_methods:
                    try:
                        # Apply binning
                        discretizer = KBinsDiscretizer(
                            n_bins=n_bins,
                            encode="ordinal",
                            strategy=strategy,
                            subsample=None,  # Use all data
                            random_state=10
                        )

                        binned_X = discretizer.fit_transform(X)

                        # Calculate MI after binning
                        binned_mi_scores = mutual_info_classif(X=binned_X, y=y)

                        # Store results
                        results.append({
                            'column': column_idx,
                            'n_bins': n_bins,
                            'strategy': strategy,
                            'method': method,
                            'initial_mi': initial_mi,
                            'mi_score': binned_mi_scores[0],
                            'mi_improvement': binned_mi_scores[0] - initial_mi,
                            'mi_ratio': binned_mi_scores[0] / initial_mi if initial_mi > 0 else np.nan
                        })

                    except Exception as e:
                        # Handle cases where binning fails
                        results.append({
                            'column': column_idx,
                            'n_bins': n_bins,
                            'strategy': strategy,
                            'method': method,
                            'initial_mi': initial_mi,
                            'mi_score': np.nan,
                            'mi_improvement': np.nan,
                            'mi_ratio': np.nan
                        })
                        print(
                            f"Error with column {column_idx}, bins={n_bins}, strategy={strategy}, method={method}: {e}")

            else:
                # For uniform and kmeans strategies (no quantile method needed)
                try:
                    discretizer = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy=strategy,
                        subsample=None,
                        random_state=10
                    )

                    binned_X = discretizer.fit_transform(X)
                    binned_mi_scores = mutual_info_classif(X=binned_X, y=y)

                    results.append({
                        'column': column_idx,
                        'n_bins': n_bins,
                        'strategy': strategy,
                        'method': 'N/A',
                        'initial_mi': initial_mi,
                        'mi_score': binned_mi_scores[0],
                        'mi_improvement': binned_mi_scores[0] - initial_mi,
                        'mi_ratio': binned_mi_scores[0] / initial_mi if initial_mi > 0 else np.nan
                    })

                except Exception as e:
                    results.append({
                        'column': column_idx,
                        'n_bins': n_bins,
                        'strategy': strategy,
                        'method': 'N/A',
                        'initial_mi': initial_mi,
                        'mi_score': np.nan,
                        'mi_improvement': np.nan,
                        'mi_ratio': np.nan
                    })
                    print(f"Error with column {column_idx}, bins={n_bins}, strategy={strategy}: {e}")

    # Create DataFrame for current column
    column_results_df = pd.DataFrame(results)

    # Display basic statistics for current column
    successful_results = column_results_df['mi_score'].notna().sum()
    failed_results = column_results_df['mi_score'].isna().sum()
    print(f"Column {column_idx} - Successful: {successful_results}, Failed: {failed_results}")

    # Save detailed results for current column
    detailed_filename = f"{results_dir}/column_{column_idx}_detailed_results.csv"
    column_results_df.to_csv(detailed_filename, index=False)

    # Create and save pivot table for current column
    pivot_table = column_results_df.pivot_table(
        values='mi_score',
        index=['n_bins'],
        columns=['strategy', 'method'],
        aggfunc='first'
    )
    pivot_filename = f"{results_dir}/column_{column_idx}_pivot_mi_scores.csv"
    pivot_table.to_csv(pivot_filename)

    # Create and save improvement pivot table
    improvement_pivot = column_results_df.pivot_table(
        values='mi_improvement',
        index=['n_bins'],
        columns=['strategy', 'method'],
        aggfunc='first'
    )
    improvement_filename = f"{results_dir}/column_{column_idx}_pivot_improvements.csv"
    improvement_pivot.to_csv(improvement_filename)

    # Find and save best results for current column
    best_results = column_results_df.dropna().nlargest(10, 'mi_score')
    best_filename = f"{results_dir}/column_{column_idx}_best_results.csv"
    best_results.to_csv(best_filename, index=False)

    # Print best result for current column
    if not best_results.empty:
        best_row = best_results.iloc[0]
        print(f"Column {column_idx} - Best result: MI={best_row['mi_score']:.6f}, "
              f"Bins={best_row['n_bins']}, Strategy={best_row['strategy']}, "
              f"Method={best_row['method']}, Improvement={best_row['mi_improvement']:.6f}")

# Create summary across all columns
print(f"\n{'=' * 80}")
print("CREATING OVERALL SUMMARY...")
print(f"{'=' * 80}")

# Combine all results into one summary file
all_results = []
summary_by_column = []

for column_idx in range(2, 32):
    try:
        detailed_filename = f"{results_dir}/column_{column_idx}_detailed_results.csv"
        if os.path.exists(detailed_filename):
            column_df = pd.read_csv(detailed_filename)
            all_results.append(column_df)

            # Create summary for this column
            column_summary = {
                'column': column_idx,
                'initial_mi': column_df['initial_mi'].iloc[0] if not column_df.empty else np.nan,
                'best_mi': column_df['mi_score'].max() if not column_df.empty else np.nan,
                'best_improvement': column_df['mi_improvement'].max() if not column_df.empty else np.nan,
                'successful_combinations': column_df['mi_score'].notna().sum(),
                'total_combinations': len(column_df)
            }

            # Get best combination details
            if not column_df.empty and column_df['mi_score'].notna().any():
                best_idx = column_df['mi_score'].idxmax()
                column_summary.update({
                    'best_n_bins': column_df.loc[best_idx, 'n_bins'],
                    'best_strategy': column_df.loc[best_idx, 'strategy'],
                    'best_method': column_df.loc[best_idx, 'method']
                })

            summary_by_column.append(column_summary)
    except Exception as e:
        print(f"Error processing summary for column {column_idx}: {e}")

# Save combined results
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f"{results_dir}/all_columns_combined_results.csv", index=False)

# Save summary by column
if summary_by_column:
    summary_df = pd.DataFrame(summary_by_column)
    summary_df.to_csv(f"{results_dir}/summary_by_column.csv", index=False)

    print("\nSUMMARY BY COLUMN:")
    print(summary_df.round(6))

print(f"\n{'=' * 80}")
print("PROCESSING COMPLETE!")
print(f"{'=' * 80}")
print(f"Results saved in '{results_dir}' directory:")
print("- Individual column detailed results: column_X_detailed_results.csv")
print("- Individual column pivot tables: column_X_pivot_mi_scores.csv")
print("- Individual column improvements: column_X_pivot_improvements.csv")
print("- Individual column best results: column_X_best_results.csv")
print("- Combined results: all_columns_combined_results.csv")
print("- Summary by column: summary_by_column.csv")
print(f"\nTotal files created: {len(os.listdir(results_dir))} files")