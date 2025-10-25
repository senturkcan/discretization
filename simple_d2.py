# Simple binning comparison for columns 2-31 with multiple runs
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

# Multiple random states to eliminate luck
random_states = [10, 42, 123, 456, 789, 999, 1337, 2023, 5555, 9999]
n_runs = len(random_states)

# Load dataset
df = pd.read_csv("wdbc.data", delimiter=",")

# Encode target variable (M/B to 0/1)
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, 1])  # Convert M/B to numeric

# Parameters (cannot be changed)
bin_sizes = [4, 8, 16, 32, 64, 100, 140, 160, 180, 200]
strategies = ["uniform", "quantile", "kmeans"]
quantile_methods = [
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
    "interpolated_inverted_cdf", "hazen", "weibull", "linear",
    "median_unbiased", "normal_unbiased"
]

# Process each column
for col in range(2, 32):
    print(f"Processing column {col}...")

    X = df.iloc[:, [col]]

    # Use mutual_info_classif for both continuous and discrete features with categorical target
    initial_mi = mutual_info_classif(X, y, random_state=42)[0]

    results = []

    # Test all combinations with multiple runs
    for bins in bin_sizes:
        for strategy in strategies:
            if strategy == "quantile":
                # Test all quantile methods
                for method in quantile_methods:
                    mi_scores = []

                    # Run with multiple random states
                    for rs in random_states:
                        try:
                            discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal",
                                                           strategy=strategy, quantile_method=method, random_state=rs)
                            binned_X = discretizer.fit_transform(X)
                            # Use mutual_info_classif for discretized features
                            new_mi = mutual_info_classif(binned_X, y, random_state=rs)[0]
                            mi_scores.append(new_mi)
                        except:
                            continue

                    # Calculate statistics if we have results
                    if mi_scores:
                        results.append({
                            'n_bins': bins,
                            'strategy': strategy,
                            'method': method,
                            'initial_mi': initial_mi,
                            'mean_mi': np.mean(mi_scores),
                            'std_mi': np.std(mi_scores),
                            'min_mi': np.min(mi_scores),
                            'max_mi': np.max(mi_scores),
                            'n_runs': len(mi_scores),
                            'mean_improvement': np.mean(mi_scores) - initial_mi
                        })
            else:
                # For uniform and kmeans
                mi_scores = []

                # Run with multiple random states
                for rs in random_states:
                    try:
                        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal",
                                                       strategy=strategy, random_state=rs)
                        binned_X = discretizer.fit_transform(X)
                        new_mi = mutual_info_classif(binned_X, y, random_state=rs)[0]
                        mi_scores.append(new_mi)
                    except:
                        continue

                # Calculate statistics if we have results
                if mi_scores:
                    results.append({
                        'n_bins': bins,
                        'strategy': strategy,
                        'method': 'N/A',
                        'initial_mi': initial_mi,
                        'mean_mi': np.mean(mi_scores),
                        'std_mi': np.std(mi_scores),
                        'min_mi': np.min(mi_scores),
                        'max_mi': np.max(mi_scores),
                        'n_runs': len(mi_scores),
                        'mean_improvement': np.mean(mi_scores) - initial_mi
                    })

    # Save results for this column
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"column_{col}_results.csv", index=False)

    # Print best result based on mean performance
    if not results_df.empty:
        best = results_df.loc[results_df['mean_mi'].idxmax()]
        print(f"Column {col} - Best mean MI: {best['mean_mi']:.4f} ± {best['std_mi']:.4f} "
              f"(bins={best['n_bins']}, strategy={best['strategy']}, method={best['method']}, "
              f"range: {best['min_mi']:.4f}-{best['max_mi']:.4f})")

print(f"Done! Each combination tested with {n_runs} different random initializations.")