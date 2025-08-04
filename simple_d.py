#Guideline:
#select dataset
#find mutual information
#apply KBinsDiscretizer from scikitlearn
#find the new mutual information
#make it dynamic so it can be applied for a lot of variations of binning

#improvments:
#birkaç kez initialize edilmeli ki daha doğru sonuçlar alınsın şans azalsın
#make it for all columns
#make it for all possiable bin sizes (it will requaire checking best combinations automatically)
#import
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
#mutual_info_classif for discrete target variable
#sklearn.feature_selection.mutual_info_regression for continuous target variable

# Load dataset + makeM,B 0,1
df = pd.read_csv("wdbc.data", delimiter=",")
y = LabelEncoder().fit_transform(df.iloc[:, 1])

# possiable values
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
    initial_mi = mutual_info_regression(X, y)[0]

    results = []

    # Test all combinations
    for bins in bin_sizes:
        for strategy in strategies:
            if strategy == "quantile":
                # Test all quantile methods
                for method in quantile_methods:
                    try:
                        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal",
                                                       strategy=strategy, random_state=10)
                        binned_X = discretizer.fit_transform(X)
                        new_mi = mutual_info_classif(binned_X, y)[0]

                        results.append({
                            'n_bins': bins,
                            'strategy': strategy,
                            'method': method,
                            'initial_mi': initial_mi,
                            'binned_mi': new_mi,
                            'improvement': new_mi - initial_mi
                        })
                    except:
                        # Skip if binning fails
                        continue
            else:
                # For uniform and kmeans
                try:
                    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal",
                                                   strategy=strategy, random_state=10)
                    binned_X = discretizer.fit_transform(X)
                    new_mi = mutual_info_classif(binned_X, y)[0]

                    results.append({
                        'n_bins': bins,
                        'strategy': strategy,
                        'method': 'N/A',
                        'initial_mi': initial_mi,
                        'binned_mi': new_mi,
                        'improvement': new_mi - initial_mi
                    })
                except:
                    # Skip if binning fails
                    continue

    # Save results for each column
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"column_{col}_results.csv", index=False)

    # Print best result
    if not results_df.empty:
        best = results_df.loc[results_df['binned_mi'].idxmax()]
        print(f"Column {col} - Best: {best['binned_mi']:.4f} "
              f"(bins={best['n_bins']}, strategy={best['strategy']}, method={best['method']})")

print("Done.")