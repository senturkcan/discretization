import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import math

# Load the dataset (has headers)
df = pd.read_csv('parkinsons_plus.data')

# Calculate number of bins using Sturges' rule
n_samples = len(df)
n_bins = int(1 + math.log2(n_samples))

# Get continuous columns (all except column 0 (ID) and column 17 (target))
continuous_cols = [col for col in df.columns if col != df.columns[0] and col != df.columns[17]]

# Apply binning to continuous columns
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

# Create binned dataset
df_binned = df.copy()

# Get bin assignments for continuous columns
bin_assignments = discretizer.fit_transform(df[continuous_cols])

# Replace bin numbers with average values of each bin
for col_idx, col in enumerate(continuous_cols):
    original_values = df[col]
    bin_labels = bin_assignments[:, col_idx]

    # Calculate average for each bin and replace bin numbers with averages
    binned_values = np.zeros_like(bin_labels, dtype=float)
    for bin_num in range(n_bins):
        mask = bin_labels == bin_num
        if np.any(mask):
            bin_average = round(original_values[mask].mean(), 6)
            binned_values[mask] = bin_average

    df_binned[col] = binned_values

# Save the binned dataset
df_binned.to_csv('parkinsons_plus_binned.csv', index=False)

print("Done! Binned dataset saved as 'parkinsons_plus_binned.csv'")
print("Each bin value is now the average of the original values in that bin")