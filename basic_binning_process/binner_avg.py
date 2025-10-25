import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import math

# Load the dataset
df = pd.read_csv('wdbc.data', header=None)

# Calculate number of bins using Sturges' rule
n_samples = len(df)
n_bins = int(1 + math.log2(n_samples))

# Apply binning to columns 2 onwards (continuous features)
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

# Create binned dataset
df_binned = df.copy()

# Get bin assignments for continuous columns
continuous_cols = list(range(2, df.shape[1]))
bin_assignments = discretizer.fit_transform(df.iloc[:, continuous_cols])

# Replace bin numbers with average values of each bin
for col_idx, col in enumerate(continuous_cols):
    original_values = df.iloc[:, col]
    bin_labels = bin_assignments[:, col_idx]

    # Calculate average for each bin and replace bin numbers with averages
    binned_values = np.zeros_like(bin_labels, dtype=float)
    for bin_num in range(n_bins):
        mask = bin_labels == bin_num
        if np.any(mask):
            bin_average = round(original_values[mask].mean(), 6)
            binned_values[mask] = bin_average

    df_binned.iloc[:, col] = binned_values

# Convert target variable from text to numeric
df_binned.iloc[:, 1] = df_binned.iloc[:, 1].map({'B': 0, 'M': 1})

# Save the binned dataset
df_binned.to_csv('wdbc_binned.csv', index=False, header=False)

print("Done! Binned dataset saved as 'wdbc_binned.csv'")
print("Each bin value is now the average of the original values in that bin")