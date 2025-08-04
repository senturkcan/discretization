
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import math

# Load the dataset
df = pd.read_csv('beans.csv')

# Calculate number of bins using Sturges' rule
n_samples = len(df)
n_bins = int(1 + math.log2(n_samples))

# Get continuous columns (columns 0-6, excluding the target column 7)
continuous_cols = list(range(16))  # columns 0 to 6

# Apply binning to continuous columns
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

# Create binned dataset
df_binned = df.copy()

# Get bin assignments for continuous columns
bin_assignments = discretizer.fit_transform(df.iloc[:, continuous_cols])

# Replace bin numbers with average values of each bin
for col_idx in range(len(continuous_cols)):
    original_values = df.iloc[:, col_idx]
    bin_labels = bin_assignments[:, col_idx]

    # Calculate average for each bin and replace bin numbers with averages
    binned_values = np.zeros_like(bin_labels, dtype=float)
    for bin_num in range(n_bins):
        mask = bin_labels == bin_num
        if np.any(mask):
            bin_average = round(original_values[mask].mean(), 6)
            binned_values[mask] = bin_average

    df_binned.iloc[:, col_idx] = binned_values

# Convert target variable from text to numeric
df_binned.iloc[:, 16] = df_binned.iloc[:, 16].map({'SEKER': 0, 'BARBUNYA': 1, "BOMBAY":2, "CALI":3,"DERMOSAN":4,"HOROZ":5,"SIRA":6})

# Save the binned dataset
df_binned.to_csv('beans_binned.csv', index=False)

print("Done! Binned dataset saved as '.csv'")
print("Each bin value is now the average of the original values in that bin")