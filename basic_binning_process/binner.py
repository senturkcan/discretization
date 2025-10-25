import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import math

# Load the dataset (no headers)
df = pd.read_csv('wdbc.data', header=None)

# Calculate number of bins using Sturges' rule
n_samples = len(df)
n_bins = int(1 + math.log2(n_samples))

# Get continuous columns (columns 2-31, excluding ID column 0 and target column 1)
continuous_cols = list(range(2, 32))  # columns 2 to 31 (30 continuous columns)

# Create binned dataset
df_binned = df.copy()

# Apply binning to each continuous column separately
for col_idx in continuous_cols:
    # Create discretizer for this column
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')

    # Fit and transform this column only
    column_data = df.iloc[:, col_idx].values.reshape(-1, 1)
    binned_column = discretizer.fit_transform(column_data)

    # Replace the column with binned values
    df_binned.iloc[:, col_idx] = binned_column.flatten()

# Save the binned dataset
df_binned.to_csv('wdbc_binned.csv', index=False, header=False)

print("Done! Binned dataset saved as 'wdbc_binned.csv'")
print(f"Applied {n_bins} bins to each of the {len(continuous_cols)} continuous columns separately")
print("Column 0 (ID) and Column 1 (target) were preserved without binning")
print(f"Total columns: {df.shape[1]}")
print(f"Binned columns: {len(continuous_cols)} (columns 2-31)")
print("Each bin value represents the bin number (0 to n_bins-1) for that column")