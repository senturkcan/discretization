import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import math

# Load the dataset
df = pd.read_csv('removed_cleaned_chronic_kidney_disease.data')

# Calculate number of bins using Sturges' rule
n_samples = len(df)
n_bins = int(1 + math.log2(n_samples))

# Get continuous columns (all except column 0 (ID) and column 17 (target))
# Get column indices instead of column names for iloc
continuous_cols = [i for i in range(len(df.columns)) if i != 3 and i != 4 and i != 5 and i != 6
                   and i != 7 and i != 15 and i != 16 and i != 17 and i != 18 and i != 19 and i != 20 and i != 21]

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
df_binned.to_csv('removed_binned_cleaned_chronic_kidney_disease.csv', index=False)

print("Done! Binned dataset saved as '.csv'")
print(f"Applied {n_bins} bins to each of the {len(continuous_cols)} continuous columns separately")
print("Each bin value represents the bin number (0 to n_bins-1) for that column")