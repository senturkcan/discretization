import numpy as np

from sklearn.datasets import make_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from npeet.entropy_estimators import mi
import matplotlib.pyplot as plt

# Create synthetic data
X, y = make_regression(n_samples=500, n_features=1, noise=0.1, random_state=0)

# Compute MI using Kraskov estimator
mi_kraskov = mi(X[:, 0], y)
print(f"Kraskov MI: {mi_kraskov:.4f}")

# Compare with binned mutual information for various bin sizes
bin_sizes = [2, 4, 6, 8, 10, 15, 20, 30, 50]
mi_binned = []

for n_bins in bin_sizes:
    # Discretize X
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_binned = discretizer.fit_transform(X)
    
    # Compute MI with discretized X
    mi_bin = mutual_info_regression(X_binned, y, discrete_features=True)
    mi_binned.append(mi_bin[0])  # only one feature
    print(f"Bins: {n_bins} → Binned MI: {mi_bin[0]:.4f}")

# Plot the comparison
plt.figure(figsize=(8, 5))
plt.plot(bin_sizes, mi_binned, marker='o', label='Binned MI (scikit-learn)')
plt.axhline(y=mi_kraskov, color='r', linestyle='--', label='Kraskov MI (npeet)')
plt.xlabel("Number of Bins")
plt.ylabel("Mutual Information")
plt.title("Binned MI vs. Kraskov MI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
