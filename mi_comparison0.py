import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

file_path = "wdbc.data"
df = pd.read_csv(file_path, delimiter=",")
class MutualInfoAnalyzer:
    def __init__(self, data_path=None, df=None):
        """
        Initialize the analyzer with data.

        Parameters:
        data_path: str, path to CSV file
        df: pandas DataFrame, alternatively provide DataFrame directly
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

        # Encode target variable (M/B to 1/0)
        self.le = LabelEncoder()
        self.df.iloc[:, 1] = self.le.fit_transform(self.df.iloc[:, 1])

        # Store target and feature columns
        self.target = self.df.iloc[:, 1]
        self.features = self.df.iloc[:, 2:]  # All columns after column 1
        self.feature_names = self.features.columns

        print(f"Dataset loaded: {self.df.shape[0]} instances, {self.df.shape[1]} columns")
        print(f"Target distribution: {dict(Counter(self.target))}")
        print(f"Feature columns: {len(self.feature_names)}")

    def calculate_mutual_info_basic(self, X, y, bins=10):
        """
        Calculate mutual information using basic formula with binning.

        MI(X,Y) = H(Y) - H(Y|X)
        where H(Y) is entropy of Y and H(Y|X) is conditional entropy
        """
        # Discretize continuous variable X
        X_binned = pd.cut(X, bins=bins, duplicates='drop')

        # Calculate H(Y)
        y_counts = Counter(y)
        total = len(y)
        h_y = -sum((count / total) * np.log2(count / total) for count in y_counts.values())

        # Calculate H(Y|X)
        h_y_given_x = 0
        for bin_val in X_binned.cat.categories:
            mask = X_binned == bin_val
            if mask.sum() == 0:
                continue

            y_subset = y[mask]
            if len(y_subset) == 0:
                continue

            subset_counts = Counter(y_subset)
            subset_total = len(y_subset)

            # Calculate conditional entropy for this bin
            bin_entropy = -sum((count / subset_total) * np.log2(count / subset_total)
                               for count in subset_counts.values() if count > 0)

            # Weight by probability of this bin
            h_y_given_x += (subset_total / total) * bin_entropy

        # MI = H(Y) - H(Y|X)
        mi = h_y - h_y_given_x
        return max(0, mi)  # Ensure non-negative

    def analyze_all_features(self, bin_sizes=[5, 10, 15, 20, 25, 30]):
        """
        Analyze all features with different binning strategies.
        """
        results = {}

        # Calculate MI using sklearn's mutual_info_regression
        print("Calculating mutual information using mutual_info_regression...")
        mi_regression_scores = mutual_info_regression(self.features, self.target, random_state=42)

        print("Calculating mutual information with different bin sizes...")

        for i, feature_name in enumerate(self.feature_names):
            print(f"Processing feature {i + 1}/{len(self.feature_names)}: {feature_name}")

            feature_data = self.features.iloc[:, i]
            mi_regression_score = mi_regression_scores[i]

            # Calculate MI for different bin sizes
            mi_basic_scores = []
            for bin_size in bin_sizes:
                try:
                    mi_score = self.calculate_mutual_info_basic(feature_data, self.target, bins=bin_size)
                    mi_basic_scores.append(mi_score)
                except Exception as e:
                    print(f"Error with {feature_name}, bin_size {bin_size}: {e}")
                    mi_basic_scores.append(0)

            results[feature_name] = {
                'mi_regression': mi_regression_score,
                'mi_basic': mi_basic_scores,
                'bin_sizes': bin_sizes
            }

        self.results = results
        return results

    def plot_comparison(self, feature_name, figsize=(10, 6)):
        """
        Plot comparison for a single feature.
        """
        if not hasattr(self, 'results'):
            raise ValueError("Run analyze_all_features() first")

        if feature_name not in self.results:
            raise ValueError(f"Feature {feature_name} not found in results")

        data = self.results[feature_name]

        plt.figure(figsize=figsize)

        # Plot MI with different bin sizes
        plt.plot(data['bin_sizes'], data['mi_basic'], 'bo-',
                 label='Basic MI (with binning)', linewidth=2, markersize=6)

        # Plot MI regression as horizontal line
        plt.axhline(y=data['mi_regression'], color='red', linestyle='--',
                    linewidth=2, label=f'MI Regression: {data["mi_regression"]:.4f}')

        plt.xlabel('Bin Size')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information Comparison: {feature_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_all_comparisons(self, cols=3, figsize=(15, 4)):
        """
        Plot comparisons for all features in a grid layout.
        """
        if not hasattr(self, 'results'):
            raise ValueError("Run analyze_all_features() first")

        n_features = len(self.results)
        rows = (n_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i, (feature_name, data) in enumerate(self.results.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col]

            # Plot MI with different bin sizes
            ax.plot(data['bin_sizes'], data['mi_basic'], 'bo-',
                    label='Basic MI', linewidth=1.5, markersize=4)

            # Plot MI regression as horizontal line
            ax.axhline(y=data['mi_regression'], color='red', linestyle='--',
                       linewidth=1.5, label=f'MI Reg: {data["mi_regression"]:.3f}')

            ax.set_xlabel('Bin Size', fontsize=8)
            ax.set_ylabel('MI', fontsize=8)
            ax.set_title(f'{feature_name}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        # Hide empty subplots
        for i in range(n_features, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_summary_statistics(self):
        """
        Get summary statistics of the analysis.
        """
        if not hasattr(self, 'results'):
            raise ValueError("Run analyze_all_features() first")

        summary = []
        for feature_name, data in self.results.items():
            mi_reg = data['mi_regression']
            mi_basic = data['mi_basic']

            summary.append({
                'Feature': feature_name,
                'MI_Regression': mi_reg,
                'MI_Basic_Mean': np.mean(mi_basic),
                'MI_Basic_Std': np.std(mi_basic),
                'MI_Basic_Min': np.min(mi_basic),
                'MI_Basic_Max': np.max(mi_basic),
                'Best_Bin_Size': data['bin_sizes'][np.argmax(mi_basic)],
                'Best_MI_Basic': np.max(mi_basic)
            })

        return pd.DataFrame(summary)

    def plot_summary_heatmap(self, figsize=(12, 8)):
        """
        Create a heatmap showing MI values for all features and bin sizes.
        """
        if not hasattr(self, 'results'):
            raise ValueError("Run analyze_all_features() first")

        # Prepare data for heatmap
        feature_names = list(self.results.keys())
        bin_sizes = self.results[feature_names[0]]['bin_sizes']

        # Create matrix: features x bin_sizes
        mi_matrix = np.zeros((len(feature_names), len(bin_sizes)))

        for i, feature_name in enumerate(feature_names):
            mi_matrix[i, :] = self.results[feature_name]['mi_basic']

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(mi_matrix,
                    xticklabels=bin_sizes,
                    yticklabels=feature_names,
                    annot=True,
                    fmt='.3f',
                    cmap='viridis',
                    cbar_kws={'label': 'Mutual Information'})

        plt.title('Mutual Information Heatmap: Features vs Bin Sizes')
        plt.xlabel('Bin Size')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()


# Example usage:
"""
# Load your dataset
analyzer = MutualInfoAnalyzer(data_path='your_dataset.csv')

# Or if you already have a DataFrame:
# analyzer = MutualInfoAnalyzer(df=your_dataframe)

# Analyze all features
results = analyzer.analyze_all_features(bin_sizes=[5, 10, 15, 20, 25, 30])

# Plot comparison for a specific feature
analyzer.plot_comparison('feature_name')

# Plot all comparisons
analyzer.plot_all_comparisons()

# Get summary statistics
summary = analyzer.get_summary_statistics()
print(summary)

# Create heatmap
analyzer.plot_summary_heatmap()
"""


# If you want to test with sample data:
def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 400
    n_features = 30

    # Create sample data
    data = np.random.randn(n_samples, n_features + 2)

    # First column: ID
    data[:, 0] = range(1, n_samples + 1)

    # Second column: Target (M/B)
    target_prob = 1 / (1 + np.exp(-data[:, 2]))  # Sigmoid based on first feature
    data[:, 1] = np.where(np.random.rand(n_samples) < target_prob, 'M', 'B')

    # Create DataFrame
    columns = ['ID', 'Target'] + [f'Feature_{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(data, columns=columns)

    return df

# Uncomment to test with sample data:
# sample_df = create_sample_data()
analyzer = MutualInfoAnalyzer(df=df)
results = analyzer.analyze_all_features()
analyzer.plot_all_comparisons()