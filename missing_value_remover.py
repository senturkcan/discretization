import pandas as pd

# Read the dataset with headers
df = pd.read_csv('chronic_kidney_disease.csv', header=0)

# Display basic info about the original dataset
print("Original dataset shape:", df.shape)
print("Original columns:", list(df.columns))
print("\nFirst few rows:")
print(df.head())

# Remove columns 5, 16, 17 (using 0-based indexing, so columns 4, 15, 16)
columns_to_remove = [5,16,17]  # 0-based indexing for columns 5, 16, 17
print(f"\nRemoving columns at positions: {[i+1 for i in columns_to_remove]}")
print(f"Removing columns: {[df.columns[i] for i in columns_to_remove]}")

df = df.drop(df.columns[columns_to_remove], axis=1)
print(f"Dataset shape after column removal: {df.shape}")
print("Remaining columns:", list(df.columns))

# Check for question marks in the dataset
print("\nChecking for question marks in each column:")
for col in df.columns:
    q_count = (df[col] == '?').sum()
    if q_count > 0:
        print(f"Column '{col}': {q_count} question marks")

# Remove rows that contain more than 1 question mark
question_mark_counts = df.isin(['?']).sum(axis=1)
print(f"\nRows with question marks distribution:")
print(question_mark_counts.value_counts().sort_index())

# Keep only rows with 1 or fewer question marks
df_clean = df[question_mark_counts <= 0]

# Display info about the cleaned dataset
print(f"\nOriginal dataset: {len(df)} rows")
print(f"Cleaned dataset: {len(df_clean)} rows")
print(f"Removed {len(df) - len(df_clean)} rows with more than 1 question mark")

# Show how many rows have exactly 1 question mark (these are kept)
rows_with_one_q = (question_mark_counts == 1).sum()
print(f"Rows with exactly 1 question mark (kept): {rows_with_one_q}")

# Save the cleaned dataset
df_clean.to_csv('removed_cleaned_chronic_kidney_disease.data', index=False)
print("\nCleaned dataset saved as 'cleaned_chronic_kidney_disease.data'")

# Optional: Display first few rows of cleaned data
print("\nFirst few rows of cleaned dataset:")
print(df_clean.head())