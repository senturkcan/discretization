import pandas as pd


def process_data_file(input_file, output_file, columns_to_process=None, decimal_places=6):
    """
    Process a .data file to adjust columns so the minimum value is 0,
    with results rounded to a specified number of decimal places.

    Args:
        input_file (str): Input file path
        output_file (str): Output file path
        columns_to_process (list): Columns to adjust (None = all numeric columns)
        decimal_places (int): Number of decimal places to round to (default: 6)
    """
    try:
        # Read data (assumes no header row)
        df = pd.read_csv(input_file, header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Process specified columns (or all numeric columns if None)
    if columns_to_process is None:
        columns_to_process = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    for col in columns_to_process:
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

        if df[col].isnull().all():
            print(f"Warning: Column {col} is non-numeric and was skipped.")
            continue

        min_val = df[col].min()
        if min_val < 0:
            # Shift values and round to 6 decimal places
            df[col] = (df[col] + abs(min_val)).round(decimal_places)

    # Save results
    df.to_csv(output_file, header=False, index=False)
    print(f"Processed data saved to {output_file} (rounded to {decimal_places} decimal places)")


# Example usage
if __name__ == "__main__":
    process_data_file(
        input_file="parkinsons.data",  # Replace with your input file
        output_file="parkinsons_plus.data",  # Replace with desired output file
        columns_to_process=[20],  # Specify columns (0-based indices)
        decimal_places=6  # Set precision (default: 6)
    )