import pandas as pd

def load_data(file_path):
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Loaded data as a Pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None

def clean_column(df, column_name):
    """
    Fill NaN values in a specific column with an empty string.

    Args:
    - df (DataFrame): The DataFrame to process.
    - column_name (str): Column to clean.

    Returns:
    - DataFrame: Updated DataFrame with cleaned column.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].fillna('')
    return df
