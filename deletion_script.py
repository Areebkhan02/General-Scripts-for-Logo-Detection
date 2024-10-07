import pandas as pd
import os
from tqdm import tqdm

def delete_files_from_csv(csv_file_path: str, column_index: int) -> None:
    """
    Delete image files and their corresponding label files based on file paths listed in a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file containing file paths.
        column_index (int): The index of the column containing the file paths.

    Returns:
        None
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Ensure the specified column index is within the bounds
    if column_index >= len(df.columns):
        print(f"Column index '{column_index}' is out of range.")
        return
    
    # Loop through each file path in the specified column
    for file_path in tqdm(df.iloc[:, column_index], desc="Deleting files", unit="file"):
        # Delete the image file
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted image file: {file_path}")
        else:
            print(f"Image file not found: {file_path}")
        
        # Construct the label file path
        label_file_path = file_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # Delete the label file
        if os.path.isfile(label_file_path):
            os.remove(label_file_path)
            print(f"Deleted label file: {label_file_path}")
        else:
            print(f"Label file not found: {label_file_path}")

if __name__ == "__main__":
    # Example usage
    csv_file_path = 'issues1.csv'
    column_index = 0  # Replace with the correct column index
    
    delete_files_from_csv(csv_file_path, column_index)
