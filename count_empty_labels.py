import os

def count_empty_yolo_txt_files(folder_path):
    """
    Count empty YOLO .txt files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing YOLO .txt files.

    Returns:
        Tuple[int, int]: A tuple containing the count of empty files and the total count of files checked.
    """
    empty_files_count = 0
    total_files_count = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a YOLO txt file
        if filename.endswith(".txt"):
            total_files_count += 1
            file_path = os.path.join(folder_path, filename)
            # Check if the file is empty by reading its content
            with open(file_path, 'r') as file:
                content = file.read()
                if not content.strip():
                    empty_files_count += 1
                    # print(f"Empty file: {filename}")

    return empty_files_count, total_files_count

# Example usage
folder_path = '/home/areebadnan/Desktop/data/predict2/labels'
empty_count, total_count = count_empty_yolo_txt_files(folder_path)
print(f"Total YOLO txt files: {total_count}")
print(f"Empty YOLO txt files: {empty_count}")
