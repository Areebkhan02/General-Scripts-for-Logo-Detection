import os
import random
from cleanvision import Imagelab

def clean_dataset(dataset_path):
    """
    Cleans the dataset by removing exact duplicates, near duplicates, blurry, and low-information images
    based on user input.

    Args:
        dataset_path (str): Path to the input dataset.
    """
    
    # Normalize the dataset path to handle cross-platform file path issues
    dataset_path = os.path.normpath(dataset_path)

    # Initialize imagelab with the dataset
    imagelab = Imagelab(data_path=dataset_path)

    # Find issues in the dataset
    imagelab.find_issues()

    # Display summary of issues to the user
    print("\nIssue Summary:")
    print(imagelab.issue_summary)

    # Prompt user for input on whether to apply remedies (removal of duplicates, blurry, and low-info images)
    user_input = input("\nDo you want to apply the remedy for removal of duplicates, blurry, and low-info images? (yes/no): ").strip().lower()

    if user_input == 'yes':
        print("Applying remedies...\n")

        # Step 1: Identify exact duplicates and save filenames in a list
        exact_duplicates = imagelab.issues[imagelab.issues["is_exact_duplicates_issue"] == True]
        exact_duplicate_files = exact_duplicates.index.tolist()  # List of filenames with exact duplicates
        print(f"Found {len(exact_duplicate_files)} exact duplicates.")

        # Step 2: Identify near duplicates, shuffle them, and drop 10% randomly
        near_duplicates = imagelab.issues[imagelab.issues["is_near_duplicates_issue"] == True]
        near_duplicate_files = near_duplicates.index.tolist()  # List of filenames with near duplicates
        random.shuffle(near_duplicate_files)  # Shuffle the list
        drop_count_near = int(0.10 * len(near_duplicate_files))  # Calculate 10% of the list
        near_duplicate_files = near_duplicate_files[:-drop_count_near]  # Drop 10% of the files randomly
        print(f"After dropping 10%, {len(near_duplicate_files)} near duplicates remaining.")

        # Step 3: Identify blurry and low-information images, and apply the 10% threshold
        blurry_files = imagelab.issues[imagelab.issues["is_blurry_issue"] == True].index.tolist()  # Blurry images
        random.shuffle(blurry_files)  # Shuffle the list
        drop_count_blurry = int(0.10 * len(blurry_files))  # Calculate 10% of blurry images to drop
        blurry_files = blurry_files[:-drop_count_blurry]  # Drop 10% of blurry images
        print(f"After dropping 10%, {len(blurry_files)} blurry images remaining.")

        low_info_files = imagelab.issues[imagelab.issues["is_low_information_issue"] == True].index.tolist()  # Low info images
        random.shuffle(low_info_files)  # Shuffle the list
        drop_count_low_info = int(0.10 * len(low_info_files))  # Calculate 10% of low info images to drop
        low_info_files = low_info_files[:-drop_count_low_info]  # Drop 10% of low info images
        print(f"After dropping 10%, {len(low_info_files)} low-information images remaining.")

        # Step 4: Combine all problematic files into a set to ensure uniqueness
        removal_files = set(exact_duplicate_files + near_duplicate_files + blurry_files + low_info_files)
        print(f"Total files to remove: {len(removal_files)}")

        # Step 5: Delete the problematic files from the dataset directory
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_path = os.path.normpath(file_path)  # Normalize file path for Windows compatibility
                if file_path in removal_files:  # Only delete files in the removal set
                    try:
                        print(f"Deleting {file_path}")
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

        print("Cleaned dataset by removing unwanted files.")

    else:
        print("No remedies applied. Exiting.")


# Main entry point for the script
if __name__ == "__main__":
    # Define your dataset path
    dataset_path = "/home/areebadnan/Areeb_code/work/Visua_Data/output_videos/65605d69cc6286cec0577f75/images"
    
    # Call the cleaning function
    clean_dataset(dataset_path)
