import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd

class DownloadManager:
    """
    Manages downloading of icons and videos/images from JSON data files.
    Downloads files only if they contain the target brandName_logoId.
    """

    def __init__(self):
        """
        Initializes DownloadManager with directory paths, locks, and dictionaries.
        """
        self.json_folder = 'Scripts/Json_data/json_exp'  # Define the folder name here
        self.icon_directory = 'Scripts/Json_data/json_exp/icons'
        self.video_directory = 'Scripts/Json_data/json_exp/video'
        self.image_directory = 'Scripts/Json_data/json_exp/image'
        self.icons = set()
        self.media = set()
        self.url_to_brands = {}  # Dictionary to map downloadUrl to brandName_logoId
        self.icons_lock = threading.Lock()  # Lock for icons set
        self.media_lock = threading.Lock()  # Lock for videos set
        os.makedirs(self.icon_directory, exist_ok=True)
        os.makedirs(self.video_directory, exist_ok=True)
        os.makedirs(self.image_directory, exist_ok=True)

        # Lists for prioritization
        self.high_priority = []
        self.medium_priority = []
        self.low_priority = []

    def download_and_save_file(self, url, save_path):
        """
        Downloads file from the given URL and saves it to the specified path.

        Args:
            url (str): The URL of the file to download.
            save_path (str): The path to save the downloaded file.
        """
        print(f"Downloading {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)

    def process_json_file(self, json_file):
        """
        Processes each JSON file, extracts URLs, and maps them to brandName_logoId.

        Args:
            json_file (str): The name of the JSON file to process.
        """
        with open(os.path.join(self.json_folder, json_file), 'r', encoding='utf-8') as file:
            data = json.load(file)

        for idx, item in enumerate(data):
            download_url = item.get("downloadUrl")
            brand_name = item.get("brandName")
            logo_id = item.get("logoId")  # Extract the logoId

            if download_url and brand_name and logo_id:
                # Create a combined key using brandName_logoId
                brand_logo_key = f"{brand_name}_{logo_id}"

                # Map the download URL to the brandName_logoId
                if download_url not in self.url_to_brands:
                    self.url_to_brands[download_url] = []
                self.url_to_brands[download_url].append(brand_logo_key)

        print(self.url_to_brands)

    def prioritize_downloads(self, target_brand_logo):
        """
        Categorizes URLs into high, medium, and low priority based on target_brand_logo.

        Args:
            target_brand_logo (list): List of brandName_logoId to filter the downloads.
        """
        for url, brand_logos in self.url_to_brands.items():
            matching_count = sum(1 for logo in target_brand_logo if logo in brand_logos)
            print(matching_count)
            # High priority: all logos match
            if matching_count == len(target_brand_logo):
                self.high_priority.append(url)

            # Medium priority: more than 1 match but not all
            elif matching_count > 1:
                self.medium_priority.append(url)

            # Low priority: exactly 1 match
            elif matching_count == 1:
                self.low_priority.append(url)

    def download_priority_list(self, priority_list, priority_name):
        """
        Downloads files from the given priority list.

        Args:
            priority_list (list): List of URLs to download.
            priority_name (str): The priority level (e.g., High, Medium, Low).
        """
        for url in priority_list:
            # Determine file type (video or image) based on URL extension
            if url.endswith('.mp4'):
                media_filename = os.path.join(self.video_directory, os.path.basename(url))
            else:
                media_filename = os.path.join(self.image_directory, os.path.basename(url))

            # Download the file if it doesn't already exist
            if not os.path.isfile(media_filename):
                self.download_and_save_file(url, media_filename)
    '''

    def export_to_excel(self, save_path):
        """
        Exports the priority lists to an Excel file.

        Args:
            save_path (str): The path to save the Excel file.
        """
        all_data = []

        # Add entries from each priority list
        for url in self.high_priority:
            all_data.append({'Priority': 'High', 'download_url': url})

        for url in self.medium_priority:
            all_data.append({'Priority': 'Medium', 'download_url': url})

        for url in self.low_priority:
            all_data.append({'Priority': 'Low', 'download_url': url})

        # Create a DataFrame and save it to Excel
        df = pd.DataFrame(all_data)
        df.to_excel(save_path, index=False)
        print(f"Priority lists have been exported to {save_path}")

    '''

    def export_to_excel(self, save_path):
        """
        Exports the priority lists to an Excel file with an additional 'Logos' column.
        
        Args:
            save_path (str): The path to save the Excel file.
        """
        all_data = []

        # Add entries from each priority list
        for url in self.high_priority:
            logos = ', '.join(self.url_to_brands[url])  # Join the list of logos as a string
            all_data.append({'Priority': 'High', 'download_url': url, 'Logos': logos})

        for url in self.medium_priority:
            logos = ', '.join(self.url_to_brands[url])
            all_data.append({'Priority': 'Medium', 'download_url': url, 'Logos': logos})

        for url in self.low_priority:
            logos = ', '.join(self.url_to_brands[url])
            all_data.append({'Priority': 'Low', 'download_url': url, 'Logos': logos})

        # Create a DataFrame and save it to Excel
        df = pd.DataFrame(all_data)
        df.to_excel(save_path, index=False)
        print(f"Priority lists have been exported to {save_path}")

    
    def start_download(self, target_brand_logo, excel_save_path):
        """
        Initiates the JSON processing, categorizes downloads by priority, and exports the lists.

        Args:
            target_brand_logo (list): List of brandName_logoId to filter the downloads.
            excel_save_path (str): Path to save the Excel file with priorities.
        """
        json_files = os.listdir(self.json_folder)  # Use the folder variable here
        # First, process the JSON files to build the URL-to-brand_logo mapping
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.process_json_file, json_files)

        # Categorize the download URLs into priorities
        self.prioritize_downloads(target_brand_logo)

        # Download files in order of priority
        print("***************************************** high priority download starts ******************************")
        self.download_priority_list(self.high_priority, 'High')
        print("***************************************** high priority download ends ******************************")

        print("***************************************** Medium priority download starts ******************************")
        self.download_priority_list(self.medium_priority, 'Medium')
        print("***************************************** Medium priority download ends ******************************")

        print("***************************************** Low priority download starts ******************************")
        self.download_priority_list(self.low_priority, 'Low')
        print("***************************************** Low priority download ends ******************************")

        # Export the lists to Excel
        self.export_to_excel(excel_save_path)

        print("Media download based on priority completed and exported to Excel.")


# Start the download process based on target brandName_logoId
if __name__ == "__main__":
    # Example list of brandName_logoId to filter
    target_brand_logo = ['Viessmann_1','Audi_2']  # Example input: 'brandName_logoId'

    # Path to save the Excel file
    excel_save_path = "/home/areebadnan/Areeb_code/work/Atheritia/Scripts/Json_data/json_exp/videos.xlsx"

    # Initialize and run the download manager
    manager = DownloadManager()
    manager.start_download(target_brand_logo, excel_save_path)
