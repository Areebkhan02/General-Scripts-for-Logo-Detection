import os
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd

class DownloadManager:
    """
    Manages downloading of videos/images based on priority from an Excel file.
    """

    def __init__(self):
        """
        Initializes DownloadManager with directory paths and locks.
        """
        self.video_directory = 'Scripts/Json_data/json_exp/video'
        self.image_directory = 'Scripts/Json_data/json_exp/image'
        self.media_lock = threading.Lock()  # Lock for thread-safe downloading
        os.makedirs(self.video_directory, exist_ok=True)
        os.makedirs(self.image_directory, exist_ok=True)

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

    def download_priority_list(self, download_urls):
        """
        Downloads files from the given list of URLs using multithreading.

        Args:
            download_urls (list): List of URLs to download.
        """
        def download_file(url):
            # Determine file type (video or image) based on URL extension
            if url.endswith('.mp4'):
                media_filename = os.path.join(self.video_directory, os.path.basename(url))
            else:
                media_filename = os.path.join(self.image_directory, os.path.basename(url))

            # Download the file if it doesn't already exist
            if not os.path.isfile(media_filename):
                self.download_and_save_file(url, media_filename)

        # Multithreading download
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(download_file, download_urls)

    def start_download(self, excel_file, priority_level):
        """
        Initiates the download process based on priority from the Excel file.

        Args:
            excel_file (str): Path to the Excel file containing URLs and priority.
            priority_level (str): Priority level (High, Medium, Low) to filter the downloads.
        """
        # Read the Excel file
        df = pd.read_excel(excel_file)

        # Filter based on priority level
        priority_df = df[df['Priority'] == priority_level]

        # Get the download URLs
        download_urls = priority_df['download_url'].tolist()

        # Start downloading
        print(f"***************************************** {priority_level} priority download starts ******************************")
        self.download_priority_list(download_urls)
        print(f"***************************************** {priority_level} priority download ends ******************************")


# Start the download process based on user input priority
if __name__ == "__main__":
    # Input Excel file path
    excel_file = 'Scripts/Json_data/json_exp/videos.xlsx'

    # User input for priority (High, Medium, Low)
    priority_level = 'Medium'  # Example: 'High', 'Medium', or 'Low'

    # Initialize and run the download manager
    manager = DownloadManager()
    manager.start_download(excel_file, priority_level)
