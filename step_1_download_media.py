"""
This code defines a DownloadManager class that manages the download
of icons and videos from JSON files. It creates directories to store
the downloaded files, downloads the files concurrently using ThreadPoolExecutor,
and ensures that each file is downloaded only once.
"""

import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

class DownloadManager:
    """
    Manages downloading of icons and videos from JSON data files.
    """

    def __init__(self):
        """
        Initializes DownloadManager with directory paths and locks.
        """
        self.json_folder = 'Scripts/Json_data/json_exp'  # Define the folder name here
        self.icon_directory = 'Scripts/Json_data/json_exp/icons'
        self.video_directory = 'vScripts/Json_data/json_exp/video'
        self.image_directory = 'Scripts/Json_data/json_exp/image'
        self.icons = set()
        self.media = set()
        self.icons_lock = threading.Lock()  # Lock for icons set
        self.media_lock = threading.Lock()  # Lock for videos set
        os.makedirs(self.icon_directory, exist_ok=True)
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

    def process_json_file(self, json_file):
        """
        Processes each JSON file, extracts icon and video URLs, and downloads them.

        Args:
            json_file (str): The name of the JSON file to process.
        """
        with open(os.path.join(self.json_folder, json_file), 'r', encoding='utf-8') as file:
            data = json.load(file)

        print(json_file, len(data))
        for idx, item in enumerate(data):
            icon_url = item.get("iconUrl")
            download_url = item.get("downloadUrl")
            medium = item.get('medium')
            logo = item.get('brandName')
            flag = False
            print(idx, medium, logo)

            if download_url.endswith('.mp4'):
                flag = True

            # Download and save icons
            if icon_url and logo not in self.icons:
                with self.icons_lock:
                    self.icons.add(logo)
                icon_filename = os.path.join(self.icon_directory, f"{logo}.png")
                if not os.path.isfile(icon_filename):
                    self.download_and_save_file(icon_url, icon_filename)

            # Download and save videos
            if download_url and medium not in self.media:
                with self.media_lock:
                    self.media.add(medium)
                if flag:
                    media_filename = os.path.join(self.video_directory, f"{medium}.mp4")
                else:
                    media_filename = os.path.join(self.image_directory, f"{medium}.jpg")
                if not os.path.isfile(media_filename):
                    self.download_and_save_file(download_url, media_filename)

        print(f"{json_file} Icons and videos downloaded and saved.")

    def start_download(self):
        """
        Initiates the download process for JSON files in parallel.
        """
        json_files = os.listdir(self.json_folder)  # Use the folder variable here
        with ThreadPoolExecutor(max_workers=15) as executor:
            executor.map(self.process_json_file, json_files)

        print("All icons and videos downloaded and saved.")

# Start downloading process
if __name__ == "__main__":
    manager = DownloadManager()
    manager.start_download()
