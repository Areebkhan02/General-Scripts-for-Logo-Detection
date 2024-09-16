import os
import requests
from googleapiclient.discovery import build
from tqdm import tqdm
import yt_dlp as youtube_dl

# Set your Google Custom Search API Key and Search Engine ID
#'AIzaSyBv1b1Mz6rVpsYHki5ns-qLfT0ZkGuUw5o' API key
# '03a4c9f15b5ad42de' search engine ID 

API_KEY =  'xxxx' # Replace with your Google API key
SEARCH_ENGINE_ID = 'xxx'  # Replace with your search engine ID

def search_text(query, api_key, search_engine_id):
    """Searches Google Images using a provided text query."""
    # Search for the text query using Google Custom Search API
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(
        q=query, cx=search_engine_id, searchType='image', num=10
    ).execute()
    
    return res['items']

def download_image(image_url, save_path):
    """Downloads an image from a URL."""
    img_data = requests.get(image_url).content
    with open(save_path, 'wb') as handler:
        handler.write(img_data)

def search_videos(query, max_results=5):
    """Searches for YouTube videos related to the provided text query."""
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
        'quiet': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)['entries']
    
    # Filter videos to be 2 minutes or less
    short_videos = [video for video in search_results if video['duration'] <= 120]
    
    return short_videos[:4]  # Limit to the first 2 short videos

def download_video(video_info, output_dir):
    """Downloads a video using the provided video information."""
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_dir, f"{video_info['title']}.%(ext)s"),
        'quiet': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_info['webpage_url']])

def main(query, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for the text query on Google Images
    print("Searching for images...")
    search_results = search_text(query, API_KEY, SEARCH_ENGINE_ID)
    
    # Download the first 3 images
    print("Downloading images...")
    for i, item in enumerate(tqdm(search_results[:3], desc="Images"), 1):
        image_save_path = os.path.join(output_dir, f"image_{i}.jpg")
        download_image(item['link'], image_save_path)
    
    # Search for videos related to the query
    print("Searching for videos...")
    video_results = search_videos(query)
    
    # Download the first 2 videos that are 2 minutes or less
    print("Downloading videos...")
    for video in tqdm(video_results, desc="Videos"):
        download_video(video, output_dir)

    print("Download complete! Files saved to:", output_dir)

if __name__ == "__main__":
    # Provide your search query here
    query = 'cocacola logo in a match stadium'  # Replace with your search query
    
    # Specify the output directory where images and videos will be saved
    output_dir = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/Google_Search_Test/output/'  # Replace with your desired output directory
    
    main(query, output_dir)
