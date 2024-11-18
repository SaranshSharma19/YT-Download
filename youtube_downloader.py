from pathlib import Path
import yt_dlp
import os
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_url(url: str) -> bool:
    """Validate if the URL is a valid YouTube URL."""
    return "youtube.com" in url or "youtu.be" in url

def get_safe_filename(title: str) -> str:
    """Convert title to a safe filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    filename = ''.join(char if char not in invalid_chars else '_' for char in title)
    return filename[:200]  # Limit filename length to 200 characters

def download_youtube_video(video_url: str, save_path: str) -> Tuple[Optional[str], str]:
    """
    Download YouTube video with enhanced error handling and progress tracking.
    
    Args:
        video_url (str): YouTube video URL
        save_path (str): Directory to save the downloaded video
        
    Returns:
        Tuple[Optional[str], str]: Tuple of (file_path, status_message)
    """
    try:
        if not validate_url(video_url):
            return None, "Invalid YouTube URL. Please check the URL and try again."

        # Check save path permissions
        if not os.access(save_path, os.W_OK):
            return None, f"Permission error: Cannot write to directory {save_path}"
        
        ydl_opts = {
            'format': 'best',  # Simplified format setting
            'merge_output_format': 'mp4',
            'outtmpl': str(Path(save_path) / '%(title)s.%(ext)s'),
            'retries': 5,
            'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
            'nocheckcertificate': True,
            'verbose': True  # Enable detailed yt-dlp output for debugging
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Clear yt-dlp cache to prevent stale data issues
            ydl.cache.remove()
            
            logger.info("Extracting video information...")
            info = ydl.extract_info(video_url, download=False)
            
            # Create a safe filename for the video
            video_title = get_safe_filename(info['title'])
            file_path = Path(save_path) / f"{video_title}.mp4"
            
            # Check if file already exists
            if file_path.exists():
                return str(file_path), "Video already exists in downloads folder."
            
            # Start the download
            logger.info("Starting download...")
            ydl.download([video_url])
            
            # Verify the file exists after download
            downloaded_files = list(Path(save_path).glob("*.*"))
            logger.info(f"Files in directory after download: {[str(file) for file in downloaded_files]}")
            if file_path.exists() and file_path.stat().st_size > 0:
                logger.info(f"Downloaded file path: {file_path}")
                logger.info(f"File size: {file_path.stat().st_size} bytes")
                return str(file_path), "Video downloaded successfully!"
            else:
                return None, "Download completed, but file verification failed."

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Download error: {e}")
        return None, f"Download failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None, f"An unexpected error occurred: {str(e)}"

# Test download outside Streamlit
if __name__ == "__main__":
    url = input("Enter YouTube URL to download: ")
    save_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(save_dir, exist_ok=True)
    file_path, status = download_youtube_video(url, save_dir)
    print(status)
    if file_path:
        print(f"File downloaded to: {file_path}")
