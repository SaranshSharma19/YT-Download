import yt_dlp
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_url(url: str) -> bool:
    """Validate if the URL is a valid YouTube URL."""
    return "youtube.com" in url or "youtu.be" in url

def download_youtube_video(video_url: str, save_path: str) -> str:
    """
    Download YouTube video with enhanced error handling.
    
    Args:
        video_url (str): YouTube video URL
        save_path (str): Directory to save the downloaded video
        
    Returns:
        str: Status message indicating the result
    """
    try:
        if not validate_url(video_url):
            return "Invalid YouTube URL. Please check the URL and try again."
        
        # Define output filename explicitly
        output_filename = Path(save_path) / "downloaded_video.mp4"

        # yt-dlp options
        ydl_opts = {
            'format': 'best',  # Simplified format setting for highest quality
            'outtmpl': str(output_filename),  # Force specific output filename
            'retries': 3,
            'ffmpeg_location': None,
            'nocheckcertificate': True,
            'verbose': True  # Enable detailed yt-dlp output
        }
        
        # Log existing files in save_path before download
        logger.info(f"Contents of save path ({save_path}) before download:")
        logger.info(list(Path(save_path).glob("*")))
        
        # Start download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Starting download...")
            ydl.download([video_url])

        # Verify file exists and has size > 0
        if output_filename.exists() and output_filename.stat().st_size > 0:
            logger.info(f"Download successful! File located at: {output_filename}")
            return f"Video downloaded successfully: {output_filename}"
        else:
            logger.error("Download completed but file verification failed.")
            return "Download completed but file verification failed."

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Download error: {e}")
        return f"Download failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

# Test function
if __name__ == "__main__":
    url = input("Enter YouTube URL to download: ")
    save_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(save_dir, exist_ok=True)
    status = download_youtube_video(url, save_dir)
    print(status)
