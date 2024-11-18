from pathlib import Path
import streamlit as st
import yt_dlp
import os
import imageio_ffmpeg
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

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_hook(d):
            """Update progress bar and status text based on download progress."""
            if d['status'] == 'downloading':
                try:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    if total:
                        progress = d['downloaded_bytes'] / total
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {progress:.1%}")
                except Exception as e:
                    logger.warning(f"Progress calculation error: {e}")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4 format
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'quiet': True,
            'progress_hooks': [progress_hook],
            'retries': 5,
            'fragment_retries': 5,
            'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe()
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            status_text.text("Extracting video information...")
            info = ydl.extract_info(video_url, download=False)
            
            # Create a safe filename for the video
            video_title = get_safe_filename(info['title'])
            file_path = os.path.join(save_path, f"{video_title}.mp4")
            
            # Check if the file already exists
            if os.path.exists(file_path):
                return file_path, "Video already exists in the downloads folder."
            
            # Start the download
            status_text.text("Starting download...")
            ydl.download([video_url])
            
            # Verify that the file exists and has a size greater than 0
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                progress_bar.progress(1.0)
                status_text.text("Download complete!")
                return file_path, "Video downloaded successfully!"
            else:
                return None, "Download completed but file verification failed."

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Download error: {e}")
        return None, f"Download failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None, f"An unexpected error occurred: {str(e)}"
    finally:
        progress_bar.empty()
        status_text.empty()

def main():
    st.set_page_config(page_title="YouTube Video Downloader", page_icon="▶️")
    
    st.title("▶️ YouTube Video Downloader")
    st.markdown("""
    Download YouTube videos in highest quality MP4 format.
    Please ensure you have the right to download the video content.
    """)

    video_url = st.text_input("Enter YouTube video URL:", placeholder="https://youtube.com/watch?v=...")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        download_button = st.button("Download Video")

    if download_button:
        if video_url:
            save_path = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(save_path, exist_ok=True)
            
            with st.spinner("Processing download..."):
                file_path, status = download_youtube_video(video_url, save_path)
            
            if file_path and os.path.exists(file_path):
                st.success(status)
                with open(file_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=file,
                        file_name=os.path.basename(file_path),
                        mime="video/mp4",
                    )
                st.info(f"Video also saved to: {file_path}")
            else:
                st.error(status)
        else:
            st.error("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()
