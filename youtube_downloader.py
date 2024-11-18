from pathlib import Path
import streamlit as st
import yt_dlp
import os
import imageio_ffmpeg
import logging
import tempfile
from datetime import datetime

# Set up detailed logging
log_file = os.path.join(tempfile.gettempdir(), f'youtube_downloader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_ffmpeg():
    """Verify FFmpeg installation and return its path."""
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
        return ffmpeg_path
    except Exception as e:
        logger.error(f"FFmpeg verification failed: {e}")
        return None

def check_disk_space(path):
    """Check if there's enough disk space (at least 2GB free)."""
    try:
        total, used, free = os.statvfs(path)[0:6:2]
        free_space = free * total
        logger.info(f"Free disk space: {free_space / (1024**3):.2f} GB")
        return free_space > (2 * 1024**3)  # 2GB minimum
    except Exception as e:
        logger.error(f"Disk space check failed: {e}")
        return False

def download_youtube_video(video_url: str, save_path: str):
    """
    Download YouTube video with enhanced error handling and logging.
    """
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # System checks
        if not verify_ffmpeg():
            return None, "FFmpeg not found or not working properly."
        
        if not check_disk_space(save_path):
            return None, "Insufficient disk space (minimum 2GB required)."

        progress_bar = st.progress(0)
        status_text = st.empty()
        download_info = {"started": False, "complete": False, "progress": 0}

        def progress_hook(d):
            if d['status'] == 'downloading':
                download_info["started"] = True
                try:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        progress = downloaded / total
                        download_info["progress"] = progress
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {progress:.1%} ({downloaded}/{total} bytes)")
                        logger.debug(f"Download progress: {progress:.1%}")
                except Exception as e:
                    logger.warning(f"Progress calculation error: {e}")
            elif d['status'] == 'finished':
                download_info["complete"] = True
                logger.info("Download finished, starting merge process")
                status_text.text("Processing video...")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'verbose': True,
            'progress_hooks': [progress_hook],
            'retries': 10,
            'fragment_retries': 10,
            'file_access_retries': 10,
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-movflags', '+faststart'
            ],
            'ffmpeg_location': verify_ffmpeg()
        }

        logger.info("Starting download with options: %s", str(ydl_opts))
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            status_text.text("Extracting video information...")
            logger.info("Extracting video information from URL: %s", video_url)
            
            info = ydl.extract_info(video_url, download=False)
            safe_title = "".join(x for x in info['title'] if x.isalnum() or x in (' ', '-', '_')).rstrip()
            temp_file_path = os.path.join(temp_dir, f"{safe_title}.mp4")
            final_file_path = os.path.join(save_path, f"{safe_title}.mp4")
            
            logger.info(f"Temporary file path: {temp_file_path}")
            logger.info(f"Final file path: {final_file_path}")
            
            # Download video
            logger.info("Starting video download")
            ydl.download([video_url])
            
            if not download_info["started"]:
                logger.error("Download never started")
                return None, "Download failed to start"
            
            if not download_info["complete"]:
                logger.error("Download never completed")
                return None, "Download was interrupted"

            # Verify temporary file
            if not os.path.exists(temp_file_path):
                logger.error(f"Temporary file not found: {temp_file_path}")
                return None, "Download failed - temporary file not found"
            
            temp_size = os.path.getsize(temp_file_path)
            logger.info(f"Temporary file size: {temp_size} bytes")
            
            if temp_size == 0:
                logger.error("Downloaded file is empty")
                return None, "Download failed - file is empty"

            # Move file to final location
            try:
                os.replace(temp_file_path, final_file_path)
                logger.info(f"Successfully moved file to: {final_file_path}")
                
                final_size = os.path.getsize(final_file_path)
                logger.info(f"Final file size: {final_size} bytes")
                
                if final_size > 0:
                    return final_file_path, "Video downloaded successfully!"
                else:
                    logger.error("Final file is empty")
                    return None, "Download failed - final file is empty"
                    
            except Exception as e:
                logger.error(f"Error moving file: {e}")
                return None, f"Error saving file: {str(e)}"

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube-DL download error: {e}")
        return None, f"Download failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None, f"An unexpected error occurred: {str(e)}"
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file}: {e}")
                os.rmdir(temp_dir)
                logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")
        
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def main():
    st.set_page_config(page_title="YouTube Video Downloader", page_icon="▶️")
    
    st.title("▶️ YouTube Video Downloader")
    st.markdown("""
    Download YouTube videos in highest quality MP4 format.
    Please ensure you have the right to download the video content.
    """)

    # Display log file location
    st.sidebar.text("Debug Information")
    st.sidebar.text(f"Log file location:\n{log_file}")

    video_url = st.text_input("Enter YouTube video URL:", placeholder="https://youtube.com/watch?v=...")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        download_button = st.button("Download Video", type="primary")
    
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
                        key="video_download"
                    )
                st.info(f"Video saved to: {file_path}")
            else:
                st.error(status)
                st.error("Check the log file for detailed error information.")
        else:
            st.error("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()
