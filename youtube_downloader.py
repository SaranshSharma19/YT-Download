from pathlib import Path
import streamlit as st
import yt_dlp
import os
import imageio_ffmpeg
import logging
from datetime import datetime
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_agent():
    """Return a random modern user agent."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    ]
    return random.choice(user_agents)

def download_youtube_video(video_url: str, save_path: str):
    """Download YouTube video with enhanced security handling."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                try:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        progress = downloaded / total
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {progress:.1%}")
                except:
                    pass
            elif d['status'] == 'finished':
                status_text.text("Processing video...")

        # Enhanced options to avoid 403 errors
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'progress_hooks': [progress_hook],
            'retries': 3,
            'fragment_retries': 3,
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': True,
            'nocheckcertificate': True,
            'http_chunk_size': 10485760,  # 10MB chunks
            'user_agent': get_user_agent(),
            'headers': {
                'Referer': 'https://www.youtube.com/',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            },
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental'
            ],
            'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe()
        }

        status_text.text("Preparing download...")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First get video info without downloading
            try:
                info = ydl.extract_info(video_url, download=False)
                if info is None:
                    return None, "Could not retrieve video information. Please check the URL."
                
                video_title = info.get('title', 'video')
                safe_title = "".join(x for x in video_title if x.isalnum() or x in (' ', '-', '_')).rstrip()
                file_path = os.path.join(save_path, f"{safe_title}.mp4")
                
                # Download the video
                status_text.text("Starting download...")
                ydl.download([video_url])
                
                # Verify download
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    progress_bar.progress(1.0)
                    status_text.text("Download complete!")
                    return file_path, "Video downloaded successfully!"
                else:
                    alt_file_path = os.path.join(save_path, f"{safe_title}.mkv")
                    if os.path.exists(alt_file_path) and os.path.getsize(alt_file_path) > 0:
                        return alt_file_path, "Video downloaded successfully!"
                    return None, "Download completed but file not found. Please try again."
                    
            except yt_dlp.utils.DownloadError as e:
                if "HTTP Error 403" in str(e):
                    logger.error("403 error encountered, retrying with alternate options...")
                    # Try alternate download method
                    ydl_opts['format'] = 'best'  # Simplify format selection
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                        ydl2.download([video_url])
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            return file_path, "Video downloaded successfully!"
                return None, "Could not access video. Please verify the URL and try again."
            
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        return None, f"Download error: {str(e)}"
    finally:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def main():
    st.set_page_config(
        page_title="YouTube Video Downloader",
        page_icon="▶️",
        initial_sidebar_state="collapsed"
    )
    
    st.title("▶️ YouTube Video Downloader")
    st.markdown("""
    Download YouTube videos in highest quality MP4 format.
    Please ensure you have the right to download the video content.
    """)

    video_url = st.text_input(
        "Enter YouTube video URL:",
        placeholder="https://youtube.com/watch?v=..."
    )
    
    if st.button("Download Video", type="primary"):
        if video_url:
            if not ("youtube.com" in video_url or "youtu.be" in video_url):
                st.error("Please enter a valid YouTube URL.")
                return
                
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
        else:
            st.error("Please enter a YouTube URL.")

if __name__ == "__main__":
    main()
