import streamlit as st
import yt_dlp
import os
import tempfile
import shutil
from pathlib import Path
import time
import random

def sanitize_filename(filename):
    """Create a safe filename."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    # Limit length to avoid path too long errors
    return filename[:200]

def get_unique_filename(base_path, title):
    """Generate a unique filename if file already exists."""
    base = sanitize_filename(title)
    path = os.path.join(base_path, f"{base}.mp4")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(base_path, f"{base}_{counter}.mp4")
        counter += 1
    return path

def download_youtube_video(video_url: str, save_path: str):
    """Download YouTube video with enhanced file handling."""
    # Create a temporary directory for downloading
    temp_dir = tempfile.mkdtemp()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        def progress_hook(d):
            if d['status'] == 'downloading':
                try:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        progress = downloaded / total
                        progress_bar.progress(progress)
                        downloaded_mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        status_text.text(f"Downloading: {progress:.1%} ({downloaded_mb:.1f}MB/{total_mb:.1f}MB)")
                except:
                    status_text.text("Downloading...")
            elif d['status'] == 'finished':
                status_text.text("Processing video...")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'progress_hooks': [progress_hook],
            'retries': 5,
            'fragment_retries': 5,
            'ignoreerrors': False,
            'no_warnings': True,
            'quiet': True,
            'noprogress': False,
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental'
            ]
        }

        status_text.text("Extracting video information...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First get video info
            info = ydl.extract_info(video_url, download=False)
            if not info:
                return None, "Could not extract video information"
            
            video_title = info.get('title', 'video')
            temp_path = os.path.join(temp_dir, f"{video_title}.mp4")
            
            # Download the video
            status_text.text("Starting download...")
            ydl.download([video_url])
            
            # Find the downloaded file
            downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp4')]
            if not downloaded_files:
                return None, "No video file found after download"
            
            # Get the actual downloaded file
            temp_file = os.path.join(temp_dir, downloaded_files[0])
            if not os.path.exists(temp_file):
                return None, "Downloaded file not found in temporary directory"
            
            # Verify file size
            if os.path.getsize(temp_file) < 1024:  # Less than 1KB
                return None, "Downloaded file is too small, likely corrupted"
            
            # Create final path
            final_path = get_unique_filename(save_path, video_title)
            
            # Move file to final location
            try:
                shutil.move(temp_file, final_path)
            except Exception as e:
                return None, f"Error moving file: {str(e)}"
            
            # Final verification
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                return final_path, "Video downloaded successfully!"
            else:
                return None, "File verification failed after moving"
            
    except yt_dlp.utils.DownloadError as e:
        return None, f"Download error: {str(e)}"
    except Exception as e:
        return None, f"An error occurred: {str(e)}"
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        progress_bar.empty()
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

    # Input for video URL
    video_url = st.text_input(
        "Enter YouTube video URL:",
        placeholder="https://youtube.com/watch?v=..."
    )
    
    # Download section
    if st.button("Download Video", type="primary"):
        if not video_url:
            st.error("Please enter a YouTube URL.")
            return
            
        if not ("youtube.com" in video_url or "youtu.be" in video_url):
            st.error("Please enter a valid YouTube URL.")
            return
            
        # Create downloads directory if it doesn't exist
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads_path, exist_ok=True)
        
        # Process the download
        with st.spinner("Processing download..."):
            file_path, status = download_youtube_video(video_url, downloads_path)
        
        # Handle the result
        if file_path and os.path.exists(file_path):
            st.success(status)
            
            # Get file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Create download button
            with open(file_path, "rb") as file:
                st.download_button(
                    label=f"⬇️ Download Video ({file_size_mb:.1f}MB)",
                    data=file,
                    file_name=os.path.basename(file_path),
                    mime="video/mp4",
                    key="video_download"
                )
            st.info(f"Video saved to: {file_path}")
        else:
            st.error(status)
            st.info("Please verify the URL and try again.")

if __name__ == "__main__":
    main()
