import streamlit as st
import yt_dlp
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import platform

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        # Try to run ffmpeg
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        return False

def get_installation_instructions():
    """Get FFmpeg installation instructions based on OS."""
    system = platform.system().lower()
    
    if system == 'windows':
        return """
        To install FFmpeg on Windows:
        1. Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/
        2. Extract the zip file
        3. Add the bin folder to your System PATH
        
        Alternatively, use the command:
        ```
        winget install FFmpeg
        ```
        """
    elif system == 'darwin':  # macOS
        return """
        To install FFmpeg on macOS:
        1. Install Homebrew if not installed:
           ```
           /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
           ```
        2. Install FFmpeg:
           ```
           brew install ffmpeg
           ```
        """
    else:  # Linux
        return """
        To install FFmpeg on Linux:
        
        Ubuntu/Debian:
        ```
        sudo apt update
        sudo apt install ffmpeg
        ```
        
        Fedora:
        ```
        sudo dnf install ffmpeg
        ```
        """

def download_youtube_video(video_url: str, save_path: str):
    """Download YouTube video with FFmpeg check."""
    if not check_ffmpeg():
        return None, "FFmpeg is not installed. Please install FFmpeg first."
    
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
            'retries': 3,
            'fragment_retries': 3,
            'ignoreerrors': False,
            'no_warnings': True,
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental'
            ]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            if info is None:
                return None, "Could not download video"
            
            video_path = os.path.join(temp_dir, ydl.prepare_filename(info))
            if not os.path.exists(video_path):
                video_path = video_path.rsplit('.', 1)[0] + '.mp4'
            
            if not os.path.exists(video_path):
                return None, "Downloaded file not found"
            
            final_path = os.path.join(save_path, os.path.basename(video_path))
            shutil.move(video_path, final_path)
            
            return final_path, "Video downloaded successfully!"
            
    except Exception as e:
        return None, f"Download error: {str(e)}"
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        progress_bar.empty()
        status_text.empty()

def main():
    st.set_page_config(page_title="YouTube Video Downloader", page_icon="▶️")
    
    st.title("▶️ YouTube Video Downloader")
    
    # Check FFmpeg installation
    if not check_ffmpeg():
        st.error("⚠️ FFmpeg is not installed!")
        st.markdown("### FFmpeg Installation Instructions")
        st.code(get_installation_instructions())
        st.warning("After installing FFmpeg, please restart this application.")
        return

    st.markdown("""
    Download YouTube videos in highest quality MP4 format.
    Please ensure you have the right to download the video content.
    """)

    video_url = st.text_input(
        "Enter YouTube video URL:",
        placeholder="https://youtube.com/watch?v=..."
    )
    
    if st.button("Download Video", type="primary"):
        if not video_url:
            st.error("Please enter a YouTube URL.")
            return
            
        if not ("youtube.com" in video_url or "youtu.be" in video_url):
            st.error("Please enter a valid YouTube URL.")
            return
            
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads_path, exist_ok=True)
        
        with st.spinner("Processing download..."):
            file_path, status = download_youtube_video(video_url, downloads_path)
        
        if file_path and os.path.exists(file_path):
            st.success(status)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
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

if __name__ == "__main__":
    main()
