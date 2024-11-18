from pathlib import Path
import streamlit as st
import yt_dlp
import os
import imageio_ffmpeg

def download_youtube_video(video_url, save_path):
    try:
        st.write(f"Downloading video from: {video_url}")
        
        # Define options for yt_dlp
        ydl_opts = {
            'format': 'bestvideo+bestaudio',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': True,
            'retries': 3,
            'fragment_retries': 5,
            'ignoreerrors': False,
            'noprogress': False,
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental'
            ]
        }

        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.write("Downloading video...")
            info = ydl.extract_info(video_url, download=True)
            file_path = os.path.join(save_path, f"{info['title']}.mp4")
        
        st.write('Video downloaded successfully!')
        return file_path, "Video downloaded successfully!"
        
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None, f"An error occurred: {e}"

# Streamlit app UI
st.title("YouTube Video Downloader")

video_url = st.text_input("Enter YouTube video URL:")

if st.button("Download Video"):
    if video_url:
        save_path = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(save_path, exist_ok=True)
        file_path, status = download_youtube_video(video_url, save_path)
        
        if file_path:
            st.success(status)
            with open(file_path, "rb") as file:
                st.download_button(
                    label="Click here to download the video",
                    data=file,
                    file_name=os.path.basename(file_path),
                    mime="video/mp4"
                )
        else:
            st.error(status)
    else:
        st.error("Please enter a valid YouTube URL.")
