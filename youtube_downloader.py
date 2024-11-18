from pathlib import Path
import streamlit as st
import yt_dlp
import os

def download_youtube_video(video_url, save_path):
    try:
        print(f"Downloading video from: {video_url}")

        ydl_opts = {
            'format': 'bestvideo+bestaudio',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': True,
            'concurrent-fragments': 5,
            'retries': 3,
            'fragment-retries': 5,
            'external-downloader': 'aria2c',
            'external-downloader-args': ['-x', '16', '-s', '16', '-k', '1M'],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading video...")
            info = ydl.extract_info(video_url, download=True)
            file_path = os.path.join(save_path, f"{info['title']}.mp4")
        
        print('Video downloaded successfully!')
        return file_path, "Video downloaded successfully!"
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, f"An error occurred: {e}"

st.title("YouTube Video Downloader")

video_url = st.text_input("Enter YouTube video URL:")

if st.button("Download Video"):
    if video_url:
        save_path = os.path.join(os.path.expanduser("~"), "Downloads")
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
