import streamlit as st
import yt_dlp
import os

def download_youtube_video(video_url, save_path):
    try:
        st.write(f"Downloading video from: {video_url}")
        
        ydl_opts = {
            'format': 'bestvideo+bestaudio',  # Download best video and best audio quality
            'merge_output_format': 'mp4',     # Ensure merged output is in mp4 format
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),  # Save with title as filename
            'quiet': False,
            'no_warnings': True,
            'concurrent-fragments': 5,       # Use 5 connections per file fragment
            'retries': 3,                    # Retry 3 times upon failure
            'fragment-retries': 5,           # Retry fragments up to 5 times if they fail
            'external-downloader': 'aria2c', # Use aria2c for faster downloads (requires aria2c installed)
            'external-downloader-args': ['-x', '16', '-s', '16', '-k', '1M'],  # aria2c optimizations
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.write("Downloading video...")
            ydl.download([video_url])
            
        st.success('Video downloaded successfully!')
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app interface
st.title("YouTube Video Downloader")

video_url = st.text_input("Enter the YouTube video URL:")
save_path = st.text_input("Enter the save path (e.g., Downloads folder path):", value=os.path.expanduser("~/Downloads"))

if st.button("Download Video"):
    if video_url and save_path:
        # Ensure save path exists
        if not os.path.exists(save_path):
            st.error("The specified path does not exist.")
        else:
            download_youtube_video(video_url, save_path)
    else:
        st.warning("Please provide both a video URL and a download path.")
