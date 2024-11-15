import streamlit as st
import yt_dlp
import os

def download_youtube_video(video_url, save_path):
    try:
        print(f"Downloading video from: {video_url}")
        
        # yt-dlp options to specify format and output path
        # ydl_opts = {
        #     'format': 'best[ext=mp4]',  # Get best mp4 quality
        #     'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),  # Save with title as filename
        #     'quiet': False,
        #     'no_warnings': True,
        # }

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
            print("Downloading video...")
            ydl.download([video_url])
            
        print('Video downloaded successfully!')
        
    except Exception as e:
        print(f"An error occurred: {e}")

st.title("YouTube Video Downloader")

video_url = st.text_input("Enter YouTube video URL:")

if st.button("Download Video"):
    if video_url:
        save_path = os.path.join(os.path.expanduser("~"), "Downloads")
        status = download_youtube_video(video_url, save_path)
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
