import streamlit as st
import yt_dlp
import os
import tempfile

def download_youtube_video(video_url):
    try:
        print(f"Downloading video from: {video_url}")
        
        # Create a temporary directory to store the downloaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestvideo+bestaudio',  # Download best video and best audio quality
                'merge_output_format': 'mp4',     # Ensure merged output is in mp4 format
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),  # Save with title as filename in temp dir
                'quiet': False,
                'no_warnings': True,
                'concurrent-fragments': 5,        # Use 5 connections per file fragment
                'retries': 3,                     # Retry 3 times upon failure
                'fragment-retries': 5,            # Retry fragments up to 5 times if they fail
                'external-downloader': 'aria2c',  # Use aria2c for faster downloads (requires aria2c installed)
                'external-downloader-args': ['-x', '16', '-s', '16', '-k', '1M'],  # aria2c optimizations
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Downloading video...")
                ydl.download([video_url])
                
            # Get the downloaded file path (it should be saved in the temp directory)
            video_title = video_url.split("v=")[-1]  # Extract video ID or title
            file_path = os.path.join(temp_dir, f"{video_title}.mp4")
            status = "Video downloaded successfully!"
            
            return file_path, status
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, "Download failed. Please check the URL or try again later."

st.title("YouTube Video Downloader")

video_url = st.text_input("Enter YouTube video URL:")

if st.button("Download Video"):
    if video_url:
        # Call the download function and get the file path and status
        file_path, status = download_youtube_video(video_url)
        
        if file_path:
            st.success(status)
            
            # Open the file in binary mode and create a download button
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
