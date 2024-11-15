# from pytube import YouTube
# from moviepy.editor import VideoFileClip
# import os
# import sys
# from urllib.parse import parse_qs, urlparse

# def extract_video_id(url):
#     parsed_url = urlparse(url)
#     if parsed_url.hostname == 'youtu.be':
#         return parsed_url.path[1:]
#     if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
#         if parsed_url.path == '/watch':
#             return parse_qs(parsed_url.query).get('v', [None])[0]
#     return None

# def download_youtube_segment(video_url, start_time, end_time, output_path):
#     try:
#         # Extract video ID from the URL
#         video_id = extract_video_id(video_url)
#         if not video_id:
#             raise Exception("Invalid YouTube URL")
        
#         clean_url = f"https://www.youtube.com/watch?v={video_id}"
#         print(f"Downloading video from: {clean_url}")
        
#         # Try to create YouTube object
#         print(clean_url)
#         yt = YouTube(clean_url)
#         print("Hello")
        
#         # Print available streams for debugging
#         print("Available streams:", yt.streams)
        
#         # Get highest quality MP4 stream
#         video_stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
        
#         print("hi")  # Ensure this line gets printed
#         if not video_stream:
#             raise Exception("No suitable video stream found")
        
#         print("Downloading full video...")
#         temp_path = "/home/saransh.sharma/Desktop/temp_video.mp4"  # Specify file path
#         video_path = video_stream.download(filename=temp_path)
        
#         print(f"Extracting segment from {start_time}s to {end_time}s...")
#         with VideoFileClip(temp_path) as video:
#             new = video.subclip(start_time, end_time)
#             new.write_videofile(output_path)
        
#         os.remove(temp_path)
#         print(f"Successfully created clip: {output_path}")
    
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     # Update video URL by removing the timestamp part
#     video_url = "https://www.youtube.com/watch?v=3c-iBn73dDE"  
#     start_time = 30  # Start time in seconds
#     end_time = 45    # End time in seconds
#     output_path = "highlight_clip.mp4"
    
#     download_youtube_segment(video_url, start_time, end_time, output_path)



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
        save_path = str(Path.home() / "Downloads")
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
