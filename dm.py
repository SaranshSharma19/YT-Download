import json
import requests
import re
from typing import List, Tuple, Optional
from moviepy.editor import VideoFileClip
import yt_dlp
import os
import sys
from urllib.parse import parse_qs, urlparse

def download_youtube_segment(video_url, start_time, end_time, output_dir, part_num):
    # Generate video name from URL
    video_id = get_video_id(video_url)
    output_filename = f"part_{part_num}_{video_id}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    temp_path = os.path.join(os.getcwd(), "temp_full_video.mp4")
    
    # Check if full video is already downloaded
    if not os.path.exists(temp_path):
        try:
            print(f"Downloading video from: {video_url}")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': temp_path,     
                'quiet': False,            
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Downloading full video...")
                ydl.download([video_url])
        except Exception as e:
            print(f"An error occurred during download: {str(e)}")
            raise

    try:
        print(f"Extracting segment from {start_time}s to {end_time}s...")
        with VideoFileClip(temp_path) as video:
            new = video.subclip(start_time, end_time)            
            new.write_videofile(output_path, 
                              codec='libx264',
                              audio_codec='aac',
                              temp_audiofile='temp-audio.m4a',
                              remove_temp=True)
        
        print(f"Successfully created clip: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {str(e)}")

# if __name__ == "__main__":
#     video_url = "https://www.youtube.com/watch?v=3c-iBn73dDE"
#     start_time = 1
#     end_time = 45 
#     output_path = "highlight_clip.mp4"
    
#     download_youtube_segment(video_url, start_time, end_time, output_path)



def get_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    if 'youtu.be' in url:
        return url.split('/')[-1]
    elif 'youtube.com' in url:
        if 'v=' in url:
            return url.split('v=')[1].split('&')[0]
    return None

def get_most_replayed_segments(video_url: str) -> List[Tuple[int, int, float]]:
    """
    Get all segments with their replay intensity scores.
    Returns list of tuples: (start_time_seconds, end_time_seconds, intensity_score)
    """
    video_id = get_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL")
        return []

    # YouTube API endpoint for getting video info
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response_text = response.text
        
        # Extract the ytInitialData from the response
        data_match = re.search(r'ytInitialData\s*=\s*({.+?});', response_text)
        if not data_match:
            print("Could not find YouTube initial data")
            return []
            
        data = json.loads(data_match.group(1))
        
        mutations = data.get('frameworkUpdates', {}).get('entityBatchUpdate', {}).get('mutations', [])
        
        for mutation in mutations:
            if 'payload' in mutation and 'macroMarkersListEntity' in mutation['payload']:
                markers = mutation['payload']['macroMarkersListEntity']['markersList']['markers']
                
                segments = []
                for marker in markers:
                    start_ms = int(marker['startMillis'])
                    duration_ms = int(marker['durationMillis'])
                    intensity = float(marker['intensityScoreNormalized'])
                    
                    # Convert to seconds
                    start_sec = start_ms / 1000
                    end_sec = (start_ms + duration_ms) / 1000
                    
                    segments.append((start_sec, end_sec, intensity))
                
                # Find segments with high replay intensity (e.g., top 10% of intensities)
                threshold = sorted([s[2] for s in segments], reverse=True)[len(segments)//10]
                most_replayed = [s for s in segments if s[2] >= threshold]
                
                merged_segments = []
                current_segment = None
                
                for segment in most_replayed:
                    if current_segment is None:
                        current_segment = list(segment)
                    elif segment[0] <= current_segment[1]:
                        # Merge segments
                        current_segment[1] = segment[1]
                        current_segment[2] = max(current_segment[2], segment[2])
                    else:
                        merged_segments.append(tuple(current_segment))
                        current_segment = list(segment)
                
                if current_segment:
                    merged_segments.append(tuple(current_segment))
                
                return merged_segments
        
        print("Could not find heatmap data in response")
        return []

    except Exception as e:
        print(f"Error: {str(e)}")
        return []

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=Ux63B5QxCQ8'
    segments = get_most_replayed_segments(video_url)
    output_dir = os.path.expanduser("~/Downloads")
    
    if segments:
        print("\nMost replayed segments:")
        for i, (start, end, intensity) in enumerate(segments, 1):
            minutes_start = int(start) // 60
            seconds_start = int(start) % 60
            minutes_end = int(end) // 60
            seconds_end = int(end) % 60
            start_time = f"{minutes_start}:{seconds_start:02d}"
            end_time = f"{minutes_end}:{seconds_end:02d}"
            print(f"Time: {start_time} to {end_time} (Intensity: {intensity:.2f})")
            download_youtube_segment(video_url, start, end, output_dir, i)
            
    else:
        print("Failed to get most replayed segments")

    # Clean up temp file after all segments are processed
    temp_path = os.path.join(os.getcwd(), "temp_full_video.mp4")
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {str(e)}")
