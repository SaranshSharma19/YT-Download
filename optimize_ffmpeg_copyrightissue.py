import json
import math
import shutil
from moviepy.editor import ColorClip, vfx, afx
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
import requests
import re
from typing import List, Tuple, Optional
import yt_dlp
import os
from urllib.parse import parse_qs, urlparse
import speech_recognition as sr
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import whisper
from functools import wraps
from time import time, sleep
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from googleapiclient import errors as googleapiclient_errors
from PIL import Image
from moviepy.config import change_settings
import glob
import random
from moviepy.editor import concatenate_videoclips
import subprocess  # Ensure to import subprocess
from google.cloud import speech_v1 as speech
change_settings({"IMAGEMAGICK_BINARY": "convert"})

class YoutubeSegmentDownloader:
    INSTAGRAM_REEL_SIZE = (1920, 1080)  # 9:16 aspect ratio
    YOUTUBE_SHORTS_SIZE = (1080, 1920)  # 9:16 aspect ratio
    
    RETRY_CONFIG = {
        'max_attempts': 5,    # Increased from 3
        'wait_min': 300,      # 5 minutes
        'wait_max': 1800,     # 30 minutes
    }
    
    def __init__(self, video_url: str, output_dir: str):
        self.video_url = video_url
        self.output_dir = output_dir
        self.video_id = self.get_video_id(video_url)
        self.temp_path = os.path.join(os.getcwd(), f"temp_full_video_{self.video_id}.mp4")
        self.downloaded = False
        self.model = whisper.load_model("base")  # Load Whisper model for subtitles
        self.video_title = self.get_video_title()


    @staticmethod
    def get_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        if 'youtu.be' in url:
            return url.split('/')[-1]
        elif 'youtube.com' in url:
            if 'v=' in url:
                return url.split('v=')[1].split('&')[0]
        return None

    def download_full_video(self):
        """Download the full video only once."""
        if not os.path.exists(self.temp_path) and not self.downloaded:
            try:
                print(f"Downloading video from: {self.video_url}")
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best[height>=1080]',
                    'outtmpl': self.temp_path,
                    'quiet': False,
                    'no_warnings': False,
                    'progress': True,
                    'socket_timeout': 30,
                    'retries': 5,
                    'fragment_retries': 5,
                    'retry_sleep': 3,
                    'http_chunk_size': 10485760,  # 10MB per chunk
                    'postprocessor_args': [
                        '-c:v', 'copy',
                        '-c:a', 'copy',
                    ],
                    'merge_output_format': 'mp4'
                }

                def progress_hook(d):
                    if d['status'] == 'downloading':
                        try:
                            print(f"\rDownloading... {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}", end='')
                        except:
                            print(f"\rDownloading... please wait", end='')
                    elif d['status'] == 'finished':
                        print("\nDownload completed!")

                ydl_opts['progress_hooks'] = [progress_hook]

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print("Starting download...")
                    ydl.download([self.video_url])
                self.downloaded = True
            except Exception as e:
                print(f"\nAn error occurred during download: {str(e)}")
                if os.path.exists(self.temp_path):
                    os.remove(self.temp_path)  # Clean up partial download
                raise

    def extract_segment(self, start_time: float, end_time: float, part_num: int) -> str:
        """Extract a segment from the downloaded video using ffmpeg."""
        output_filename = f"part_{part_num}_{self.video_id}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            print(f"Extracting segment from {start_time}s to {end_time}s...")
            command = [
                'ffmpeg', '-i', self.temp_path,
                '-ss', str(start_time), '-to', str(end_time),
                '-preset', 'ultrafast',
                '-c', 'copy', output_path
            ]
            subprocess.run(command, check=True)
            print(f"Successfully created clip: {output_path}")
            return output_path
        except Exception as e:
            print(f"An error occurred during segment extraction: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
                print("Cleaned up temporary files")
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {str(e)}")

    def resize_video_with_padding(self, input_path: str, platform: str) -> str:
        """
        Resize video for specific platform using ffmpeg, maintaining aspect ratio and adding padding.
        """
        target_size = self.INSTAGRAM_REEL_SIZE if platform == 'instagram' else self.YOUTUBE_SHORTS_SIZE
        output_path = input_path.replace('.mp4', f'_{platform}_padded.mp4')

        command = [
            'ffmpeg', '-i', input_path,
            '-vf', f"scale='if(gt(iw/ih,{target_size[0]}/{target_size[1]}),{target_size[0]},-1)':'if(gt(iw/ih,{target_size[0]}/{target_size[1]}),-1,{target_size[1]})',pad={target_size[0]}:{target_size[1]}:(ow-iw)/2:(oh-ih)/2",
            '-preset', 'ultrafast',
            '-c:a', 'aac', output_path
        ]
        subprocess.run(command, check=True)
        return output_path

    @staticmethod
    def _parse_timestamp(timestamp: str) -> float:
        """Convert SRT timestamp to seconds."""
        hours, minutes, seconds = timestamp.replace(',', '.').split(':')
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    def generate_subtitles(self, video_path: str) -> Tuple[str, str]:
        """Generate subtitles using Whisper and return subtitle file path and text."""
        print("Generating subtitles...")
        result = self.model.transcribe(video_path)
        
        # Generate SRT file
        srt_path = video_path.replace('.mp4', '.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = self._format_timestamp(segment['start'])
                end = self._format_timestamp(segment['end'])
                text = segment['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        print(result['text'])
        return srt_path, result['text']
    
    def add_subtitles_to_video(self, video_path: str, srt_path: str) -> str:
        """Add subtitles to video using ffmpeg."""
        output_path = video_path.replace('.mp4', '_subtitled.mp4')

        command = [
            'ffmpeg', '-i', video_path,
            '-vf', f"subtitles={srt_path}",
            '-preset', 'ultrafast',
            '-c:a', 'aac', output_path
        ]
        subprocess.run(command, check=True)

        return output_path

    @retry(
        stop=stop_after_attempt(RETRY_CONFIG['max_attempts']),
        wait=wait_exponential(multiplier=RETRY_CONFIG['wait_min'], max=RETRY_CONFIG['wait_max']),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: print(f"Retry attempt {retry_state.attempt_number} after {retry_state.outcome.exception()}")
    )
    def process_segment(self, start_time: float, end_time: float, part_num: int) -> None:
        """Process a single segment and save only the final copyright-protected versions."""
        try:
            # Create folder with video title (sanitized)
            safe_title = "".join(x for x in self.video_title if x.isalnum() or x in (' ', '-', '_')).rstrip()
            output_folder = os.path.join(self.output_dir, safe_title)
            os.makedirs(output_folder, exist_ok=True)
            
            # Extract segment
            segment_path = self.extract_segment(start_time, end_time, part_num)
            
            # Generate subtitles
            srt_path, transcript_text = self.generate_subtitles(segment_path)
            
            # Process for Instagram with maximum copyright protection
            instagram_version = self.apply_instagram_copyright_protection(
                self.resize_video_with_padding(segment_path, 'instagram')
            )
            # Add subtitles to Instagram version
            instagram_subtitled = self.add_subtitles_to_video(instagram_version, srt_path)
            
            # Process for YouTube with copyright protection
            youtube_version = self.apply_copyright_protection(
                self.resize_video_with_padding(segment_path, 'youtube')
            )
            # Add subtitles to YouTube version
            youtube_subtitled = self.add_subtitles_to_video(youtube_version, srt_path)
            
            # Save with new naming convention
            final_paths = {
                'instagram': os.path.join(output_folder, f'{safe_title}_part{part_num}_instagram.mp4'),
                'youtube': os.path.join(output_folder, f'{safe_title}_part{part_num}_youtube.mp4')
            }
            
            # Move final versions to output folder (now using subtitled versions)
            shutil.move(instagram_subtitled, final_paths['instagram'])
            shutil.move(youtube_subtitled, final_paths['youtube'])
            
            # Clean up all intermediate files
            for file in [segment_path, srt_path, instagram_version, youtube_version]:
                if os.path.exists(file):
                    os.remove(file)
            
            print(f"\nFinal videos saved in: {output_folder}")
            print(f"Instagram version: {final_paths['instagram']}")
            print(f"YouTube version: {final_paths['youtube']}")
            
        except Exception as e:
            print(f"Error processing segment {part_num}: {str(e)}")
            raise

    def apply_instagram_copyright_protection(self, video_path: str) -> str:
        """Apply subtle transformations to avoid copyright detection using ffmpeg."""
        output_path = video_path.replace('.mp4', '_protected.mp4')

        command = [
            'ffmpeg', '-i', video_path,
            '-filter_complex', "[0:v]setpts=PTS*1.02[v];[0:a]atempo=1.02[a]",
            '-preset', 'ultrafast',
            '-map', "[v]", '-map', "[a]",
            output_path
        ]
        subprocess.run(command, check=True)

        return output_path
    
    # def apply_instagram_copyright_protection(self, video_path: str) -> str:
    #     """Apply subtle transformations to avoid copyright detection using FFmpeg."""
    #     output_path = video_path.replace('.mp4', '_protected.mp4')

    #     # Prepare random transformation parameters
    #     speed_factor = random.choice([1.03, 0.97])
    #     audio_volume = random.uniform(0.95, 1.05)
    #     zoom_factor = 1.02
    #     mirror_effect = random.choice([True, False])  # Randomly decide to apply mirror effect
    #     watermark_text = "@postshuffle"

    #     # Create a watermark image using ImageMagick
    #     watermark_path = "watermark.png"
    #     subprocess.run([
    #         'convert',
    #         '-size', '400x100',
    #         'xc:transparent',
    #         '-gravity', 'center',
    #         '-fill', 'rgba(255,255,255,0.7)',
    #         '-font', 'DejaVu-Sans',  # Use a common font available on most systems
    #         '-pointsize', '40',
    #         '-annotate', '0', watermark_text,
    #         watermark_path
    #     ], check=True)

    #     # Prepare FFmpeg command
    #     filter_complex = f"[0:v]setpts={1/speed_factor}*PTS,scale=iw*{zoom_factor}:ih*{zoom_factor},"
        
    #     # Add mirror effect if chosen
    #     if mirror_effect:
    #         filter_complex += "hflip,"  # Apply horizontal flip

    #     # Add color adjustment
    #     filter_complex += "eq=brightness=0.1:contrast=1.1,"  # Slight brightness and contrast adjustment

    #     # Add watermark overlay
    #     filter_complex += f"[1:v]scale2ref=w=oh*mdar:h=0.2*ih[wm][base];[base][wm]overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2[outv]"

    #     # Complete filter chain
    #     filter_complex = f"{filter_complex}[outv];[0:a]volume={audio_volume}[a]"

    #     # Final command
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-i', video_path,
    #         '-i', watermark_path,
    #         '-filter_complex', filter_complex,
    #         '-map', '[outv]',
    #         '-map', '[a]',
    #         '-c:v', 'libx264',
    #         '-preset', 'medium',
    #         '-crf', '23',
    #         '-c:a', 'aac',
    #         '-b:a', '192k',
    #         output_path
    #     ]

    #     # Execute FFmpeg command
    #     try:
    #         subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"FFmpeg error: {e.stderr}")
    #         raise
    #     finally:
    #         # Clean up temporary watermark file
    #         if os.path.exists(watermark_path):
    #             os.unlink(watermark_path)

    #     return output_path


    def apply_copyright_protection(self, video_path: str) -> str:
        """Apply various transformations to help avoid copyright detection using ffmpeg."""
        output_path = video_path.replace('.mp4', '_protected.mp4')

        command = [
            'ffmpeg', '-i', video_path,
            '-filter_complex', "[0:v]setpts=PTS*1.02[v];[0:a]atempo=1.02[a]",
            '-preset', 'ultrafast',
            '-map', "[v]", '-map', "[a]",
            output_path
        ]
        subprocess.run(command, check=True)

        return output_path

    def get_video_title(self) -> str:
        """Get the title of the YouTube video."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(self.video_url, download=False)
                return info.get('title', 'Untitled Video')
        except Exception as e:
            print(f"Could not get video title: {str(e)}")
            return 'Untitled Video'

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def get_most_replayed_segments(video_url: str) -> List[Tuple[int, int, float]]:
    """Get all segments with their replay intensity scores."""
    video_id = YoutubeSegmentDownloader.get_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL")
        return []

    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response_text = response.text
        
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
                    
                    start_sec = start_ms / 1000
                    end_sec = (start_ms + duration_ms) / 1000
                    
                    segments.append((start_sec, end_sec, intensity))
                
                threshold = sorted([s[2] for s in segments], reverse=True)[len(segments)//10]
                most_replayed = [s for s in segments if s[2] >= threshold]
                
                merged_segments = []
                current_segment = None
                
                for segment in most_replayed:
                    if current_segment is None:
                        current_segment = list(segment)
                    elif segment[0] <= current_segment[1]:
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
    video_url = 'https://www.youtube.com/watch?v=HxjDgR8itZM&t=18s'
    output_dir = os.path.expanduser("~/Downloads")
    
    downloader = YoutubeSegmentDownloader(video_url, output_dir)
    segments = get_most_replayed_segments(video_url)
    
    if segments:
        downloader.download_full_video()
        
        print("\nProcessing segments:")
        for i, (start, end, intensity) in enumerate(segments, 1):
            if end - start >= 10:
                minutes_start = int(start) // 60
                seconds_start = int(start) % 60
                minutes_end = int(end) // 60
                seconds_end = int(end) % 60
                start_time = f"{minutes_start}:{seconds_start:02d}"
                end_time = f"{minutes_end}:{seconds_end:02d}"
                print(f"Processing segment {i}: {start_time} to {end_time}")
                downloader.process_segment(start, end, i)
        
        downloader.cleanup()
        print("\nAll segments processed and saved locally!")
    else:
        print("Failed to get most replayed segments")