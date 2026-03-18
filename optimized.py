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
            
    def resize_video_with_padding(self, input_path: str, platform: str) -> str:
        """
        Resize video for specific platform, maintaining aspect ratio and adding padding.
        """
        # Import PIL's Resampling at the top of the file
        from PIL import Image

        # Add this line to monkey patch PIL's ANTIALIAS
        if not hasattr(Image, 'ANTIALIAS'):
            Image.ANTIALIAS = Image.Resampling.LANCZOS

        target_size = self.INSTAGRAM_REEL_SIZE if platform == 'instagram' else self.YOUTUBE_SHORTS_SIZE
        output_path = input_path.replace('.mp4', f'_{platform}_padded.mp4')

        with VideoFileClip(input_path) as video:
            w, h = video.size
            aspect_ratio = w / h
            target_ratio = target_size[0] / target_size[1]

            if aspect_ratio > target_ratio:
                new_h = target_size[1]
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = target_size[0]
                new_h = int(new_w / aspect_ratio)

            resized_video = video.resize(width=new_w, height=new_h)
            background = ColorClip(size=target_size, color=(0, 0, 0)).set_duration(video.duration)
            final_clip = CompositeVideoClip([background, resized_video.set_position("center")])

            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
        return output_path

    def extract_segment(self, start_time: float, end_time: float, part_num: int) -> str:
        """Extract a segment from the downloaded video."""
        output_filename = f"part_{part_num}_{self.video_id}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            print(f"Extracting segment from {start_time}s to {end_time}s...")
            with VideoFileClip(self.temp_path) as video:
                new = video.subclip(start_time, end_time)
                new.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    preset='ultrafast',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
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
        """Add subtitles to video."""
        output_path = video_path.replace('.mp4', '_subtitled.mp4')
        
        # Add a font fallback mechanism
        try:
            font = 'Arial'
            # Test clip to verify font works
            test_clip = TextClip("Test", font=font, fontsize=24)
            test_clip.close()
        except:
            # Fallback to a system font that should exist
            font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # Common on Ubuntu
            if not os.path.exists(font):
                # Second fallback
                font = 'DejaVuSans'  # Try using just the font name
        
        with VideoFileClip(video_path) as video:
            subtitle_clips = []
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_data = f.read().strip().split('\n\n')
                for block in srt_data:
                    lines = block.split('\n')
                    if len(lines) >= 3:
                        timestamps = lines[1].split(' --> ')
                        start_time = self._parse_timestamp(timestamps[0])
                        end_time = self._parse_timestamp(timestamps[1])
                        text = ' '.join(lines[2:])
                        
                        txt_clip = (TextClip(text, 
                                           font=font,
                                           fontsize=24, 
                                           color='white',
                                           stroke_color='black',
                                           stroke_width=1)
                                  .set_position(('center', 'bottom'))
                                  .set_duration(end_time - start_time)
                                  .set_start(start_time))
                        
                        subtitle_clips.append(txt_clip)
        
            final_clip = CompositeVideoClip([video] + subtitle_clips)
            final_clip.write_videofile(output_path)
        
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
            instagram_version = self.apply_instagram_copyright_protection(self.resize_video_with_padding(segment_path, 'instagram'))
            # Add subtitles to Instagram version
            instagram_subtitled = self.add_subtitles_to_video(instagram_version, srt_path)
            
            # Process for YouTube with copyright protection
            youtube_version = self.apply_copyright_protection(self.resize_video_with_padding(segment_path, 'youtube'))
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
        """Apply subtle transformations to avoid copyright detection."""
        output_path = video_path.replace('.mp4', '_protected.mp4')
        
        with VideoFileClip(video_path) as video:
            # 1. Minimal speed adjustment (±3%)
            speed_factor = random.uniform(0.97, 1.03)
            modified = video.fx(vfx.speedx, speed_factor)
            
            # 2. Subtle audio modifications
            if modified.audio is not None:
                modified = modified.set_audio(
                    modified.audio.volumex(random.uniform(0.95, 1.05))
                )
            
            # 3. Very slight zoom
            zoom_factor = random.uniform(1.02, 1.05)
            modified = modified.fx(vfx.resize, zoom_factor)
            
            # 4. Minimal crop
            w, h = modified.size
            crop_margin = int(w * 0.02)  # 2% margin
            modified = modified.crop(
                x1=random.randint(0, crop_margin),
                y1=random.randint(0, crop_margin),
                x2=w-random.randint(0, crop_margin),
                y2=h-random.randint(0, crop_margin)
            )
            
            # 5. Very minimal color adjustments (±1%)
            # color_factor = random.uniform(0.99, 1.01)
            # modified = modified.fx(vfx.colorx, color_factor)
            
            # # 6. Extremely subtle brightness/contrast (±1%)
            # modified = modified.fx(vfx.lum_contrast,
            #                     lum=random.uniform(-0.01, 0.01),
            #                     contrast=random.uniform(0.99, 1.01))
            
            # 7. Add watermark
            watermark_text = "@postshuffle"
            txt_clip = (TextClip(watermark_text,
                                fontsize=40,
                                color='white',
                                stroke_color='black',
                                stroke_width=1,
                                font='Arial-Bold')
                    .set_duration(modified.duration)
                    .set_opacity(0.7))
            
            margin = 40
            x_pos = random.randint(margin, w - txt_clip.w - margin)
            y_pos = random.randint(margin, h - txt_clip.h - margin)
            
            # Add subtle moving watermark
            watermark_moving = txt_clip.set_position(
                lambda t: (x_pos + math.sin(t) * 10,  # Reduced movement
                        y_pos + math.cos(t) * 10)
            )
            
            # Composite video with watermark
            modified = CompositeVideoClip([modified, watermark_moving])
            
            # 8. Add small black margins
            margin_size = random.randint(10, 20)
            margin_color = (0, 0, 0)
            final_clip = modified.margin(mar=margin_size, color=margin_color)
            
            # Write with balanced quality settings
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                bitrate='6000k',
                audio_bitrate='192k',
                preset='medium',
                threads=4
            )
        
        return output_path


    def apply_copyright_protection(self, video_path: str) -> str:
        """Apply various transformations to help avoid copyright detection."""
        output_path = video_path.replace('.mp4', '_protected.mp4')
        
        with VideoFileClip(video_path) as video:
            # 1. Slight speed adjustment (1.03x or 0.97x)
            speed_factor = random.choice([1.03, 0.97])
            modified = video.fx(vfx.speedx, speed_factor)
            
            # 2. Audio modifications
            if modified.audio is not None:
                modified = modified.set_audio(
                    modified.audio.volumex(random.uniform(0.95, 1.05))
                )
            
            # 3. Add slight zoom (1.02x)
            modified = modified.fx(vfx.resize, 1.02)
            
            # 4. Add subtle video effects
            modified = modified.fx(vfx.colorx, 1.1)  # Slight color adjustment
            
            # 5. Add mirror effect to a small portion
            if random.choice([True, False]):
                modified = modified.fx(vfx.mirror_x)
            
            # 6. Add watermark
            watermark_text = "@postshuffle"
            w, h = modified.size
            txt_clip = (TextClip(watermark_text,
                                fontsize=50,
                                color='white',
                                stroke_color='black',
                                stroke_width=2,
                                font='Arial-Bold')
                       .set_duration(modified.duration)
                       .set_opacity(0.8))
            
            # Random position for watermark (avoiding edges)
            margin = 50
            x_pos = random.randint(margin, w - txt_clip.w - margin)
            y_pos = random.randint(margin, h - txt_clip.h - margin)
            
            # Add moving watermark effect
            watermark_moving = txt_clip.set_position(
                lambda t: (x_pos + math.sin(t) * 20,  # Gentle horizontal movement
                          y_pos + math.cos(t) * 20)   # Gentle vertical movement
            )
            
            # Composite video with watermark
            modified = CompositeVideoClip([modified, watermark_moving])
            
            # 7. Add black margins
            margin_size = random.randint(20, 40)
            margin_color = (0, 0, 0)  # Black color
            final_clip = modified.margin(mar=margin_size, color=margin_color)
            
            # Write with higher quality settings
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                bitrate='8000k',
                audio_bitrate='384k',
                preset='slow',
                threads=4
            )
        
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
    video_url = 'https://www.youtube.com/watch?v=bUxd3jqCr94'
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