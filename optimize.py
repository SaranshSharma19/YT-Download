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

from typing import Tuple, Optional, Dict, Any, List
import re

class YouTubeSubtitleDownloader:
    def __init__(self, video_url: str):
        """Initialize with a YouTube video URL."""
        self.video_url = video_url
        self.video_id = self.extract_video_id(video_url)
    
    def extract_video_id(self, url: str) -> str:
        """Extract the video ID from a YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
            r'(?:youtube\.com\/embed\/)([^&\n?]+)',
            r'(?:youtube\.com\/v\/)([^&\n?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get video metadata."""
        ydl_opts = {'quiet': True, 'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(self.video_url, download=False)
    
    def list_available_subtitles(self) -> Tuple[Dict[str, list], Dict[str, list]]:
        """List available subtitles."""
        info = self.get_video_info()
        return info.get('subtitles', {}), info.get('automatic_captions', {})
    
    def download_subtitle(self, output_dir: str = ".", preferred_langs: Optional[List[str]] = None,
                          prefer_auto: bool = False, output_filename: Optional[str] = None) -> Tuple[str, str]:
        """Download the best available subtitle file."""
        if preferred_langs is None:
            preferred_langs = ['hi', 'en']
        
        os.makedirs(output_dir, exist_ok=True)
        info = self.get_video_info()
        
        regular_subs, auto_subs = info.get('subtitles', {}), info.get('automatic_captions', {})
        sources = [('auto', auto_subs), ('regular', regular_subs)] if prefer_auto else [('regular', regular_subs), ('auto', auto_subs)]
        
        subtitle_url, subtitle_lang, subtitle_type = None, None, None
        
        for source_type, source in sources:
            for lang in preferred_langs:
                if lang in source:
                    subtitle_lang, subtitle_type = lang, source_type
                    formats = source[lang]
                    subtitle_url = next((fmt['url'] for fmt in formats if fmt.get('ext') == 'srt'), formats[0]['url'])
                    break
            if subtitle_url:
                break
        
        if not subtitle_url:
            print("No subtitles found.")
            return '', ''
        
        subtitle_filename = output_filename or f"{self.video_id}_{subtitle_lang}_{subtitle_type}.srt"
        subtitle_path = os.path.join(output_dir, subtitle_filename)
        
        try:
            response = requests.get(subtitle_url)
            response.raise_for_status()
            
            with open(subtitle_path, 'wb') as f:
                f.write(response.content)
            
            with open(subtitle_path, 'r', encoding='utf-8', errors='replace') as f:
                subtitle_text = f.read()
            
            return subtitle_path, subtitle_text
        except requests.RequestException as e:
            print(f"Error downloading subtitles: {e}")
            return '', ''

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
        self.subtitle_downloader = YouTubeSubtitleDownloader(video_url)


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
        """Extract a segment from the downloaded video using."""
        output_filename = f"part_{part_num}_{self.video_id}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            print(f"Extracting segment from {start_time}s to {end_time}s...")
            command = [
                'ffmpeg', '-y', '-i', self.temp_path,
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
        Resize video for specific platform using, maintaining aspect ratio and adding padding.
        """
        target_size = self.INSTAGRAM_REEL_SIZE if platform == 'instagram' else self.YOUTUBE_SHORTS_SIZE
        output_path = input_path.replace('.mp4', f'_{platform}_padded.mp4')

        command = [
            'ffmpeg', '-y', '-i', input_path,
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
        """Generate subtitles using downloaded or Whisper-generated captions."""
        output_dir = os.path.dirname(video_path)
        output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.srt"
        
        srt_path, subtitle_text = self.subtitle_downloader.download_subtitle(output_dir, ['hi', 'en'], False, output_filename)
        
        if not srt_path:
            print("Falling back to Whisper transcription...")
            result = self.model.transcribe(video_path)
            
            srt_path = video_path.replace('.mp4', '.srt')
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result['segments'], 1):
                    start, end, text = self._format_timestamp(segment['start']), self._format_timestamp(segment['end']), segment['text'].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            subtitle_text = result['text']
            print("subtitle_text" , subtitle_text)
        
        return srt_path, subtitle_text
    
    def add_subtitles_to_video(self, video_path: str, subtitle_data: List[Dict[str, Any]]) -> str:
        """Add subtitles to video using FFmpeg."""
        output_path = video_path.replace('.mp4', '_subtitled.mp4')
        srt_path = f"{video_path.replace('.mp4', '')}.srt"
        
        # Create proper SRT file from the JSON data
        self.create_srt_from_json(subtitle_data, srt_path)
        
        # Check if the subtitle file exists
        if not os.path.exists(srt_path):
            print(f"Subtitle file not found: {srt_path}")
            raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
        
        print(f"Adding subtitles from: {srt_path} to video: {video_path}")
        
        # Check if the output file exists and remove it
        if os.path.exists(output_path):
            os.remove(output_path)

        # FFmpeg command
        command = [
            'ffmpeg', '-y', '-i', video_path,
            # '-vf', f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=1,Shadow=1,Alignment=2',fade=t=out:st=3:d=1:alpha=1,fade=t=in:st=0:d=1",
            '-vf', f"subtitles=srt_path:force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=1,Shadow=1,Alignment=2,MarginV=0',",
            '-preset', 'ultrafast',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(command, check=True)
            print(f"Successfully added subtitles to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {str(e)}")
            # Use a simpler approach as fallback
            simple_command = [
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f"subtitles={srt_path}",
                '-preset', 'ultrafast',
                '-c:a', 'copy',
                output_path
            ]
            try:
                subprocess.run(simple_command, check=True)
                print(f"Successfully added subtitles using simple method")
                return output_path
            except Exception as e2:
                print(f"All subtitle methods failed: {str(e2)}")
                return video_path  # Return original video if all methods fail

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
            srt_path, subtitle_json = self.generate_subtitles(segment_path)
            
            # Parse the subtitle JSON if it's a string
            text_events = []
            if isinstance(subtitle_json, str) and subtitle_json.startswith('{'):
                try:
                    subtitle_data = json.loads(subtitle_json)
                    # Extract events from the JSON structure
                    events = subtitle_data.get('events', [])
                    
                    # Filter and adjust events to match the segment timing
                    segment_duration = end_time - start_time
                    for event in events:
                        if 'segs' in event:
                            # Combine all text segments
                            text = ''.join(seg.get('utf8', '') for seg in event['segs'] if 'utf8' in seg)
                            if text:
                                # Adjust timestamp to be relative to the segment
                                event_start = event.get('tStartMs', 0) / 1000  # Convert to seconds
                                
                                # Only include events that overlap with our segment
                                if (event_start + (event.get('dDurationMs', 0) / 1000) >= start_time and 
                                    event_start <= end_time):
                                    
                                    # Adjust timestamps to be relative to segment start
                                    adjusted_start = max(0, event_start - start_time) * 1000  # Convert back to ms
                                    adjusted_duration = min(
                                        (event.get('dDurationMs', 0) / 1000),  # Original duration
                                        (end_time - max(start_time, event_start))  # Capped by segment end
                                    ) * 1000  # Convert back to ms
                                    
                                    text_events.append({
                                        'tStartMs': int(adjusted_start),
                                        'dDurationMs': int(adjusted_duration),
                                        'utf8': text
                                    })
                except json.JSONDecodeError:
                    # If not valid JSON, create a single subtitle event
                    text_events = [{
                        'tStartMs': 0,  # Start at the beginning of the segment
                        'dDurationMs': int(segment_duration * 1000),
                        'utf8': subtitle_json
                    }]
            else:
                # If not JSON, create a single subtitle event
                text_events = [{
                    'tStartMs': 0,  # Start at the beginning of the segment
                    'dDurationMs': int((end_time - start_time) * 1000),
                    'utf8': subtitle_json if isinstance(subtitle_json, str) else "No subtitles available"
                }]
            
            # If no events were found, create a dummy event
            if not text_events:
                text_events = [{
                    'tStartMs': 0,
                    'dDurationMs': int((end_time - start_time) * 1000),
                    'utf8': "No subtitles available for this segment"
                }]
            
            # Process for Instagram with maximum copyright protection
            instagram_version = self.apply_instagram_copyright_protection(
                self.resize_video_with_padding(segment_path, 'instagram')
            )
            # Add subtitles to Instagram version
            instagram_subtitled = self.add_subtitles_to_video(instagram_version, text_events)
            
            # Process for YouTube with copyright protection
            youtube_version = self.apply_copyright_protection(
                self.resize_video_with_padding(segment_path, 'youtube')
            )
            # Add subtitles to YouTube version
            youtube_subtitled = self.add_subtitles_to_video(youtube_version, text_events)
            
            # Save with new naming convention
            final_paths = {
                'instagram': os.path.join(output_folder, f'{safe_title}_part{part_num}_instagram.mp4'),
                'youtube': os.path.join(output_folder, f'{safe_title}_part{part_num}_youtube.mp4')
            }
            
            # Move final versions to output folder
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
        """Apply subtle transformations to avoid copyright detection using."""
        output_path = video_path.replace('.mp4', '_protected.mp4')

        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-filter_complex', "[0:v]setpts=PTS*1.02[v];[0:a]atempo=1.02[a]",
            '-preset', 'ultrafast',
            '-map', "[v]", '-map', "[a]",
            output_path
        ]
        subprocess.run(command, check=True)

        return output_path
    
    def apply_copyright_protection(self, video_path: str) -> str:
        """Apply various transformations to help avoid copyright detection using ffmpeg."""
        output_path = video_path.replace('.mp4', '_protected.mp4')

        command = [
            'ffmpeg', '-y', '-i', video_path,
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
    
    def create_srt_from_json(self, subtitle_data: List[Dict[str, Any]], output_path: str) -> None:
        """Create a properly timed SRT file from the JSON subtitle data."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                subtitle_index = 1
                
                # Sort events by start time to ensure correct order
                if isinstance(subtitle_data, list):
                    # If it's a list of events, sort them
                    subtitle_events = sorted(subtitle_data, key=lambda x: x.get('tStartMs', 0))
                else:
                    # If it's a single event or something else, put it in a list
                    subtitle_events = [subtitle_data]
                
                # Process each subtitle event
                for event in subtitle_events:
                    # Skip if this isn't a subtitle event with text
                    if not isinstance(event, dict) or 'utf8' not in event:
                        continue
                    
                    # Get the timestamps
                    start_time = event.get('tStartMs', 0) / 1000  # Convert to seconds
                    duration = event.get('dDurationMs', 5000) / 1000  # Default to 5 seconds if not specified
                    end_time = start_time + duration
                    
                    # Get the text
                    text = event.get('utf8', '').strip()
                    if not text:
                        continue  # Skip empty subtitles
                    
                    # Format timestamps in SRT format: HH:MM:SS,mmm
                    start_str = self._format_timestamp(start_time)
                    end_str = self._format_timestamp(end_time)
                    
                    # Write the subtitle entry
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{text}\n\n")
                    
                    subtitle_index += 1
                
                # If no subtitles were written, add a dummy subtitle
                if subtitle_index == 1:
                    f.write("1\n00:00:00,000 --> 00:00:05,000\nNo subtitles available\n\n")
            
            print(f"Created SRT file: {output_path}")
        except Exception as e:
            print(f"Error creating SRT file: {str(e)}")
            # Create a basic SRT file as fallback
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("1\n00:00:00,000 --> 00:00:05,000\nSubtitle generation failed\n\n")
                print("Created fallback SRT file")
            except:
                print("Failed to create even a fallback SRT file")

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

def convert_to_srt(subtitle_data: List[Dict[str, Any]], output_path: str) -> None:
    """Convert subtitle data to SRT format and save to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for index, segment in enumerate(subtitle_data, start=1):
                start_time = segment.get('tStartMs', 0) / 1000  # Convert milliseconds to seconds
                duration = segment.get('dDurationMs', 0) / 1000  # Convert milliseconds to seconds
                end_time = start_time + duration
                
                # Format timestamps ensuring correct format (HH:MM:SS,mmm)
                start_time_str = format_timestamp(start_time)
                end_time_str = format_timestamp(end_time)
                
                # Clean the text and ensure it's not empty
                text = segment.get('utf8', '').strip()
                if not text:
                    text = "..."  # Placeholder for empty text
                
                # Write to SRT file with strict formatting
                f.write(f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n\n")
        
        # Verify and fix the SRT file if needed
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix common SRT formatting issues
        if not content.strip():
            # Create a basic subtitle if the file is empty
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("1\n00:00:00,000 --> 00:00:05,000\nSubtitles unavailable\n\n")
            print("Created fallback subtitle file")
        
        print(f"SRT file created successfully: {output_path}")
    except Exception as e:
        print(f"Error creating SRT file: {str(e)}")
        
        # Create emergency subtitle file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("1\n00:00:00,000 --> 00:00:05,000\nSubtitles unavailable\n\n")
            print("Created emergency subtitle file after error")
        except:
            print("Failed to create even emergency subtitle file")

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=fIQajr_iooY'
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
    