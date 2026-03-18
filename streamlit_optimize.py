import streamlit as st
import os
import yt_dlp
import whisper
import json
import re
import requests
from typing import List, Tuple, Dict, Any
import subprocess
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class YouTubeToolkit:
    def __init__(self):
        # Initialize Whisper model for subtitle generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=device)
    
    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from YouTube URL."""
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

    def download_full_video(self, video_url: str, output_dir: str) -> str:
        """Download full YouTube video."""
        try:
            video_id = self.extract_video_id(video_url)
            output_path = os.path.join(output_dir, f"{video_id}_full_video.mp4")
            
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best[height>=1080]',
                'outtmpl': output_path,
                'quiet': False,
                'no_warnings': False,
                'progress': True,
                'socket_timeout': 30,
                'retries': 5,
                'fragment_retries': 5,
                'retry_sleep': 3,
                'http_chunk_size': 10485760,  # 10MB per chunk
                'merge_output_format': 'mp4'
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            return output_path
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return ""


    def clean_subtitle_text(self, text: str) -> str:
        """
        Clean subtitle text by:
        - Removing timestamps
        - Removing line numbers
        - Removing extra whitespaces
        - Removing newline characters
        """
        # Remove timestamp patterns
        text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
        
        # Remove line numbers
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespaces and newlines
        text = ' '.join(text.split())
        
        # Split lines by newline characters and format them
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Only keep non-empty lines
                cleaned_lines.append(line.strip())        
        # Join cleaned lines with newline characters
        return '\n'.join(cleaned_lines)

    def download_subtitles(self, video_url: str, output_dir: str, preferred_langs: List[str] = None) -> Dict[str, str]:
        """Download subtitles from YouTube video and save as a .txt file."""
        if preferred_langs is None:
            preferred_langs = ['en', 'hi']
        
        try:
            video_id = self.extract_video_id(video_url)
            
            # Use yt-dlp to get subtitle information
            ydl_opts = {
                'writesubtitles': True, 
                'writeautomaticsub': True,
                'quiet': True,
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
            
            regular_subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})
            
            # Try to find subtitles
            subtitle_path = ""
            subtitle_text = ""
            
            for lang in preferred_langs:
                # First try regular subtitles
                if lang in regular_subs:
                    subtitle_url = regular_subs[lang][0]['url']
                    subtitle_path = os.path.join(output_dir, f"{video_id}_{lang}_subtitles.srt")
                    subtitle_type = "regular"
                    break
                
                # Then try automatic captions
                elif lang in auto_subs:
                    subtitle_url = auto_subs[lang][0]['url']
                    subtitle_path = os.path.join(output_dir, f"{video_id}_{lang}_auto_subtitles.srt")
                    subtitle_type = "auto"
                    break
            
            # Download subtitle file
            if subtitle_url:
                response = requests.get(subtitle_url)
                with open(subtitle_path, 'wb') as f:
                    f.write(response.content)
                
                # Read subtitle text and extract relevant utf8 values
                with open(subtitle_path, 'r', encoding='utf-8') as f:
                    subtitle_lines = f.readlines()
                
                # Create a .txt file to store the cleaned subtitles
                txt_file_path = os.path.join(output_dir, f"{video_id}_subtitles.txt")
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    for line in subtitle_lines:
                        # Check if the line contains utf8 values
                        if 'utf8' in line:
                            # Extract the utf8 value
                            utf8_value = re.search(r'"utf8":\s*"([^"]+)"', line)
                            if utf8_value:
                                # Clean the utf8 value by removing musical notes and newlines
                                cleaned_value = utf8_value.group(1).replace('♪', '').replace('\\n', ' ').strip()
                                txt_file.write(cleaned_value + '\n')  # Write cleaned line to .txt file
                
                subtitle_text = self.clean_subtitle_text(' '.join(subtitle_lines))  # Clean the original subtitle text
            
            # Fallback to Whisper transcription if no subtitles found
            if not subtitle_path:
                st.warning("No subtitles found. Generating subtitles using Whisper...")
                temp_video_path = self.download_full_video(video_url, output_dir)
                
                if temp_video_path:
                    result = self.whisper_model.transcribe(temp_video_path)
                    subtitle_path = temp_video_path.replace('.mp4', '.srt')
                    
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        for i, segment in enumerate(result['segments'], 1):
                            start = self._format_timestamp(segment['start'])
                            end = self._format_timestamp(segment['end'])
                            text = segment['text'].strip()
                            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                    
                    subtitle_text = result['text']
                    subtitle_text = self.clean_subtitle_text(subtitle_text)  # Clean Whisper-generated subtitles
                    subtitle_type = "whisper"
            
            return {
                "path": txt_file_path,  # Return the path of the .txt file
                "text": subtitle_text,
                "type": subtitle_type
            }
        
        except Exception as e:
            st.error(f"Error downloading subtitles: {str(e)}")
            return {}
        
    def download_segment(self, video_url: str, start: float, end: float, output_dir: str) -> str:
        """Download a specific segment of the YouTube video."""
        try:
            video_id = self.extract_video_id(video_url)
            
            # First, download the full video
            full_video_path = self.download_full_video(video_url, output_dir)
            
            if not full_video_path:
                st.error("Failed to download full video")
                return ""

            # Prepare output path for segment
            output_path = os.path.join(output_dir, f"{video_id}_segment_{int(start)}_{int(end)}.mp4")
            
            # Use ffmpeg to extract the segment
            command = [
                'ffmpeg', 
                '-i', full_video_path,  # Input full video path
                '-ss', str(start),       # Start time
                '-to', str(end),         # End time
                '-c', 'copy',            # Use stream copy to avoid re-encoding
                output_path              # Output path
            ]
            
            # Run the ffmpeg command
            result = subprocess.run(command, capture_output=True, text=True)
            
            # Check if the command was successful
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                st.error(f"FFmpeg error: {result.stderr}")
                return ""
        
        except Exception as e:
            st.error(f"Error downloading segment: {str(e)}")
            return ""
        
    def extract_segment(self, start_time: float, end_time: float, part_num: int) -> str:
        """Extract a segment from the downloaded video using."""
        print("Hello")
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

    def extract_video_segments(self, video_url: str, output_dir: str) -> List[Tuple[float, float, float]]:
        """Extract most replayed segments from YouTube video."""
        video_id = self.extract_video_id(video_url)
        print("Hello")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
            response_text = response.text
            
            data_match = re.search(r'ytInitialData\s*=\s*({.+?});', response_text)
            if not data_match:
                st.error("Could not find YouTube initial data")
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
                    
                    # Filter top 10% most replayed segments
                    if segments:
                        threshold = sorted([s[2] for s in segments], reverse=True)[len(segments)//10]
                        most_replayed = [s for s in segments if s[2] >= threshold]
                        
                        return most_replayed
            
            st.error("Could not find heatmap data in response")
            return []
        
        except Exception as e:
            st.error(f"Error extracting segments: {str(e)}")
            return []

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    st.title("YouTube Video Toolkit 🎥")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    output_dir = st.sidebar.text_input("Output Directory", value=os.path.expanduser("~/Downloads"))
    preferred_langs = st.sidebar.multiselect(
        "Preferred Subtitle Languages", 
        options=['en', 'hi', 'es', 'fr', 'de', 'ru', 'ar', 'zh', 'ja', 'pt'],
        default=['en', 'hi']
    )
    
    # Main input for YouTube URL
    video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    # Create toolkit instance
    toolkit = YouTubeToolkit()
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Full Video Download", "Subtitle Download", "Segment Extraction"])
    
    with tab1:
        st.header("Download Full Video")
        if st.button("Download Full Video", key="full_video_download"):
            if video_url:
                with st.spinner("Downloading video..."):
                    video_path = toolkit.download_full_video(video_url, output_dir)
                    if video_path:
                        st.success(f"Video downloaded successfully: {video_path}")
            else:
                st.warning("Please enter a valid YouTube URL")
    
    with tab2:
        st.header("Download Subtitles")
        if st.button("Download Subtitles", key="subtitle_download"):
            if video_url:
                with st.spinner("Downloading subtitles..."):
                    subtitle_result = toolkit.download_subtitles(video_url, output_dir, preferred_langs)
                    if subtitle_result:
                        st.success(f"Subtitles downloaded: {subtitle_result['path']}")
                        st.text_area("Subtitle Preview", subtitle_result['text'], height=200)
            else:
                st.warning("Please enter a valid YouTube URL")
    
    with tab3:
        st.header("Extract Most Replayed Segments")
        if st.button("Extract Segments", key="segment_extraction"):
            if video_url:
                with st.spinner("Extracting most replayed segments..."):
                    segments = toolkit.extract_video_segments(video_url, output_dir)
                    if segments:
                        st.success(f"Found {len(segments)} most replayed segments")
                        segment_data = []
                        for i, (start, end, intensity) in enumerate(segments, 1):
                            minutes_start = int(start) // 60
                            seconds_start = int(start) % 60
                            minutes_end = int(end) // 60
                            seconds_end = int(end) % 60
                            
                            segment_data.append({
                                "Segment": i,
                                "Start Time": f"{minutes_start}:{seconds_start:02d}",
                                "End Time": f"{minutes_end}:{seconds_end:02d}",
                                "Replay Intensity": f"{intensity:.2f}"
                            })
                        
                        # Initialize session state for button clicks
                        if 'clicked_segments' not in st.session_state:
                            st.session_state.clicked_segments = {}

                        # Debug print to check if the segment data is being processed
                        print("Processing segment data...")

                        for segment in segment_data:
                            st.write(f"**Segment {segment['Segment']}**: {segment['Start Time']} to {segment['End Time']}")
                            
                            # Unique key for each button
                            button_key = f"download_segment_{segment['Segment']}"
                            
                            # Debug print to check if the button is rendered
                            print(f"Rendering button for Segment {segment['Segment']} with key: {button_key}")
                            
                            if st.button(f"Download Segment {segment['Segment']}", key=button_key):
                                st.session_state.clicked_segments[segment['Segment']] = True  # Mark this segment as clicked
                                print("Button clicked for segment:", segment['Segment'])  # Debug print
                                segment_path = toolkit.extract_segment(video_url, start, end, output_dir)
                                
                                if segment_path:
                                    st.success(f"Segment {segment['Segment']} downloaded successfully: {segment_path}")
                                else:
                                    st.error(f"Failed to download segment {segment['Segment']}")
                            else:
                                if segment['Segment'] in st.session_state.clicked_segments:
                                    print(f"Button already clicked for segment: {segment['Segment']}")  # Debug print for already clicked
                                else:
                                    print("Button not clicked for segment:", segment['Segment'])  # Debug print for button state
                    else:
                        st.warning("No segments found or extraction failed")
            else:
                st.warning("Please enter a valid YouTube URL")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 YouTube Video Toolkit | Powered by yt-dlp & Whisper")

if __name__ == "__main__":
    main()