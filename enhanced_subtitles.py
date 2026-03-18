import whisper
import numpy as np
from typing import List, Dict, Tuple
import re
import json
import subprocess
import random
from pathlib import Path
import ffmpeg
import os
import tempfile

class EnhancedSubtitleGenerator:
    def __init__(self, model_name: str = "base"):
        """Initialize with Whisper model."""
        self.model = whisper.load_model(model_name)
        
        # Modern, bold fonts that match the examples in the image
        self.fonts = [
            "Impact",        # Classic bold font, highly readable
            "Montserrat-Bold",   # Modern, clean sans-serif
            "Oswald-Bold",       # Bold, attention-grabbing
            "Anton"              # Ultra-bold display font
        ]
        
        # Vibrant colors matching the examples (in ASS BGR hex format)
        self.primary_colors = [
            "&H00FFFFFF",  # White
            "&H000000FF",  # Red
            "&H0000FFFF",  # Yellow
            "&H0013E3FF",  # Vibrant orange (#FFE313)
            "&H0089FC33",  # Lime green (#33FC89)
        ]
        
        # Highlight colors for important words
        self.highlight_colors = [
            "&H000000FF",  # Red
            "&H0000FFFF",  # Yellow
            "&H0013E3FF",  # Orange
            "&H0089FC33",  # Green
        ]
        
        # Words that should be emphasized
        self.emphasis_words = [
            "you", "free", "instantly", "amazing", "shocking", 
            "incredible", "top", "best", "worst", "never", "always",
            "secret", "revealed", "perfect", "ultimate", "easy", "simple",
            "viral", "trending", "exclusive", "breaking", "finally", "ever",
            "insane", "crazy", "unbelievable", "must", "need", "important",
            "success", "succeed", "money", "wealth", "rich", "poor", "fast",
            "quick", "instant", "proven", "guaranteed", "powerful", "essential",
            "top", "truth", "lie", "fake", "real", "authentic", "warning"
        ]
            
    def _should_emphasize_word(self, word: str) -> bool:
        """Determine if a word should be emphasized."""
        word = word.lower().strip(".,!?;:()")
        
        # Direct match with emphasis words
        if word in self.emphasis_words:
            return True
            
        # ALL CAPS words should be emphasized
        if word.isupper() and len(word) > 1:
            return True
            
        # Random emphasis for important-sounding words
        if len(word) > 6 and random.random() < 0.3:
            return True
            
        return False
    
    def _format_ass_timestamp(self, seconds: float) -> str:
        """Format time in ASS format (H:MM:SS.CC)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        cents = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{cents:02d}"
    
    def _clean_text(self, text: str) -> str:
        """Clean up text and remove unnecessary spaces."""
        # Clean and convert to uppercase
        return re.sub(r'\s+', ' ', text).strip().upper()
    
    def _split_segment_for_visual_appeal(self, segment: Dict) -> List[Dict]:
        """Split a segment into visually appealing chunks similar to the example image."""
        text = segment['text'].strip()
        text = self._clean_text(text)
        
        # Split on natural pauses
        phrases = re.split(r'[,;:!?\n]|\.\s', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        if not phrases:
            phrases = [text]
        
        # For viral video style, we want short, impactful segments
        enhanced_segments = []
        
        for phrase in phrases:
            words = phrase.split()
            
            if len(words) <= 5:
                # Keep very short phrases intact
                enhanced_segments.append(phrase)
            else:
                # Split longer phrases into chunks of 4-5 words
                chunks = []
                i = 0
                while i < len(words):
                    # Fixed chunk size to 4-5 words maximum
                    chunk_size = min(5, len(words) - i)
                    chunks.append(' '.join(words[i:i + chunk_size]))
                    i += chunk_size
                
                enhanced_segments.extend(chunks)
        
        # Calculate timing for each segment
        total_duration = segment['end'] - segment['start']
        
        # Use word-length based timing distribution
        total_chars = sum(len(s) for s in enhanced_segments)
        
        time_segments = []
        current_time = segment['start']
        
        for i, text_segment in enumerate(enhanced_segments):
            # Calculate duration based on relative length and a minimum display time
            relative_length = len(text_segment) / max(1, total_chars)
            segment_duration = max(0.8, relative_length * total_duration)
            
            # Ensure we don't exceed total duration
            if i == len(enhanced_segments) - 1:
                # Last segment ends exactly at original end time
                segment_end = segment['end']
            else:
                segment_end = min(segment['end'], current_time + segment_duration)
            
            time_segments.append({
                'text': text_segment,
                'start': current_time,
                'end': segment_end
            })
            
            current_time = segment_end
        
        return time_segments

    def _get_clean_style(self, text: str, segment_index: int, total_segments: int) -> Tuple[int, int, str, int]:
        """Get clean, bold style similar to the examples in the image."""
        # Choose font based on content
        font_index = segment_index % len(self.fonts)
        
        # Fixed large font size for all text
        font_size = 120
        
        # Choose color based on text content and position
        if any(word in text for word in ["SUCCESS", "VIRAL", "AMAZING", "BEST"]):
            color = self.primary_colors[random.randint(1, 3)]  # Red, Yellow, or Orange
        elif segment_index % 3 == 0:
            color = self.primary_colors[3]  # Orange
        elif segment_index % 5 == 0:
            color = self.primary_colors[4]  # Green
        else:
            color = self.primary_colors[0]  # White
        
        # Position below center of screen
        alignment = 8  # Below center alignment (8)
        
        return font_index, font_size, color, alignment
        
    def _highlight_key_words(self, text: str) -> str:
        """Highlight key words with different colors for emphasis."""
        words = text.split()
        processed_words = []
        
        for word in words:
            clean_word = word.strip(".,!?;:()")
            if self._should_emphasize_word(clean_word.lower()):
                # Choose a random highlight color
                highlight_color = random.choice(self.highlight_colors)
                
                # Use the punctuation characters that might have been stripped
                pre_punctuation = ""
                post_punctuation = ""
                
                if word.startswith(tuple(".,!?;:()")):
                    pre_punctuation = word[0]
                    word = word[1:]
                
                if word.endswith(tuple(".,!?;:()")):
                    post_punctuation = word[-1]
                    word = word[:-1]
                
                # Create emphasized word
                emphasized_word = f"{pre_punctuation}{{\\c{highlight_color}}}{word}{{\\c}}{post_punctuation}"
                processed_words.append(emphasized_word)
            else:
                processed_words.append(word)
        
        return " ".join(processed_words)

    def process_video(self, video_path: str, output_path: str = None) -> str:
        """Process video and generate styled ASS subtitles."""
        # Transcribe using Whisper
        result = self.model.transcribe(video_path)
        
        # Split segments into smaller chunks for visual appeal
        enhanced_segments = []
        for segment in result['segments']:
            enhanced_segments.extend(self._split_segment_for_visual_appeal(segment))
        
        # Generate ASS file
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.ass'))
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # ASS Header with video dimensions
            f.write("[Script Info]\n")
            f.write("Title: Bold Modern Subtitles\n")
            f.write("ScriptType: v4.00+\n")
            f.write("PlayResX: 1080\n")  # Width for vertical video
            f.write("PlayResY: 1920\n")  # Height for vertical video
            f.write("WrapStyle: 0\n")
            f.write("ScaledBorderAndShadow: yes\n")
            f.write("\n")
            
            # Styles Section
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            
            # Define styles for each font
            for i, font in enumerate(self.fonts):
                f.write(f"Style: Font{i},{font},120,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,0,8,10,10,200,1\n")
            
            f.write("\n")
            
            # Events Section
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            total_segments = len(enhanced_segments)
            
            for i, segment in enumerate(enhanced_segments):
                start = self._format_ass_timestamp(segment['start'])
                end = self._format_ass_timestamp(segment['end'])
                
                # Apply clean, bold style
                font_index, font_size, text_color, alignment = self._get_clean_style(segment['text'], i, total_segments)
                font_style = f"Font{font_index}"
                
                # Process text for word-level highlighting
                processed_text = self._highlight_key_words(segment['text'])
                
                # Format with style elements but keep it clean and bold
                # Use \fad for proper fade effects
                fade_in_cs = 15  # 0.15 seconds fade in
                fade_out_cs = 15  # 0.15 seconds fade out
                
                formatted_text = f"{{\\fs{font_size}\\c{text_color}\\b1\\bord3\\shad0\\fad({fade_in_cs},{fade_out_cs})}}{processed_text}"
                
                # Write the dialogue line with proper alignment
                f.write(f"Dialogue: 0,{start},{end},{font_style},,10,10,200,,{{\\an{alignment}}}{formatted_text}\n")
        
        return output_path

    def create_subtitled_video(self, video_path: str, output_path: str) -> str:
        """Create video with overlaid ASS subtitles with clean, bold style."""
        # Generate the ASS subtitles
        ass_path = self.process_video(video_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Use FFmpeg to apply ASS subtitles
        try:
            # Build the FFmpeg command
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"ass={ass_path}",
                '-c:v', 'libx264', 
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'copy',
                output_path
            ]
            
            subprocess.run(command, check=True)
            print(f"Successfully created subtitled video: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error applying subtitles: {e}")
            # Fallback to alternative method
            try:
                video = ffmpeg.input(video_path)
                audio = video.audio
                
                subtitled_video = video.filter('subtitles', ass_path)
                
                ffmpeg.output(
                    subtitled_video, 
                    audio, 
                    output_path,
                    vcodec='libx264',
                    acodec='aac'
                ).run(quiet=True, overwrite_output=True)
                
                print(f"Successfully created subtitled video (fallback method 1): {output_path}")
                return output_path
            except Exception as e2:
                print(f"Fallback method failed too: {e2}")
                # Basic fallback approach
                try:
                    command = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-vf', f"subtitles={ass_path}:force_style='FontSize=120,Alignment=8,BorderStyle=4,Outline=3'",
                        '-c:a', 'copy',
                        output_path
                    ]
                    subprocess.run(command, check=True)
                    print(f"Successfully created subtitled video (fallback method 2): {output_path}")
                    return output_path
                except Exception as e3:
                    print(f"All subtitle methods failed: {e3}")
                    return video_path  # Return original if all methods fail

def enhance_video_generator(generator_class):
    """Decorator to enhance VideoGenerator with clean, bold subtitles."""
    original_add_subtitles = generator_class._add_subtitles
    
    def enhanced_add_subtitles(self, video_path: str) -> str:
        subtitle_generator = EnhancedSubtitleGenerator()
        output_dir = getattr(self, 'output_dir', os.path.dirname(video_path))
        subtitled_dir = os.path.join(output_dir, 'subtitled')
        os.makedirs(subtitled_dir, exist_ok=True)
        output_path = os.path.join(subtitled_dir, f"subtitled_{os.path.basename(video_path)}")
        
        try:
            return subtitle_generator.create_subtitled_video(video_path, output_path)
        except Exception as e:
            print(f"Enhanced subtitles failed: {str(e)}")
            # Try original method as fallback
            try:
                return original_add_subtitles(self, video_path)
            except Exception as e2:
                print(f"Original subtitle method failed too: {e2}")
                return video_path
    
    generator_class._add_subtitles = enhanced_add_subtitles
    return generator_class