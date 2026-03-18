import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pydub import AudioSegment
import random
from pedalboard import Pedalboard, Chorus, Reverb, LowpassFilter, Compressor, Gain, HighpassFilter
import tempfile
import pyrubberband as pyrb
import yt_dlp
import requests
from PIL import Image, ImageFilter
import moviepy.editor as mpy
import argparse
import re

class SlowedReverbGenerator:
    def __init__(self, input_path=None):
        """Initialize the Slowed + Reverb Generator."""
        self.input_path = input_path
        self.output_path = None
        self.sample_rate = 44100
        self.audio_data = None
        self.processed_audio = None
        
    def load_audio(self, input_path=None):
        """Load audio file."""
        if input_path:
            self.input_path = input_path
        
        if self.input_path is None:
            raise ValueError("No input audio path provided!")
            
        print(f"Loading audio from {self.input_path}...")
        try:
            self.audio_data, self.sample_rate = librosa.load(self.input_path, sr=44100)  # Force 44.1kHz
            
            # Normalize audio to prevent issues
            if np.max(np.abs(self.audio_data)) > 0:
                self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
                
            self.processed_audio = self.audio_data.copy()
            return self
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
    
    def slow_down(self, slow_factor=0.75):
        """Slow down the audio using rubberband."""
        print(f"Slowing down audio to {slow_factor:.0%} of original speed...")
        
        try:
            # Use librosa's time stretch for better quality
            self.processed_audio = librosa.effects.time_stretch(self.processed_audio, rate=slow_factor)
            return self
        except Exception as e:
            print(f"Error in slow_down: {e}")
            return self
    
    def lower_pitch(self, semitones=-2):
        """Lower the pitch of the audio."""
        print(f"Lowering pitch by {abs(semitones)} semitones...")
        
        # Use librosa's pitch shift
        self.processed_audio = librosa.effects.pitch_shift(
            self.processed_audio, 
            sr=self.sample_rate, 
            n_steps=semitones
        )
        return self
    
    def apply_pitch_wobble(self, depth=0.001, rate=0.2):
        """Apply subtle pitch wobbling for tape effect."""
        print("Applying subtle pitch wobble effect...")
        y = self.processed_audio.copy()
        
        # Create a slow LFO for the pitch wobble
        time = np.arange(len(y)) / self.sample_rate
        wobble = depth * np.sin(2 * np.pi * rate * time)
        
        # Create time-varying sample indices
        indices = np.arange(len(y)) + (wobble * self.sample_rate)
        
        # Ensure indices are within bounds
        indices = np.clip(indices, 0, len(y) - 1)
        
        # Apply the wobble effect - mix with original to keep it subtle
        wobbly_audio = np.interp(np.arange(len(y)), indices, y)
        self.processed_audio = 0.8 * y + 0.2 * wobbly_audio
        return self
    
    def apply_eq(self, low_boost=4, high_cut=4000):
        """Apply EQ adjustments for slowed + reverb style."""
        print("Applying slowed + reverb EQ adjustments...")
        y = self.processed_audio.copy()
        
        # High-cut filter (low-pass) to reduce high frequencies
        b, a = signal.butter(2, high_cut / (self.sample_rate / 2), btype='low')
        filtered_highs = signal.filtfilt(b, a, y)
        
        # Apply high cut filtering (80% filtered, 20% original highs)
        high_cut_mix = 0.8
        y = (1 - high_cut_mix) * y + high_cut_mix * filtered_highs
        
        # Low boost (increase bass)
        if low_boost > 0:
            low_shelf_freq = 200  # Hz
            b, a = signal.butter(1, low_shelf_freq / (self.sample_rate / 2), btype='low')
            filtered_lows = signal.filtfilt(b, a, y)
            
            # Create shelf mask
            low_mask = np.ones_like(y)
            low_cutoff = 200 / (self.sample_rate / 2)
            b, a = signal.butter(1, low_cutoff, btype='low')
            low_mask = signal.filtfilt(b, a, low_mask)
            
            # Apply the shelf
            shelf_gain = 10 ** (low_boost / 20)  # Convert dB to linear gain
            shelf_mix = 0.6  # Apply 60% of the boost
            y = y * (1 - shelf_mix * low_mask) + filtered_lows * shelf_mix * low_mask * shelf_gain
        
        self.processed_audio = y
        return self
    
    def apply_reverb_effects(self, reverb_level=0.8, chorus_level=0.1):
        """Apply heavy reverb and chorus effects."""
        print(f"Applying heavy reverb ({reverb_level:.0%}) and subtle chorus...")
        y = self.processed_audio.copy()
        
        # Create a chain of effects for slowed + reverb style
        board = Pedalboard([
            Chorus(rate_hz=0.7, depth=0.6, mix=chorus_level),  # Increased chorus depth
            LowpassFilter(cutoff_frequency_hz=4000),  # Less aggressive lowpass
            HighpassFilter(cutoff_frequency_hz=20),   # Lower highpass for more bass
            Compressor(threshold_db=-15, ratio=2.5, attack_ms=10, release_ms=200),  # Modified compression
            Reverb(
                room_size=0.9,     # Larger room size
                damping=0.4,       # Less damping for longer reverb tail
                wet_level=reverb_level,
                dry_level=max(0.2, 1.0 - reverb_level),  # Keep some dry signal
                width=1.0          # Full stereo width
            ),
            Gain(gain_db=-1)  # Prevent clipping
        ])
        
        # Process the audio with the pedalboard
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, y, self.sample_rate)
            
            # Load audio for processing
            audio = AudioSegment.from_wav(temp_path)
            samples = np.array(audio.get_array_of_samples())
            
            # Convert to float and normalize
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
                if np.max(np.abs(samples)) > 0:
                    samples = samples / np.max(np.abs(samples))
            
            # Apply the effects
            processed = board(samples, self.sample_rate)
            
            # Mix with original for better balance
            self.processed_audio = 0.2 * y + 0.8 * processed
            
            # Clean up
            os.remove(temp_path)
            
        return self
    
    def add_subtle_drums(self, bpm=70, pattern_style='minimal', drums_volume=0.05):
        """Add very subtle lo-fi drum beats."""
        if drums_volume <= 0:
            print("Skipping drum beats...")
            return self
            
        print(f"Adding subtle {pattern_style} lo-fi drum beats at {bpm} BPM...")
        y = self.processed_audio.copy()
        
        # Create drum patterns - more minimal patterns
        patterns = {
            'minimal': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Just occasional kicks
            'chill': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],    # Very sparse pattern
            'classic': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Basic boom-bap but will be quiet
        }
        
        # Select pattern or default to minimal
        pattern = patterns.get(pattern_style, patterns['minimal'])
        
        # Create a synthetic kick drum
        seconds_per_beat = 60 / bpm
        seconds_per_step = seconds_per_beat / 4  # 16th notes
        
        # Calculate how many patterns to generate for the entire audio length
        audio_len_seconds = len(y) / self.sample_rate
        pattern_len_seconds = len(pattern) * seconds_per_step
        num_patterns = int(np.ceil(audio_len_seconds / pattern_len_seconds))
        
        # Initialize drum track
        drum_track = np.zeros(len(y))
        
        # Generate kick drum samples - mellower kick
        kick_duration = int(0.2 * self.sample_rate)  # 200ms kick
        kick = np.zeros(kick_duration)
        
        # Synthesize kick drum - softer attack
        t = np.linspace(0, 1, kick_duration)
        decay = np.exp(-4 * t)  # Slower decay
        sine = np.sin(2 * np.pi * 55 * t)  # 55 Hz base frequency
        sine_decay = 0.6 * sine * decay
        
        # Add a soft pitch drop
        freq = np.linspace(120, 55, kick_duration)
        t_array = np.arange(kick_duration) / self.sample_rate
        pitch_drop = np.sin(2 * np.pi * freq * t_array) * np.exp(-10 * t_array)
        
        kick = sine_decay + 0.15 * pitch_drop
        
        # Generate snare drum samples - mellower snare
        snare_duration = int(0.15 * self.sample_rate)  # 150ms snare
        snare = np.zeros(snare_duration)
        
        # Synthesize snare: mix of sine and noise - less attack
        t = np.linspace(0, 1, snare_duration)
        decay = np.exp(-10 * t)  # Slower decay for softer attack
        noise = np.random.normal(0, 0.7, snare_duration) * decay  # Quieter noise
        sine = np.sin(2 * np.pi * 150 * t) * decay * 0.4
        snare = 0.6 * noise + 0.2 * sine
        
        # Generate hi-hat samples - much quieter hats
        hat_duration = int(0.1 * self.sample_rate)  # 100ms hi-hat
        hat = np.zeros(hat_duration)
        
        # Synthesize hi-hat: filtered noise
        t = np.linspace(0, 1, hat_duration)
        decay = np.exp(-20 * t)  # Faster decay for shorter hats
        noise = np.random.normal(0, 0.5, hat_duration) * decay  # Quieter noise
        # Apply high-pass filter
        b, a = signal.butter(3, 5000 / (self.sample_rate / 2), btype='high')
        hat = signal.filtfilt(b, a, noise) * 0.25  # Much reduced volume
        
        # Place the drum hits according to the pattern
        for pattern_num in range(num_patterns):
            for i, hit in enumerate(pattern):
                if hit == 1:
                    # Add a kick
                    position = int(pattern_num * pattern_len_seconds * self.sample_rate + 
                                i * seconds_per_step * self.sample_rate)
                    
                    # Make sure we don't write past the end of the array
                    if position < len(drum_track) - len(kick):
                        drum_track[position:position + len(kick)] += kick
                
                # Add very occasional snare on beats 5 and 13
                if i in [4, 12] and pattern_style != 'minimal':
                    position = int(pattern_num * pattern_len_seconds * self.sample_rate + 
                                i * seconds_per_step * self.sample_rate)
                    
                    if position < len(drum_track) - len(snare):
                        drum_track[position:position + len(snare)] += snare * 0.7  # Even quieter snares
                
                # Add rare hi-hat if using classic pattern
                if i % 4 == 0 and pattern_style == 'classic':
                    position = int(pattern_num * pattern_len_seconds * self.sample_rate + 
                                i * seconds_per_step * self.sample_rate)
                    
                    if position < len(drum_track) - len(hat):
                        drum_track[position:position + len(hat)] += hat * 0.6  # Very quiet hats
        
        # Apply reverb to the drum track to make it blend
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Write the drum track to the temp file
        if np.max(np.abs(drum_track)) > 0:
            drum_track = drum_track / np.max(np.abs(drum_track))
        sf.write(temp_path, drum_track, self.sample_rate)
        
        # Apply reverb to drums
        board = Pedalboard([
            LowpassFilter(cutoff_frequency_hz=3000),
            Reverb(room_size=0.7, wet_level=0.4, dry_level=0.6),
        ])
        
        # Process the drums with reverb
        audio = AudioSegment.from_wav(temp_path)
        samples = np.array(audio.get_array_of_samples())
        
        # Convert to float and normalize
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
            if max(abs(samples)) > 0:
                samples = samples / max(abs(samples))
        
        # Apply the effects
        processed_drums = board(samples, self.sample_rate)
        
        # Mix the drum track with the processed audio
        if np.max(np.abs(processed_drums)) > 0:
            processed_drums = processed_drums / np.max(np.abs(processed_drums)) * drums_volume
            
        # Mix with higher ratio of original - drums very subtle
        self.processed_audio = y * 0.95 + processed_drums * 0.95  # Subtle drums
        
        # Normalize final result to prevent clipping
        if np.max(np.abs(self.processed_audio)) > 0.98:
            self.processed_audio = self.processed_audio / np.max(np.abs(self.processed_audio)) * 0.98
        
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return self
    
    def apply_8d_audio(self, rotation_speed=0.05):
        """Apply 8D audio effect by panning the audio."""
        print("Applying 8D audio effect...")
        y = self.processed_audio.copy()
        num_samples = len(y)
        time = np.arange(num_samples) / self.sample_rate

        # Create a panning effect using a sine wave
        pan = np.sin(2 * np.pi * rotation_speed * time)
        left = y * (1 - pan) * 0.5
        right = y * (1 + pan) * 0.5

        # Combine left and right channels into stereo
        self.processed_audio = np.column_stack((left, right))
        return self

    def save(self, output_path):
        """Save the processed audio."""
        self.output_path = output_path

        if self.processed_audio is None:
            raise ValueError("No processed audio to save. Run effects first.")

        print(f"Saving slowed + reverb + 8D audio to {output_path}...")
        if self.processed_audio.ndim == 2:  # Stereo audio
            sf.write(output_path, self.processed_audio, self.sample_rate, format='WAV', subtype='PCM_16')
        else:  # Mono audio
            sf.write(output_path, self.processed_audio, self.sample_rate)
        return self


class YouTubeSlowedReverbCreator:
    def __init__(self, video_url, output_dir="output"):
        self.video_url = video_url
        self.video_id = self.extract_video_id(video_url)
        self.output_dir = output_dir
        self.temp_dir = os.path.join(output_dir, "temp")
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # File paths
        self.temp_audio_path = os.path.join(self.temp_dir, f"{self.video_id}_audio.m4a")
        self.thumbnail_path = os.path.join(self.temp_dir, f"{self.video_id}_thumbnail.jpg")
        self.output_audio_path = os.path.join(self.output_dir, f"{self.video_id}_slowed_reverb.wav")
        self.output_video_path = os.path.join(self.output_dir, f"{self.video_id}_slowed_reverb.mp4")
        
        # Download status
        self.downloaded_audio = False
        self.downloaded_thumbnail = False
        
    def extract_video_id(self, url):
        """Extract the YouTube video ID from the URL."""
        # Handle different YouTube URL formats
        video_id_match = re.search(r'(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|&v=)([^#\&\?\n<>\'\"]+)', url)
        if video_id_match:
            return video_id_match.group(1)
        else:
            raise ValueError("Could not extract video ID from URL")
    
    def download_audio_only(self):
        """Download only the audio from the YouTube video."""
        if not os.path.exists(self.temp_audio_path) and not self.downloaded_audio:
            try:
                print(f"Downloading audio from: {self.video_url}")
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio',
                    'outtmpl': self.temp_audio_path,
                    'quiet': False,
                    'no_warnings': False,
                    'progress': True,
                    'retries': 5,
                    'socket_timeout': 30,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'm4a',
                    }],
                    'postprocessor_args': ['-metadata', f'comment=Downloaded from {self.video_url}'],
                }

                def progress_hook(d):
                    if d['status'] == 'downloading':
                        try:
                            print(f"\rDownloading audio... {d['_percent_str']} at {d['_speed_str']}", end='')
                        except:
                            print(f"\rDownloading audio... please wait", end='')
                    elif d['status'] == 'finished':
                        print("\nAudio download completed!")

                ydl_opts['progress_hooks'] = [progress_hook]

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print("Starting audio download...")
                    ydl.download([self.video_url])
                self.downloaded_audio = True
                return True
            except Exception as e:
                print(f"An error occurred during audio download: {str(e)}")
                if os.path.exists(self.temp_audio_path):
                    os.remove(self.temp_audio_path)  # Clean up partial download
                return False
        return True
    
    def download_thumbnail(self):
        """Download the thumbnail from the YouTube video."""
        if not os.path.exists(self.thumbnail_path) and not self.downloaded_thumbnail:
            try:
                # YouTube thumbnail URL pattern (try high quality first)
                thumbnail_url = f"https://img.youtube.com/vi/{self.video_id}/maxresdefault.jpg"
                
                print(f"Downloading thumbnail: {thumbnail_url}")
                response = requests.get(thumbnail_url)
                
                # If the high-quality thumbnail isn't available, try standard quality
                if response.status_code != 200:
                    thumbnail_url = f"https://img.youtube.com/vi/{self.video_id}/hqdefault.jpg"
                    print(f"High quality thumbnail not found, trying: {thumbnail_url}")
                    response = requests.get(thumbnail_url)
                
                if response.status_code == 200:
                    with open(self.thumbnail_path, 'wb') as f:
                        f.write(response.content)
                    print("Thumbnail downloaded successfully!")
                    self.downloaded_thumbnail = True
                    return True
                else:
                    print(f"Failed to download thumbnail, status code: {response.status_code}")
                    return False
            except Exception as e:
                print(f"An error occurred during thumbnail download: {str(e)}")
                return False
        return True
    
    def process_audio(self, slow_factor=0.75, pitch_change=-2, add_drums=True, 
                     bpm=70, drum_pattern='minimal', drum_volume=0.05, reverb_level=0.8, enable_8d=True, rotation_speed=0.05):
        """Process the audio with slowed, reverb, and optional 8D effects."""
        try:
            processor = SlowedReverbGenerator(self.temp_audio_path)
            
            # Modified effects chain order
            processor.load_audio().apply_eq().slow_down(slow_factor=slow_factor).lower_pitch(semitones=pitch_change).apply_pitch_wobble(depth=0.002, rate=0.1).apply_reverb_effects(reverb_level=reverb_level)
            
            # Add drums if requested
            if add_drums:
                processor.add_subtle_drums(
                    bpm=int(bpm * slow_factor),  # Adjust BPM to match slowed tempo
                    pattern_style=drum_pattern,
                    drums_volume=drum_volume
                )
            
            # Apply 8D audio effect if enabled
            if enable_8d:
                processor.apply_8d_audio(rotation_speed=rotation_speed)

            # Save the processed audio
            processor.save(self.output_audio_path)
            print(f"Slowed + reverb + 8D audio created: {self.output_audio_path}")
            return True
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return False
    
    def enhance_thumbnail(self):
        """Apply lo-fi aesthetic effects to the thumbnail."""
        try:
            print("Applying lo-fi effects to thumbnail...")
            # Open the image
            img = Image.open(self.thumbnail_path)
            
            # Slightly desaturate
            
            # Save the enhanced thumbnail
            enhanced_thumbnail_path = os.path.join(self.temp_dir, f"{self.video_id}_enhanced_thumbnail.jpg")
            img.save(enhanced_thumbnail_path)
            
            return enhanced_thumbnail_path
        except Exception as e:
            print(f"Error enhancing thumbnail: {str(e)}")
            return self.thumbnail_path  # Return original if enhancement fails
    
    def create_video(self):
        """Create a video with the processed audio and enhanced thumbnail using ffmpeg."""
        try:
            print("Creating final video with thumbnail and slowed+reverb audio using ffmpeg...")
            
            # Enhance the thumbnail
            enhanced_thumbnail = self.enhance_thumbnail()
            
            # Create a temporary video from the thumbnail
            temp_video_path = os.path.join(self.temp_dir, f"{self.video_id}_temp_video.mp4")
            ffmpeg_cmd_thumbnail = [
                'ffmpeg',
                '-loop', '1',  # Loop the image
                '-i', enhanced_thumbnail,  # Input image
                '-c:v', 'libx264',  # Use H.264 codec
                '-t', str(librosa.get_duration(filename=self.output_audio_path)),  # Set duration to match audio
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-vf', 'scale=1280:720',  # Scale to 720p
                temp_video_path,
                '-y'  # Overwrite if exists
            ]
            subprocess.run(ffmpeg_cmd_thumbnail, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Combine the video and processed audio
            ffmpeg_cmd = [
                'ffmpeg', 
                '-i', temp_video_path,  # Video input
                '-i', self.output_audio_path,  # Lo-fi audio input
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # AAC audio codec
                '-map', '0:v:0',  # Use the first video stream from first input
                '-map', '1:a:0',  # Use the first audio stream from second input
                '-shortest',  # End when the shortest input ends
                self.output_video_path,  # Output file
                '-y'  # Overwrite if exists
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Slowed + reverb video created successfully: {self.output_video_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating video with ffmpeg: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during video creation: {e}")
            return False
    
    def cleanup(self):
        """Remove temporary files."""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                print("Temporary files cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def process(self, slow_factor=0.75, pitch_change=-2, add_drums=True, 
               bpm=70, drum_pattern='minimal', drum_volume=0.05, reverb_level=0.8,
               cleanup_temp=True, enable_8d=True, rotation_speed=0.05):
        """Run the complete process to create a slowed+reverb video from YouTube."""
        success = True
        
        # Step 1: Download the audio
        if not self.download_audio_only():
            print("Failed to download audio, aborting process.")
            return False
        
        # Step 2: Download the thumbnail
        if not self.download_thumbnail():
            print("Failed to download thumbnail, aborting process.")
            return False
        
        # Step 3: Process the audio
        if not self.process_audio(
            slow_factor=slow_factor,
            pitch_change=pitch_change,
            add_drums=add_drums,
            bpm=bpm,
            drum_pattern=drum_pattern,
            drum_volume=drum_volume,
            reverb_level=reverb_level,
            enable_8d=enable_8d,
            rotation_speed=rotation_speed
        ):
            print("Failed to process audio, aborting process.")
            return False
        
        # Step 4: Create the final video
        if not self.create_video():
            print("Failed to create final video.")
            success = False
        
        # Step 5: Clean up temporary files if requested
        if cleanup_temp:
            self.cleanup()
        
        if success:
            print(f"YouTube slowed + reverb video created successfully: {self.output_video_path}")
        
        return success


def create_yt_slowed_reverb(video_url, output_dir="output", slow_factor=0.75, pitch_change=-2,
                          add_drums=True, bpm=70, drum_pattern='minimal', drum_volume=0.05,
                          reverb_level=0.8, cleanup_temp=True, enable_8d=True, rotation_speed=0.05):
    """
    Create a slowed + reverb version of a YouTube video.
    
    Parameters:
    -----------
    video_url : str
        URL of the YouTube video
    output_dir : str
        Directory to save output files
    slow_factor : float
        Speed factor (0.85 = 85% of original speed, i.e., 15% slower)
    pitch_change : int
        Semitones to pitch shift (negative = lower pitch)
    add_drums : bool
        Whether to add subtle lo-fi drum patterns
    bpm : int
        Beats per minute for the drum patterns
    drum_pattern : str
        Type of drum pattern: 'minimal', 'chill', or 'classic'
    drum_volume : float
        Volume of the drums (0.0 to 1.0)
    reverb_level : float
        Level of reverb (0.0 to 1.0)
    cleanup_temp : bool
        Whether to clean up temporary files after processing
    enable_8d : bool
        Whether to enable 8D audio effect
    rotation_speed : float
        Speed of 8D audio rotation
    """
    creator = YouTubeSlowedReverbCreator(video_url, output_dir)
    return creator.process(
        slow_factor=slow_factor,
        pitch_change=pitch_change,
        add_drums=add_drums,
        bpm=bpm,
        drum_pattern=drum_pattern,
        drum_volume=drum_volume,
        reverb_level=reverb_level,
        cleanup_temp=cleanup_temp,
        enable_8d=enable_8d,
        rotation_speed=rotation_speed
    )

# Example usage
if __name__ == "__main__":
    print("DISCLAIMER: This script is for personal use only. Ensure you have permission to download and modify the content.")
    parser = argparse.ArgumentParser(description='Create slowed + reverb version of a YouTube video.')
    parser.add_argument('video_url', help='YouTube video URL')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--slow', type=float, default=0.75, 
                    help='Slow factor (0.85 = 85% speed, i.e., 15% slower)')
    parser.add_argument('--pitch', type=int, default=-2, 
                    help='Pitch adjustment in semitones (negative = lower)')
    parser.add_argument('--no-drums', action='store_false', dest='add_drums', 
                    help='Skip adding drum patterns')
    parser.add_argument('--bpm', type=int, default=70, help='Beats per minute for drums')
    parser.add_argument('--drum-pattern', choices=['minimal', 'chill', 'classic'], 
                    default='minimal', help='Type of drum pattern')
    parser.add_argument('--drum-volume', type=float, default=0.05, 
                    help='Volume of drums (0.0 to 1.0)')
    parser.add_argument('--reverb', type=float, default=0.8, 
                    help='Level of reverb (0.0 to 1.0)')
    parser.add_argument('--keep-temp', action='store_true', 
                    help='Keep temporary files after processing')
    parser.add_argument('--enable-8d', action='store_true', help='Enable 8D audio effect')
    parser.add_argument('--rotation-speed', type=float, default=0.05, help='8D audio rotation speed (default: 0.05)')

    args = parser.parse_args()

    create_yt_slowed_reverb(
        args.video_url,
        output_dir=args.output_dir,
        slow_factor=args.slow,
        pitch_change=args.pitch,
        add_drums=args.add_drums,
        bpm=args.bpm,
        drum_pattern=args.drum_pattern,
        drum_volume=args.drum_volume,
        reverb_level=args.reverb,
        cleanup_temp=not args.keep_temp,
        enable_8d=args.enable_8d,
        rotation_speed=args.rotation_speed
    )