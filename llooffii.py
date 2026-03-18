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
import time
from pathlib import Path


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

        print(f"Saving slowed + reverb audio to {output_path}...")
        if self.processed_audio.ndim == 2:  # Stereo audio
            sf.write(output_path, self.processed_audio, self.sample_rate, format='WAV', subtype='PCM_16')
        else:  # Mono audio
            sf.write(output_path, self.processed_audio, self.sample_rate)
        return self


class EnhancedYouTubeSlowedReverbCreator:
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
        self.temp_video_path = os.path.join(self.temp_dir, f"{self.video_id}_video.mp4")
        self.output_audio_path = os.path.join(self.output_dir, f"{self.video_id}_slowed_reverb.wav")
        self.output_video_path = os.path.join(self.output_dir, f"{self.video_id}_slowed_reverb.mp4")
        
        # Download status
        self.downloaded_audio = False
        self.downloaded_video = False
        
    def extract_video_id(self, url):
        """Extract the YouTube video ID from the URL."""
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
    
    def download_full_video(self):
        """Download the full video from YouTube."""
        if not os.path.exists(self.temp_video_path) and not self.downloaded_video:
            try:
                print(f"Downloading full video from: {self.video_url}")
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best[height>=1080]',
                    'outtmpl': self.temp_video_path,
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
                            print(f"\rDownloading video... {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}", end='')
                        except:
                            print(f"\rDownloading video... please wait", end='')
                    elif d['status'] == 'finished':
                        print("\nVideo download completed!")

                ydl_opts['progress_hooks'] = [progress_hook]

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print("Starting video download...")
                    ydl.download([self.video_url])
                self.downloaded_video = True
                return True
            except Exception as e:
                print(f"An error occurred during video download: {str(e)}")
                if os.path.exists(self.temp_video_path):
                    os.remove(self.temp_video_path)  # Clean up partial download
                return False
        return True
    
    # def apply_copyright_protection(self, video_path):
    #     """Apply various transformations including unique visual filters to help avoid copyright detection."""
    #     print("Applying copyright protection transformations with artistic filters...")
    #     output_path = video_path.replace('.mp4', '_protected.mp4')
        
    #     # Complex filter chain with artistic effects
    #     filter_chain = [
    #         # Ensure proper scaling and alignment first
    #         "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            
    #         # Basic adjustments with aligned data
    #         "eq=brightness=0.05:saturation=0.8:contrast=1.1",
            
    #         # Add film grain effect
    #         "noise=alls=20:allf=t",
            
    #         # Color manipulation
    #         "colorbalance=rs=0.1:gs=0.1:bs=0.1:rm=0.1:gm=0.1:bm=0.1:rh=0.1:gh=0.1:bh=0.1",
            
    #         # Subtle rotation with aligned padding
    #         f"pad=iw+4:ih+4:2:2:black,rotate=angle=1*PI/180:fillcolor=black",
            
    #         # Add vignette effect
    #         "vignette=angle=PI/4:mode=backward",
            
    #         # Artistic effects with proper formatting
    #         f"format=yuv420p,hue=h={random.uniform(-15, 15)}:s=0.8",
            
    #         # Add subtle glitch-like effects with proper alignment
    #         "split[a][b];"
    #         "[a]lutrgb=r=negval:g=negval:b=negval,format=yuv420p[a1];"
    #         "[b]hue=h=0.1,format=yuv420p[b1];"
    #         "[a1][b1]blend=all_mode=overlay:all_opacity=0.1",
            
    #         # Add watermark with specified font
    #         'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    #         ':text=aesthetic:fontsize=24:x=w-tw-10:y=h-th-10'
    #         ':fontcolor=white@0.3:box=1:boxcolor=black@0.2'
    #     ]
        
    #     # Combine all filters
    #     video_filter = ",".join(filter_chain)
        
    #     # Enhanced audio filter with better frequency response
    #     audio_filter = (
    #         "aformat=channel_layouts=stereo,"  # Ensure stereo output
    #         "volume=0.95,"
    #         "highpass=f=200:width_type=o:width=0.7,"  # Smoother highpass
    #         "lowpass=f=3000:width_type=o:width=0.7"   # Smoother lowpass
    #     )
        
    #     # Enhanced FFmpeg command with better encoding settings
    #     command = [
    #         'ffmpeg', '-y',
    #         '-i', video_path,
    #         '-vf', video_filter,
    #         '-af', audio_filter,
    #         '-c:v', 'libx264',
    #         '-preset', 'ultrafast',      # Better balance of speed/quality
    #         '-profile:v', 'main',     # More compatible profile
    #         '-level', '4.0',
    #         '-crf', '23',            # Good quality
    #         '-movflags', '+faststart',
    #         '-pix_fmt', 'yuv420p',   # Ensure compatibility
    #         '-tune', 'film',         # Better for filtered content
    #         '-threads', '0',         # Use optimal thread count
    #         '-bufsize', '2M',
    #         '-maxrate', '2M',
    #         output_path
    #     ]
        
    #     try:
    #         # Run FFmpeg with error output redirected to file
    #         with open('ffmpeg_error.log', 'w') as error_log:
    #             subprocess.run(command, check=True, stderr=error_log)
    #         print("Applied artistic filters and copyright protection.")
    #         return output_path
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error applying filters: {e}")
    #         # Log error details
    #         if os.path.exists('ffmpeg_error.log'):
    #             with open('ffmpeg_error.log', 'r') as error_log:
    #                 print(f"FFmpeg error details:\n{error_log.read()}")
    #         return video_path
    #     finally:
    #         # Clean up error log
    #         if os.path.exists('ffmpeg_error.log'):
    #             os.remove('ffmpeg_error.log')
    
    # def apply_copyright_protection(self, video_path):
    #     """Apply minimal but unique transformations to help avoid copyright detection while maintaining video quality and speed."""
    #     print("Applying advanced copyright protection transformations...")
    #     output_path = video_path.replace('.mp4', '_protected.mp4')

    #     # Enhanced filter chain for uniqueness and speed
    #     filter_chain = [
    #         # Ensure proper scaling
    #         "scale=trunc(iw/2)*2:trunc(ih/2)*2",

    #         # Subtle but effective adjustments
    #         "eq=brightness=0.02:saturation=0.97:contrast=1.03",

    #         # Small hue shift
    #         f"hue=h={random.uniform(-3, 3)}",
            
    #         # Add a black & white flicker for a few frames every second
    #         "split[a][b];"
    #         "[a]format=gray,framestep=30,format=yuv420p[gray];"
    #         "[b][gray]overlay=enable='mod(n,30)':shortest=1",

    #         # Subtle vignette for aesthetic and fingerprint change
    #         "vignette=PI/8",
            
    #         # Format ensuring compatibility
    #         "format=yuv420p"
    #     ]

    #     # Combine all filters
    #     video_filter = ",".join(filter_chain)

    #     # Slight audio pitch shift and volume change for uniqueness
    #     audio_filter = "asetrate=44100*1.003,aresample=44100,volume=0.99"

    #     # Optimized FFmpeg command for speed and reasonable quality
    #     command = [
    #         'ffmpeg', '-y',
    #         '-i', video_path,
    #         '-vf', video_filter,
    #         '-af', audio_filter,
    #         '-c:v', 'libx264',
    #         '-preset', 'veryfast',
    #         '-profile:v', 'main',
    #         '-level', '4.0',
    #         '-crf', '22',
    #         '-movflags', '+faststart',
    #         '-pix_fmt', 'yuv420p',
    #         '-threads', '0',
    #         output_path
    #     ]

    #     try:
    #         # Run FFmpeg with error output redirected to file
    #         with open('ffmpeg_error.log', 'w') as error_log:
    #             subprocess.run(command, check=True, stderr=error_log)
    #         print("Applied advanced copyright protection. Processing complete.")
    #         return output_path
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error applying filters: {e}")
    #         if os.path.exists('ffmpeg_error.log'):
    #             with open('ffmpeg_error.log', 'r') as error_log:
    #                 print(f"FFmpeg error log first few lines:\n{error_log.read()[:500]}")
    #         return video_path
    #     finally:
    #         if os.path.exists('ffmpeg_error.log'):
    #             os.remove('ffmpeg_error.log')
    
    def resize_video_with_padding(self, input_path: str) -> str:
        """
        Resize video for specific platform using, maintaining aspect ratio and adding padding.
        """
        target_size = (1980, 1020)
        output_path = input_path.replace('.mp4', '_padded.mp4')

        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f"scale='if(gt(iw/ih,{target_size[0]}/{target_size[1]}),{target_size[0]},-1)':'if(gt(iw/ih,{target_size[0]}/{target_size[1]}),-1,{target_size[1]})',pad={target_size[0]}:{target_size[1]}:(ow-iw)/2:(oh-ih)/2",
            '-preset', 'ultrafast',
            '-c:a', 'aac', output_path
        ]
        subprocess.run(command, check=True)
        return output_path
    
    def apply_copyright_protection(self, video_path):
        """Apply copyright protection using FFmpeg for better performance."""
        print("Applying advanced copyright protection with FFmpeg...")
        output_path = video_path.replace('.mp4', '_protected.mp4')
        
        # Define watermarks with simpler positioning
        watermark1 = (
            "drawtext=font='DejaVu Sans Bold':"
            "text='Play Lofi Music':fontsize=40:"
            "fontcolor=white@0.7:box=1:boxcolor=black@0.5:"
            f"x=(w-text_w)/2:y=h-text_h-20"  # Centered bottom
        )

        watermark2 = (
            "drawtext=font='DejaVu Sans Bold':"
            "text='slowed + reverb':fontsize=30:"
            "fontcolor=white@0.5:box=1:boxcolor=black@0.3:"
            "x=w-text_w-20:y=20"  # Top right
        )
        
        watermark3 = (
            "drawtext=font='DejaVu Sans Bold':"
            "text='♪ music ♫':fontsize=60:"  # Larger font with music notes
            "fontcolor=white@0.2:"  # More transparent (0.2 opacity)
            "box=1:boxcolor=black@0.1:"  # Very subtle box
            "x=(w-text_w)/2:y=(h-text_h)/2"  # Centered position
        )
        
        zoom_filter = (
            "zoompan=z='min(max(zoom,1.0),1.5)+sin(on/200*PI)/20':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "d=1:s=1920x1080:fps=30"  # Fixed size specification
        )
        
        # Simplified filter chain
        filter_chain = [
            # Basic adjustments
            "scale=1920:1080",  # Force consistent resolution
            
            # Add continuous zoom
            zoom_filter,
            
            # Basic adjustments
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
            "eq=brightness=0.02:saturation=1.1:contrast=1.05",
            
            # Add margins
            "pad=iw+40:ih+40:20:20:color=black",
            
            # Add a black & white flicker for a few frames every second
            # "split[a][b];"
            # "[a]format=gray,framestep=30,format=yuv420p[gray];"
            # "[b][gray]overlay=enable='mod(n,30)':shortest=1",

            # Subtle vignette for aesthetic and fingerprint change
            "vignette=PI/8",
            
            # Add watermarks
            watermark1,
            watermark2,
            watermark3,
            
            # Final format
            "format=yuv420p"
        ]

        # Simple audio adjustment
        audio_filter = (
            "volume=0.99,"
            "aresample=44100"
        )

        # FFmpeg command with reliable settings
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', ','.join(filter_chain),
            '-af', audio_filter,
            '-c:v', 'libx264',
            '-preset', 'fast',       # Faster encoding
            '-profile:v', 'main',    # Better compatibility
            '-crf', '23',           # Good balance of quality/size
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-max_muxing_queue_size', '1024',  # Prevent muxing errors
            output_path
        ]

        try:
            # Run FFmpeg with better error handling
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Capture all output
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print("\nSuccessfully applied copyright protection.")
                return output_path
            else:
                print(f"\nFFmpeg Error: {stderr}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return video_path
                
        except Exception as e:
            print(f"\nError in copyright protection: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return video_path

    def process_audio(self, slow_factor=0.75, pitch_change=-2, add_drums=True, 
                     bpm=70, drum_pattern='minimal', drum_volume=0.05, reverb_level=0.8, enable_8d=True, rotation_speed=0.05):
        """Process the audio with slowed, reverb, and optional 8D effects."""
        try:
            # Extract audio from video if we have the video file
            if os.path.exists(self.temp_video_path) and not os.path.exists(self.temp_audio_path):
                if not self.extract_audio():
                    return False

            processor = SlowedReverbGenerator(self.temp_audio_path)
            
            # Modified effects chain order
            processor.load_audio().apply_eq().slow_down(slow_factor=slow_factor).lower_pitch(semitones=pitch_change).apply_reverb_effects(reverb_level=reverb_level)
            
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

    def slow_down_video(self, video_path, slow_factor):
        """Slow down the video to match the audio length."""
        print("Slowing down video...")
        output_path = video_path.replace('.mp4', '_slowed.mp4')
        
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-filter:v', f"setpts={slow_factor}*PTS", 
            '-preset', 'ultrafast', # Slow down video
            output_path
        ]
        subprocess.run(command, check=True)
        
        return output_path

    def get_audio_duration(self, audio_path):
        """Get the duration of the audio file."""
        audio = mpy.AudioFileClip(audio_path)
        duration = audio.duration
        audio.close()
        return duration

    def get_video_duration(self, video_path):
        """Get the duration of the video file."""
        video = mpy.VideoFileClip(video_path)
        duration = video.duration
        video.close()
        return duration

    def process_full_video(self):
        """Process the full video with the slowed + reverb effect."""
        # Ensure the video is downloaded and audio is processed
        if not os.path.exists(self.temp_video_path):
            print("Full video not found. Please download it first.")
            return False

        if not os.path.exists(self.output_audio_path):
            print("Processed audio not found. Please run process_audio first.")
            return False

        try:
            print("Processing full video...")
            
            # Get durations
            audio_duration = self.get_audio_duration(self.output_audio_path)
            video_duration = self.get_video_duration(self.temp_video_path)
            
            # Calculate the slow factor
            slow_factor = audio_duration / video_duration
            
            # Slow down the video
            slowed_video_path = self.slow_down_video(self.temp_video_path, slow_factor)
            
            # Combine video and audio processing logic here...
            combine_cmd = [
                'ffmpeg',
                '-i', slowed_video_path,
                '-i', self.output_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-preset', 'ultrafast',  # Use ultrafast preset for speed
                self.output_video_path,
                '-y'
            ]
            
            print("Combining video with processed audio...")
            subprocess.run(combine_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Check if the combined video file was created successfully
            if not os.path.exists(self.output_video_path):
                print(f"Error: Combined video file '{self.output_video_path}' was not created.")
                return False

            # Apply copyright protection
            protected_video_path = self.apply_copyright_protection(self.resize_video_with_padding(self.output_video_path))
            
            # Replace original output with protected version
            os.replace(protected_video_path, self.output_video_path)

            print(f"Video processing complete! Output saved to: {self.output_video_path}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a YouTube video.")
    parser.add_argument("video_url", type=str, help="URL of the YouTube video")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--slow", type=float, default=0.75, help="Speed factor")
    parser.add_argument("--pitch", type=int, default=-2, help="Semitones to pitch shift")
    parser.add_argument("--bpm", type=int, default=70, help="Beats per minute for the drum patterns")
    parser.add_argument("--no-drum", action='store_true', help="Disable drum patterns")
    parser.add_argument("--reverb", type=float, default=0.8, help="Level of reverb")
    parser.add_argument("--enable-8d", action='store_true', help="Enable 8D audio effect")
    parser.add_argument("--rotation-speed", type=float, default=0.05, help="Speed of 8D audio rotation")

    args = parser.parse_args()

    # Create an instance of the creator
    creator = EnhancedYouTubeSlowedReverbCreator(args.video_url, args.output_dir)

    # Download audio and video
    if not creator.download_audio_only():
        print("Failed to download audio.")
        exit(1)

    if not creator.download_full_video():
        print("Failed to download video.")
        exit(1)

    # Process the audio
    if not creator.process_audio(
        slow_factor=args.slow,
        pitch_change=args.pitch,
        add_drums=not args.no_drum,
        bpm=args.bpm,
        reverb_level=args.reverb,
        enable_8d=args.enable_8d,
        rotation_speed=args.rotation_speed
    ):
        print("Failed to process audio.")
        exit(1)

    # Process the full video
    if not creator.process_full_video():
        print("Failed to create final video.")
        exit(1)

    print("Processing completed successfully!")