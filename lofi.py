import sys
import os
import argparse
from pydub import AudioSegment
from pydub.effects import low_pass_filter
import numpy as np
import yt_dlp
import subprocess
import tempfile
import shutil
import math
from pydub.generators import Sine

class YoutubeLofiConverter:
    def __init__(self, video_url, output_path, temp_dir=None):
        """Initialize the converter with video URL and output path."""
        self.video_url = video_url
        self.output_path = output_path
        self.video_id = self.get_video_id(video_url)
        
        # Create temporary directory if not provided
        if temp_dir:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = tempfile.mkdtemp()
            
        # Set paths for temp files
        self.temp_video_path = os.path.join(self.temp_dir, f"{self.video_id}_original.mp4")
        self.temp_audio_path = os.path.join(self.temp_dir, f"{self.video_id}_audio.mp3")
        self.temp_lofi_audio_path = os.path.join(self.temp_dir, f"{self.video_id}_lofi_audio.mp3")
        
        self.downloaded = False

    @staticmethod
    def get_video_id(url):
        """Extract video ID from YouTube URL."""
        if 'youtu.be' in url:
            return url.split('/')[-1]
        elif 'youtube.com' in url:
            if 'v=' in url:
                return url.split('v=')[1].split('&')[0]
        return "video"  # Default if ID can't be extracted

    def download_full_video(self):
        """Download the full video only once."""
        if not os.path.exists(self.temp_video_path) and not self.downloaded:
            try:
                print(f"Downloading video from: {self.video_url}")
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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
                return True
            except Exception as e:
                print(f"\nAn error occurred during download: {str(e)}")
                if os.path.exists(self.temp_video_path):
                    os.remove(self.temp_video_path)  # Clean up partial download
                return False
        return True

    def extract_audio(self):
        """Extract audio from the downloaded video."""
        try:
            print("Extracting audio from video...")
            ffmpeg_cmd = [
                'ffmpeg', '-i', self.temp_video_path,
                '-q:a', '0', '-map', 'a', self.temp_audio_path, '-y'
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Audio extraction complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during audio extraction: {e}")
            return False

    def add_reverb(self, sound, gain_db=-10, delay=100, decay=2):
        """Add reverb effect to the audio"""
        # Create the reverb effect by adding delayed copies with decreasing volume
        reverb = sound.fade_out(int(len(sound) * 0.05))
        
        # Number of reverb echoes
        num_echoes = 5
        
        # Create a silent segment for the reverb tail
        reverb_tail = AudioSegment.silent(duration=delay * num_echoes)
        
        # Build reverb by adding delayed echoes with decreasing volume
        result = sound + reverb_tail
        
        for i in range(1, num_echoes + 1):
            # Calculate volume reduction for this echo
            volume_decrease = gain_db * i * decay
            
            # Create echo with appropriate delay and reduced volume
            echo = sound - abs(volume_decrease)
            
            # Position echo at the right point in time
            result = result.overlay(echo, position=delay * i)
        
        return result

    def slow_down(self, sound, factor=0.85):
        """Slow down the audio by the given factor without changing the pitch"""
        # Save original frame rate
        original_frame_rate = sound.frame_rate
        
        # Apply the slowdown by changing the frame rate
        slowed = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * factor)
        })
        
        # Convert back to the original frame rate, which changes the speed but not pitch
        return slowed.set_frame_rate(original_frame_rate)

    def enhance_bass(self, sound, bass_boost_db=2):
        """Enhance the bass frequencies slightly"""
        # Apply a low-pass filter to get the bass frequencies
        bass = low_pass_filter(sound, cutoff=200)
        
        # Increase the volume of the bass
        boosted_bass = bass + bass_boost_db
        
        # Overlay the boosted bass with the original audio
        return sound.overlay(boosted_bass)
    
    def apply_8d_audio(self, sound, rotation_speed=0.05, volume_reduction=6):
        """
        Apply 8D audio effect that makes sound appear to move around the listener's head
        Optimized version that processes audio more efficiently
        
        Parameters:
        - rotation_speed: controls how fast the sound rotates (lower = slower)
        - volume_reduction: how much to reduce volume at furthest point (in dB)
        
        Returns:
        - Audio with 8D effect applied
        """
        print("Applying 8D audio effect...")
        
        # Get audio data as numpy arrays for faster processing
        samples = np.array(sound.get_array_of_samples())
        
        # Check if stereo or mono
        is_stereo = sound.channels == 2
        
        if is_stereo:
            # Reshape for stereo processing
            samples = samples.reshape((-1, 2))
            left_samples = samples[:, 0]
            right_samples = samples[:, 1]
        else:
            # If mono, duplicate to create stereo
            left_samples = samples
            right_samples = samples.copy()
            samples = np.column_stack((left_samples, right_samples))
        
        # Duration in samples
        num_samples = len(samples)
        
        # Create arrays for adjusted samples
        left_adjusted = left_samples.astype(np.float32)
        right_adjusted = right_samples.astype(np.float32)
        
        # Sample rate and time array
        sample_rate = sound.frame_rate
        time_array = np.arange(num_samples) / sample_rate
        
        # Calculate pan position for each sample (sine wave pattern)
        pan_positions = np.sin(2 * np.pi * rotation_speed * time_array)
        
        # Calculate volume adjustments (-volume_reduction to 0)
        left_adjustments = np.where(pan_positions > 0, -volume_reduction * pan_positions, 0)
        right_adjustments = np.where(pan_positions < 0, volume_reduction * pan_positions, 0)
        
        # Convert dB adjustments to amplitude multipliers
        left_multipliers = np.power(10, left_adjustments / 20)
        right_multipliers = np.power(10, right_adjustments / 20)
        
        # Apply the multipliers to the samples
        left_adjusted = left_adjusted * left_multipliers
        right_adjusted = right_adjusted * right_multipliers
        
        # Combine back to stereo
        adjusted_samples = np.column_stack((left_adjusted, right_adjusted))
        
        # Convert back to the format pydub expects
        sample_width = sound.sample_width
        adjusted_samples = adjusted_samples.astype(
            np.int16 if sample_width == 2 else np.int32 if sample_width == 4 else np.int8
        )
        
        # Create a new AudioSegment with the adjusted samples
        output = sound._spawn(adjusted_samples.tobytes())
        
        # Apply slight reverb to enhance the 3D effect
        output = self.add_reverb(output, gain_db=-15, delay=40, decay=1.5)
        
        print("8D audio effect applied!")
        return output

    def apply_lofi_effects(self, slowdown_factor=0.85, reverb_gain=-15, reverb_delay=80, 
                          enable_8d=False, rotation_speed=0.05):
        """Apply lo-fi effects to the extracted audio"""
        try:
            print("Applying lo-fi slowed reverb effects...")
            sound = AudioSegment.from_file(self.temp_audio_path)
            
            # Step 1: Apply subtle low-pass filter (cuts high frequencies, but not too much)
            sound = low_pass_filter(sound, 4000)
            
            # Step 2: Slow down the track
            sound = self.slow_down(sound, factor=slowdown_factor)
            
            # Step 3: Add reverb
            sound = self.add_reverb(sound, gain_db=reverb_gain, delay=reverb_delay)
            
            # Step 4: Enhance bass slightly for warmth
            sound = self.enhance_bass(sound)
            
            # Step 5: Apply 8D audio effect if enabled
            if enable_8d:
                sound = self.apply_8d_audio(sound, rotation_speed=rotation_speed)
                        
            # Step 7: Subtle compression for that lo-fi feel
            sound = sound.compress_dynamic_range(threshold=-20, ratio=4.0)
            
            # Step 8: Final volume adjustment
            sound = sound.normalize()
            
            # Export the lo-fi version
            sound.export(self.temp_lofi_audio_path, format="mp3")
            print("Lo-fi effects applied successfully!")
            return True
        except Exception as e:
            print(f"Error applying lo-fi effects: {e}")
            return False

    def combine_video_audio(self):
        """Combine the original video with the modified audio"""
        try:
            print("Combining video with lo-fi audio...")
            ffmpeg_cmd = [
                'ffmpeg', 
                '-i', self.temp_video_path,  # Video input
                '-i', self.temp_lofi_audio_path,  # Lo-fi audio input
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # AAC audio codec
                '-map', '0:v:0',  # Use the first video stream from first input
                '-map', '1:a:0',  # Use the first audio stream from second input
                '-shortest',  # End when the shortest input ends
                self.output_path,  # Output file
                '-y'  # Overwrite if exists
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Success! Lo-fi video saved to: {self.output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error combining video and audio: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during combining: {e}")
            return False

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            print("Temporary files cleaned up.")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

    def process(self, slowdown_factor=0.85, reverb_gain=-15, reverb_delay=80, 
                enable_8d=False, rotation_speed=0.05, cleanup=True):
        """Process the entire conversion workflow"""
        try:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            
            # Step 1: Download the video from YouTube
            if not self.download_full_video():
                return False
                
            # Step 2: Extract audio from the video
            if not self.extract_audio():
                return False
                
            # Step 3: Apply lo-fi effects to the audio
            if not self.apply_lofi_effects(
                slowdown_factor, 
                reverb_gain, 
                reverb_delay, 
                enable_8d, 
                rotation_speed,
                bpm
            ):
                return False
                
            # Step 4: Combine the original video with the modified audio
            if not self.combine_video_audio():
                return False
                
            # Clean up temporary files if requested
            if cleanup:
                self.cleanup()
                
            return True
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert YouTube videos to lo-fi slowed reverb style with 8D audio')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--slowdown', type=float, default=0.85, help='Slowdown factor (default: 0.85)')
    parser.add_argument('--reverb-gain', type=float, default=-15, help='Reverb gain in dB (default: -15)')
    parser.add_argument('--reverb-delay', type=int, default=80, help='Reverb delay in ms (default: 80)')
    parser.add_argument('--enable-8d', action='store_true', help='Enable 8D audio effect')
    parser.add_argument('--rotation-speed', type=float, default=0.05, help='8D audio rotation speed (default: 0.05)')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not clean up temporary files')
    
    args = parser.parse_args()
    
    converter = YoutubeLofiConverter(args.url, args.output)
    success = converter.process(
        slowdown_factor=args.slowdown, 
        reverb_gain=args.reverb_gain, 
        reverb_delay=args.reverb_delay,
        enable_8d=args.enable_8d,
        rotation_speed=args.rotation_speed,
        cleanup=not args.no_cleanup
    )
    
    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed. Check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())