import argparse
from pydub import AudioSegment
from pydub.generators import Sine

class YoutubeLofiConverter:
    def __init__(self, url, output_path):
        self.url = url
        self.output_path = output_path
        self.temp_audio_path = "temp_audio.mp3"
        self.temp_lofi_audio_path = "temp_lofi_audio.mp3"

    def apply_lofi_effects(self, slowdown_factor=0.85, reverb_gain=-15, reverb_delay=80, 
                          enable_8d=False, rotation_speed=0.05, add_drums=False, bpm=60):
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
            
            # Step 6: Add soft drum effect if enabled
            if add_drums:
                sound = self.add_soft_drums(sound, bpm=bpm)
            
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

def main():
    parser = argparse.ArgumentParser(description='Convert YouTube videos to lo-fi slowed reverb style with 8D audio option')
    parser.add_argument('url', type=str, help='URL of the YouTube video to convert')
    parser.add_argument('output', type=str, help='Output file path for the converted audio')
    parser.add_argument('--slowdown', type=float, default=0.85, help='Factor by which to slow down the audio (default: 0.85)')
    parser.add_argument('--reverb-gain', type=int, default=-15, help='Gain for the reverb effect in dB (default: -15)')
    parser.add_argument('--reverb-delay', type=int, default=80, help='Delay for the reverb effect in ms (default: 80)')
    parser.add_argument('--enable-8d', action='store_true', help='Enable 8D audio effect')
    parser.add_argument('--rotation-speed', type=float, default=0.05, help='Rotation speed for 8D audio effect (default: 0.05)')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not clean up temporary files')
    parser.add_argument('--add-drums', action='store_true', help='Add soft drum effect')
    parser.add_argument('--bpm', type=int, default=60, help='Beats per minute for drum effect (default: 60)')
    
    args = parser.parse_args()
    
    converter = YoutubeLofiConverter(args.url, args.output)
    success = converter.process(
        slowdown_factor=args.slowdown, 
        reverb_gain=args.reverb_gain, 
        reverb_delay=args.reverb_delay,
        enable_8d=args.enable_8d,
        rotation_speed=args.rotation_speed,
        cleanup=not args.no_cleanup,
        add_drums=args.add_drums,
        bpm=args.bpm
    )
    
    if success:
        print(f"Conversion successful! Output saved to {args.output}")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()
