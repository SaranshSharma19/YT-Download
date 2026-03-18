import os
import time
import json
import requests
import subprocess
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from openai import OpenAI
import logging
from pathlib import Path
from enhanced_subtitles import enhance_video_generator

@enhance_video_generator
class AdvancedVideoGenerator:
    def __init__(self, 
                 openai_api_key: str, 
                 together_api_key: str, 
                 output_dir: str = "output"):
        """Initialize with updated video configuration for vertical format."""
        # Clients setup
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.flux_client = OpenAI(
            api_key=together_api_key, 
            base_url="https://api.together.xyz/v1"
        )
        
        # Directory and logging setup
        self.output_dir = Path(output_dir).resolve()
        self.setup_directories()
        self.setup_logging()
        
        # Video configuration
        self.video_config = {
            'width': 1080,
            'height': 1920,
            'target_min_duration' 
            'target_max_duration'
            'min_scene_duration': 6,
            'audio_speed': 1.0,
            'narration_volume': 1.0,
            'background_music_volume': 0.15,
            'crossfade_duration': 0.3
        }
        
        # Metadata setup
        self.metadata_file = self.output_dir / "metadata.csv"
        self.metadata = self._load_metadata()
        
        self._verify_dependencies()
    
    def _verify_dependencies(self):
        """Verify all required system dependencies."""
        dependencies = ['ffmpeg', 'ffprobe', 'auto_subtitle']
        for dep in dependencies:
            try:
                subprocess.run([dep, '-version'], 
                               capture_output=True, 
                               check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                if dep == 'auto_subtitle':
                    self.logger.warning("auto_subtitle not found. Please install it for subtitle generation.")
                else:
                    raise RuntimeError(f"{dep} is not installed or accessible")
    
    def setup_directories(self):
        """Create necessary directories for output files."""
        dirs = ['scripts', 'audio', 'images', 'temp', 'videos', 'logs', 'subtitled']
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            filename=self.output_dir / 'logs/video_generator.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_metadata(self) -> pd.DataFrame:
        """Load or create metadata tracking DataFrame."""
        if self.metadata_file.exists():
            return pd.read_csv(self.metadata_file)
        return pd.DataFrame(columns=[
            'video_id', 'title', 'status', 'script_path',
            'audio_path', 'image_paths', 'video_path',
            'subtitled_video_path', 'created_at', 'updated_at', 'error'
        ])

    def _save_metadata(self):
        """Save metadata to CSV."""
        self.metadata.to_csv(self.metadata_file, index=False)

    def _update_metadata(self, video_id: str, data: Dict):
        """Update metadata for a specific video."""
        if video_id not in self.metadata['video_id'].values:
            new_row = pd.DataFrame([{
                'video_id': video_id,
                'title': data.get('title', ''),
                'status': data.get('status', 'pending'),
                'script_path': data.get('script_path', ''),
                'audio_path': data.get('audio_path', ''),
                'image_paths': str(data.get('image_paths', [])),
                'video_path': data.get('video_path', ''),
                'subtitled_video_path': data.get('subtitled_video_path', ''),
                'created_at': data.get('created_at', datetime.now().isoformat()),
                'updated_at': datetime.now().isoformat(),
                'error': data.get('error', '')
            }])
            self.metadata = pd.concat([self.metadata, new_row], ignore_index=True)
        else:
            idx = self.metadata[self.metadata['video_id'] == video_id].index[0]
            for key, value in data.items():
                self.metadata.at[idx, key] = value
            self.metadata.at[idx, 'updated_at'] = datetime.now().isoformat()
        
        self._save_metadata()

    def _generate_prompt(self, content_type: str, sub_category: Optional[str] = None) -> Tuple[str, str]:
        """Generate a dynamic prompt based on content type and sub-category."""
        duration = self.video_config['target_max_duration']
        if content_type == "Story":
            if sub_category:
                title = f"A {sub_category} Story"
                prompt = f"Generate a captivating {sub_category} story for a faceless video short/reel that hooks viewers from the first sentence. The story must be engaging and fit within {duration} seconds when read at normal speed."
            else:
                title = "An Engaging Story"
                prompt = f"Generate a captivating story for a faceless video short/reel that hooks viewers from the first sentence. The story must be engaging and fit within {duration} seconds when read at normal speed."
        elif content_type == "Facts":
            if sub_category:
                title = f"Interesting {sub_category} Facts"
                prompt = f"Generate interesting and surprising facts about {sub_category} for a faceless video short/reel. The facts should be engaging and fit within {duration} seconds when read at normal speed."
            else:
                title = "Interesting Facts"
                prompt = f"Generate interesting and surprising facts for a faceless video short/reel. The facts should be engaging and fit within {duration} seconds when read at normal speed."
        elif content_type == "Tutorials":
            title = "A Practical Tutorial"
            prompt = f"Generate a clear and concise tutorial script on a practical topic (e.g., fixing a faucet) for a faceless video short/reel. The tutorial should be easy to follow, engaging, and fit within {duration} seconds."
        elif content_type == "Reviews":
            title = "A Insightful Review"
            prompt = f"Generate an honest and engaging review script of a popular product (e.g., a smartphone) for a faceless video short/reel. The review should be balanced and fit within {duration} seconds."
        elif content_type == "Vlogs":
            title = "A Relatable Vlog"
            prompt = f"Generate an engaging vlog-style script about a relatable experience (e.g., exploring a local spot) for a faceless video short/reel. The vlog should be personal and fit within {duration} seconds."
        elif content_type == "Educational Content":
            title = "Key Learning Insights"
            prompt = f"Generate an informative and engaging educational script on a concise topic (e.g., quantum physics basics) for a faceless video short/reel. The content should be clear and fit within {duration} seconds."
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        return title, prompt
    
    def _generate_script(self, prompt: str, content_type: str) -> str:
        """Generate a script with dynamic system prompt based on content type."""
        if content_type == "Story":
            system_prompt = f"""
Objective: Create an engaging story script for short-form video content.
Guidelines:
- Hook in the first 3 seconds with a gripping moment or question.
- Create 5-6 distinct, visually rich scenes, separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use vivid, immersive language.
- Emphasize storytelling with emotions and action.
- Include seamless transitions.
- End with a strong CTA.
Provide only the text for a voice agent to read.
"""
        elif content_type == "Facts":
            system_prompt = f"""
Objective: Create an engaging facts script for short-form video content.
Guidelines:
- Start with an attention-grabbing fact or question.
- Present 5-6 distinct facts, each separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use clear, concise language.
- Make each fact surprising or thought-provoking.
- End with a CTA encouraging engagement.
Provide only the text for a voice agent to read.
"""
        elif content_type == "Tutorials":
            system_prompt = f"""
Objective: Create a clear tutorial script for short-form video content.
Guidelines:
- Start with a brief introduction of the topic.
- Present 3-5 clear steps, each separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use simple, direct language.
- Ensure steps are easy to follow.
- End with a CTA encouraging practice.
Provide only the text for a voice agent to read.
"""
        elif content_type == "Reviews":
            system_prompt = f"""
Objective: Create an engaging review script for short-form video content.
Guidelines:
- Start with an intriguing opinion or question.
- Structure into 3-5 parts (e.g., pros, cons), separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use balanced, engaging language.
- End with a CTA for comments or shares.
Provide only the text for a voice agent to read.
"""
        elif content_type == "Vlogs":
            system_prompt = f"""
Objective: Create a relatable vlog script for short-form video content.
Guidelines:
- Start with a personal hook or anecdote.
- Structure into 3-5 moments, separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use conversational, engaging language.
- End with a CTA for follows or likes.
Provide only the text for a voice agent to read.
"""
        elif content_type == "Educational Content":
            system_prompt = f"""
Objective: Create an informative educational script for short-form video content.
Guidelines:
- Start with a curious fact or question.
- Present 3-5 key points, separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use clear, concise language.
- End with a CTA for further learning.
Provide only the text for a voice agent to read.
"""
        else:
            system_prompt = f"""
Objective: Create an engaging script for {content_type.lower()} content.
Guidelines:
- Start with an attention-grabbing introduction.
- Structure into 3-5 parts, separated by two newlines (\n\n).
- Total duration: {self.video_config['target_max_duration']} seconds.
- Use clear, engaging language.
- End with a CTA.
Provide only the text for a voice agent to read.
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )
        script = response.choices[0].message.content
        if "\n\n" not in script:
            script = script.replace(". ", ".\n\n", 1)  # Ensure at least one delimiter
        return script
    
    def _generate_voice(self, text: str) -> str:
        """Generate voice narration with speed control."""
        response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        original_audio_path = str(self.output_dir / f"audio/original_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
        with open(original_audio_path, 'wb') as f:
            f.write(response.content)
        
        speed_audio_path = str(self.output_dir / f"audio/speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
        subprocess.run([
            'ffmpeg', '-y', 
            '-i', original_audio_path, 
            '-filter:a', f'atempo={self.video_config["audio_speed"]}',
            speed_audio_path
        ], check=True)
        return speed_audio_path

    def _generate_vertical_image(self, scene: str) -> str:
        """Generate vertical images with correct aspect ratio."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = str(self.output_dir / f"images/scene_{timestamp}.png")
        
        prompt = f"""
Create a vertical format illustration (9:16 aspect ratio) with:
- Composition optimized for 1080x1920 resolution
- Main subject centered vertically
- Scene description: {scene}
- Ensure important elements are within the vertical safe zone
- Images parameters: rule of thirds, golden ratio, photorealism, cinematic realism, 8k
"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.flux_client.images.generate(
                    prompt=prompt,
                    model="black-forest-labs/FLUX.1-schnell-Free",
                    n=1,
                    size="1080x1920",
                    timeout=30
                )
                img_response = requests.get(response.data[0].url, timeout=15)
                img_response.raise_for_status()
                with open(image_path, 'wb') as f:
                    f.write(img_response.content)
                return image_path
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Image generation failed: {str(e)}")
                    raise
                time.sleep(retry_delay * (attempt + 1))

    def _mix_audio(self, narration_path: str, background_music_path: str) -> str:
        """Enhanced audio mixing with proper volume control."""
        output_path = str(self.output_dir / f"audio/final_mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', narration_path,
            '-i', background_music_path,
            '-filter_complex',
            f'[0:a]volume={self.video_config["narration_volume"]}[narration];[1:a]volume={self.video_config["background_music_volume"]}[music];[narration][music]amix=inputs=2:duration=first',
            '-c:a', 'libmp3lame',
            '-q:a', '2',
            output_path
        ], check=True)
        return output_path

    def _assemble_video(self, image_paths: List[str], audio_path: str) -> str:
        """Assemble video without built-in subtitles."""
        output_path = str(self.output_dir / f"videos/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        total_duration = self._get_audio_duration(audio_path)
        num_scenes = len(image_paths)
        scene_duration = total_duration / num_scenes
        
        list_file = self.output_dir / "temp/input.txt"
        with open(list_file, 'w') as f:
            for img in image_paths:
                f.write(f"file '{img}'\n")
                f.write(f"duration {scene_duration}\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-shortest',
            '-aspect', '9:16',
            output_path
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def _add_subtitles(self, video_path: str) -> str:
        """Add subtitles using auto_subtitle tool (enhanced by decorator)."""
        try:
            output_path = str(self.output_dir / 'subtitled' / f"subtitled_{os.path.basename(video_path)}")
            subprocess.run([
                'auto_subtitle',
                video_path,
                '-o', output_path
            ], check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Subtitle generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate subtitles: {str(e)}")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using FFprobe."""
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def create_video(self, content_type: str, sub_category:str, min_duration: int, max_duration: int) -> str:
        """Main video creation workflow with dynamic inputs."""
        try:
             # Adjust video configuration
            self.video_config['target_min_duration'] = min_duration
            self.video_config['target_max_duration'] = max_duration
            # Generate prompt
            title, prompt = self._generate_prompt(content_type, sub_category)
            video_id = f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

           
            
            # Generate script
            script = self._generate_script(prompt, content_type)
            
            # Generate voice
            narration_path = self._generate_voice(script)
            
            # Split script into scenes
            scenes = [s.strip() for s in script.split('\n\n') if s.strip()]
            
            # Generate images
            image_paths = [self._generate_vertical_image(scene) for scene in scenes]
            
            # Mix audio
            background_music_path = str(self.output_dir.parent / "bg.mp3")
            final_audio_path = self._mix_audio(narration_path, background_music_path)
            
            # Assemble video
            video_path = self._assemble_video(image_paths, final_audio_path)
            
            # Add subtitles
            subtitled_video_path = self._add_subtitles(video_path)
            
            # Update metadata
            self._update_metadata(video_id, {
                'title': title,
                'status': 'completed',
                'script_path': str(self.output_dir / 'scripts' / f'script_{video_id}.txt'),
                'audio_path': narration_path,
                'image_paths': str(image_paths),
                'video_path': video_path,
                'subtitled_video_path': subtitled_video_path,
                'created_at': datetime.now().isoformat()
            })
            return video_id
        except Exception as e:
            self.logger.error(f"Video creation failed: {str(e)}")
            self._update_metadata(video_id, {'status': 'failed', 'error': str(e)})
            raise

    def get_video_status(self, video_id: str) -> Dict:
        """Get the status and details of a video by ID."""
        video_data = self.metadata[self.metadata['video_id'] == video_id]
        if video_data.empty:
            return {'error': 'Video not found'}
        return video_data.iloc[0].to_dict()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    generator = AdvancedVideoGenerator(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        together_api_key=os.getenv('TOGETHER_API_KEY'),
        output_dir='output'
    )
    try:
        video_id = generator.create_video(content_type="Story", sub_category="Adventure")
        print(f"Created video with ID: {video_id}")
        video_details = generator.get_video_status(video_id)
        print(f"Subtitled video path: {video_details['subtitled_video_path']}")
    except Exception as e:
        print(f"Error creating video: {str(e)}")