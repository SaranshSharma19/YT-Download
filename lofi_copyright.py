#!/usr/bin/env python3
"""
Create a soothing ‘lo-fi, slowed & reverb, 8-D’ version of any YouTube video.
Author : YOU <you@example.com> – 2024
Licence: MIT
"""

import os
import re
import shutil
import argparse
import subprocess
import tempfile
import requests
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy import signal
import pyrubberband as pyrb
import noisereduce as nr
import pyloudnorm as pyln
import yt_dlp
from PIL import Image, ImageEnhance, ImageFilter

# ──────────────────────────────────────────────────────────────
# PEDALBOARD – robust imports for ANY version
# ──────────────────────────────────────────────────────────────
from pedalboard import Pedalboard, Compressor, Gain
import inspect

# ===== MBCompressor =================================================
try:
    from pedalboard import MBCompressor
    sig = inspect.signature(MBCompressor)
    if not {'low_band', 'mid_band', 'makeup_db'} <= set(sig.parameters):
        raise ImportError("MBCompressor alias is not multiband")
except Exception:
    class MBCompressor(Compressor):                      # type: ignore
        def __init__(self, *args,
                     low_band=None, mid_band=None,
                     makeup_db=0, **kwargs):
            super().__init__(*args, **kwargs)
            self._makeup = makeup_db
        def __call__(self, samples, sr):
            out = super().__call__(samples, sr)
            if self._makeup:
                out *= 10 ** (self._makeup / 20)
            return out
    print("WARNING  » Using single-band Compressor shim for MBCompressor "
          "(upgrade pedalboard for real multiband support).")

# ===== Reverb  (pre_delay_ms compatibility) =========================
from pedalboard import Reverb as _PBReverb

need_wrapper = True
try:
    need_wrapper = 'pre_delay_ms' not in inspect.signature(_PBReverb).parameters
except (ValueError, TypeError):
    need_wrapper = True

if need_wrapper:
    class Reverb(_PBReverb):                               # type: ignore
        def __init__(self, *args, pre_delay_ms=None, **kwargs):
            super().__init__(*args, **kwargs)
    print("WARNING  » Reverb has no pre_delay_ms; argument will be ignored.")
else:
    Reverb = _PBReverb

# ------------------------------------------------------------------ #
#                     ──  GLOBAL CONSTS  ──                          #
# ------------------------------------------------------------------ #
TARGET_SR   = 44_100          # we work at 44.1 kHz
TARGET_LUFS = -14.0           # loudness normalisation target (≈ YouTube)
YDL_OPTS    = {
    'format': 'bestaudio[ext=m4a]/bestaudio',
    'quiet':  False,
    'no_warnings': True,
    'retries':  5,
    'socket_timeout': 30,
    'http_headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
}

# ------------------------------------------------------------------ #
#                  ──  AUDIO  PROCESSOR  ──                          #
# ------------------------------------------------------------------ #
class SlowedReverbGenerator:
    """All DSP lives here; everything kept in float32 until export."""
    FADE_MS        = 5                    # fade-in/out to kill edge clicks
    PEAK_HEADROOM  = 0.891_250_938        # –1 dBFS linear ≈ 10**(-1/20)

    def __init__(self, path: str):
        self.path = path
        self.y, self.sr = librosa.load(path, sr=TARGET_SR, mono=True)
        self.proc = self.y.copy()

    # ---------- utilities ---------------------------------------------------
    def _fade_edges(self, y: np.ndarray) -> np.ndarray:
        """5-ms squared-cosine fade to avoid click at start/end."""
        n = int(self.FADE_MS / 1000 * self.sr)
        if n == 0:
            return y
        fade = np.linspace(0, 1, n, dtype=np.float32) ** 2
        if y.ndim == 1:
            y[:n]  *= fade
            y[-n:] *= fade[::-1]
        else:                              # stereo etc.
            y[:n, :]  *= fade[:, None]
            y[-n:, :] *= fade[::-1][:, None]
        return y

    def _peak_normalise(self, y: np.ndarray) -> np.ndarray:
        """Bring absolute peak ≤ –1 dBFS, preserving LUFS."""
        peak = np.max(np.abs(y))
        if peak > self.PEAK_HEADROOM:
            y = y * (self.PEAK_HEADROOM / peak)
        return y

    def _normalise_lufs(self, y: np.ndarray) -> np.ndarray:
        meter = pyln.Meter(self.sr)
        loud  = meter.integrated_loudness(y)
        gain_db = TARGET_LUFS - loud
        y = y * 10 ** (gain_db / 20)
        return self._peak_normalise(y)     # ← safety

    def _save_wav_24bit(self, y: np.ndarray, out_path: str):
        y = self._fade_edges(y)            # fade first/last samples
        y = self._normalise_lufs(y)
        # simple rectangular dither to 24-bit
        y = np.clip(
            y + np.random.uniform(-1/2**25, 1/2**25, y.shape),
            -1.0, 1.0
        )
        sf.write(out_path, y, self.sr, subtype="PCM_24")

    # ---------- processing chain -------------------------------------------
    def noise_reduce(self, prop_decrease: float = 0.25):
        print("• Noise-reduction")
        noise_prof = self.proc[: int(0.5 * self.sr)]
        self.proc = nr.reduce_noise(
            y=self.proc, sr=self.sr, y_noise=noise_prof,
            stationary=False, prop_decrease=prop_decrease
        )
        return self

    def slow_and_pitch(self, speed: float = 0.80, semitones: int = -2):
        print(f"• Time-stretch {speed*100:.1f}% & pitch {semitones:+} st")
        self.proc = pyrb.time_stretch(self.proc, self.sr, speed)
        if semitones:
            self.proc = pyrb.pitch_shift(self.proc, self.sr, semitones)
        return self

    def wobble(self, depth: float = 0.002, rate: float = 0.15):
        print("• Subtle tape-wobble")
        t = np.arange(len(self.proc)) / self.sr
        idx = np.arange(len(self.proc)) + depth * self.sr * np.sin(2*np.pi*rate*t)
        idx = np.clip(idx, 0, len(self.proc) - 1)
        wob = np.interp(np.arange(len(self.proc)), idx, self.proc)
        self.proc = 0.85 * self.proc + 0.15 * wob
        return self

    def eq(self):
        print("• Musical EQ (warm lows + air cut)")
        y = self.proc

        # High-pass @ 60 Hz
        b, a = signal.butter(2, 60 / (self.sr/2), 'high')
        y = signal.filtfilt(b, a, y)

        # Low-shelf +3 dB under 180 Hz
        b, a = signal.butter(1, 180 / (self.sr/2), 'low')
        low_boost = signal.filtfilt(b, a, y) * 10 ** (3/20)
        y = 0.7 * low_boost + 0.3 * y

        # Gentle +2 dB bell at 3 kHz
        w0, Q, gain = 3000 / (self.sr/2), 1.0, 10 ** (2/20)
        try:
            b, a = signal.iirpeak(w0, Q=Q, gain=gain)       # SciPy ≥ 1.8
            y = signal.filtfilt(b, a, y)
        except TypeError:
            b, a = signal.iirpeak(w0, Q=Q)
            peak = signal.filtfilt(b, a, y)
            y = y + (gain - 1) * peak

        # High-cut @ 8 kHz
        b, a = signal.butter(2, 8000 / (self.sr/2), 'low')
        self.proc = signal.filtfilt(b, a, y)
        return self

    def dynamics_reverb(self, wet: float = 0.55):
        print("• Multiband compression + reverb")
        board = Pedalboard([
            MBCompressor(threshold_db=-20, ratio=2.4, makeup_db=1.5,
                         low_band=120, mid_band=4000),
            Reverb(room_size=0.85, damping=0.35,
                   pre_delay_ms=40, wet_level=wet,
                   dry_level=1-wet, width=1.0),
            Gain(gain_db=-1.0)
        ])
        self.proc = board(self.proc.astype(np.float32), self.sr)
        return self

    def pan_8d(self, speed: float = 0.05):
        print("• 8-D constant-power panning")
        t = np.arange(len(self.proc)) / self.sr
        pan = 0.5 * (1 + np.sin(2*np.pi*speed*t))      # 0‥1
        left  = self.proc * np.cos(np.pi/2 * pan)
        right = self.proc * np.sin(np.pi/2 * pan)
        self.proc = np.column_stack((left, right))
        return self

    # ----------------------------------------------------------------------
    def render(self, out_path: str) -> str:
        print("• Saving audio …")
        self._save_wav_24bit(self.proc, out_path)
        return out_path

# ------------------------------------------------------------------ #
#                       DRUMS  (optional)                            #
# ------------------------------------------------------------------ #
def synth_lofi_drums(length_samples: int,
                     sr: int = TARGET_SR,
                     bpm: int = 70,
                     pattern: str = 'minimal',
                     volume: float = 0.04) -> np.ndarray:
    """Return a very quiet lo-fi kick track (no hats / snares by default)."""
    if volume <= 0:
        return np.zeros(length_samples, dtype=np.float32)

    patterns = {
        'minimal': [1] + [0]*15,
        'chill':   [1] + [0]*7 + [1] + [0]*7,
        'classic': [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
    }
    pat = patterns.get(pattern, patterns['minimal'])
    steps_per_beat = 4
    sec_per_beat   = 60 / bpm
    sec_per_step   = sec_per_beat / steps_per_beat
    pat_len_sec    = len(pat) * sec_per_step
    n_patterns     = int(np.ceil((length_samples/sr) / pat_len_sec))

    # synth kick
    kd = int(0.18 * sr)
    t  = np.linspace(0, 0.18, kd, False)
    kick = 0.5*np.sin(2*np.pi*55*t) * np.exp(-5*t)

    out = np.zeros(length_samples, dtype=np.float32)
    for p in range(n_patterns):
        for i, hit in enumerate(pat):
            if not hit:
                continue
            pos = int((p*pat_len_sec + i*sec_per_step) * sr)
            if pos + kd < length_samples:
                out[pos:pos+kd] += kick

    # band-limit & reverb for softness
    b, a = signal.butter(2, [200/(sr/2), 3000/(sr/2)], 'band')
    out = signal.lfilter(b, a, out)
    out = out / max(1e-9, np.max(np.abs(out))) * volume
    return out.astype(np.float32)

# ------------------------------------------------------------------ #
#             ──  YOUTUBE   LO-FI  CREATOR  ──                       #
# ------------------------------------------------------------------ #
class YouTubeSlowedReverbCreator:
    def __init__(self, url: str, out_dir: str = "output"):
        self.url  = url
        self.vid  = self._extract_vid(url)
        self.out_dir  = Path(out_dir)
        self.tmp_dir  = self.out_dir / "tmp"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

        self.temp_audio = self.tmp_dir / f"{self.vid}.m4a"
        self.thumb_path = self.tmp_dir / f"{self.vid}.jpg"
        self.proc_wav   = self.out_dir / f"{self.vid}_lofi.wav"
        self.proc_mp4   = self.out_dir / f"{self.vid}_lofi.mp4"

    # ---------- helpers ----------------------------------------------------
    @staticmethod
    def _extract_vid(u: str) -> str:
        m = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)', u)
        if not m:
            raise ValueError("Could not parse YouTube video ID.")
        return m.group(1)

    # ---------- YT download ------------------------------------------------
    def dl_audio(self):
        if self.temp_audio.exists():
            print("Audio already downloaded.")
            return
        print("Downloading audio stream …")
        ydl_opts = {**YDL_OPTS, 'outtmpl': str(self.temp_audio)}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

    def dl_thumbnail(self):
        if self.thumb_path.exists():
            print("Thumbnail already downloaded.")
            return
        for quality in ("maxresdefault", "hqdefault"):
            u = f"https://img.youtube.com/vi/{self.vid}/{quality}.jpg"
            r = requests.get(u, timeout=15)
            if r.ok:
                self.thumb_path.write_bytes(r.content)
                print(f"Thumbnail downloaded ({quality}).")
                return
        raise RuntimeError("Couldn't fetch thumbnail.")

    # ---------- thumbnail aesthetics --------------------------------------
    def enhance_thumbnail(self) -> Path:
        img = Image.open(self.thumb_path).convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)

        # lo-fi vibes
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        img = ImageEnhance.Color(img).enhance(0.90)
        img = ImageEnhance.Brightness(img).enhance(1.05)

        out = self.tmp_dir / f"{self.vid}_enh.jpg"
        img.save(out, quality=92)
        return out

    # ---------- audio processing ------------------------------------------
    def process_audio(self,
                      slow: float   = 0.80,
                      pitch: int    = -2,
                      add_drums: bool = False,
                      bpm: int      = 70,
                      drum_pattern: str = 'minimal',
                      drum_vol: float   = 0.04,
                      rotation_speed: float = 0.05):
        print("=== DSP chain ===")
        gen = (SlowedReverbGenerator(self.temp_audio)
               .noise_reduce()
               .slow_and_pitch(slow, pitch)
               .wobble()
               .eq()
               .dynamics_reverb()
               .pan_8d(rotation_speed))

        # optional drums (add before panning so they move as well)
        if add_drums:
            print("• Adding gentle drums")
            drums = synth_lofi_drums(len(gen.proc), TARGET_SR,
                                     int(bpm * slow),
                                     drum_pattern, drum_vol)
            drums = np.column_stack((drums, drums))
            gen.proc = np.clip(gen.proc + drums, -1.0, 1.0)

        gen.render(self.proc_wav)

    # ---------- video mux ---------------------------------------------------
    def mux_video(self):
        thumb = self.enhance_thumbnail()
        duration = librosa.get_duration(path=str(self.proc_wav))
        tmp_vid  = self.tmp_dir / f"{self.vid}_still.mp4"

        # 1. still-image video track
        cmd1 = [
            "ffmpeg", "-loglevel", "error",
            "-loop", "1", "-i", str(thumb),
            "-c:v", "libx264", "-t", f"{duration:.3f}",
            "-pix_fmt", "yuv420p", "-vf", "scale=1280:720",
            "-y", str(tmp_vid)
        ]
        subprocess.run(cmd1, check=True)

        # 2. mux with processed WAV
        cmd2 = [
            "ffmpeg", "-loglevel", "error",
            "-i", str(tmp_vid), "-i", str(self.proc_wav),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            "-y", str(self.proc_mp4)
        ]
        subprocess.run(cmd2, check=True)
        print(f"Finished: {self.proc_mp4}")

    # ---------- cleanup -----------------------------------------------------
    def clean(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ---------- master ------------------------------------------------------
    def run(self, **kwargs):
        self.dl_audio()
        self.dl_thumbnail()
        self.process_audio(**kwargs)
        self.mux_video()
        self.clean()

# ------------------------------------------------------------------ #
#                                   CLI                              #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Create a soothing lo-fi version of any YouTube video."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--out-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("-s", "--slow", type=float, default=0.80,
                        help="Time-stretch speed (default: 0.80)")
    parser.add_argument("-p", "--pitch", type=int, default=-2,
                        help="Pitch shift in semitones (default: -2)")
    parser.add_argument("-d", "--drums", action="store_true",
                        help="Add gentle drums")
    parser.add_argument("-b", "--bpm", type=int, default=70,
                        help="Drum pattern BPM (default: 70)")
    parser.add_argument("-r", "--pattern", default='minimal',
                        choices=['minimal', 'chill', 'classic'],
                        help="Drum pattern (default: minimal)")
    parser.add_argument("-v", "--volume", type=float, default=0.04,
                        help="Drum volume (default: 0.04)")
    parser.add_argument("-R", "--rotation-speed", type=float, default=0.05,
                        help="8-D rotation speed (default: 0.05)")

    args = parser.parse_args()

    creator = YouTubeSlowedReverbCreator(args.url, args.out_dir)
    creator.run(slow=args.slow, pitch=args.pitch,
                add_drums=args.drums, bpm=args.bpm,
                drum_pattern=args.pattern, drum_vol=args.volume,
                rotation_speed=args.rotation_speed)
    print("Done! Enjoy your lo-fi vibes.")

if __name__ == "__main__":
    main()