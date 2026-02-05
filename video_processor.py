"""
Video Processor Module - Audio extraction & frame capture
Handles extracting audio from videos and capturing key frames for visual highlights.
"""

import os
import tempfile
from pathlib import Path

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None

import numpy as np
from PIL import Image


def extract_audio(video_path: str, output_format: str = "mp3") -> str:
    """
    Extract audio from video file using MoviePy.
    
    Args:
        video_path: Path to the video file
        output_format: Audio format (mp3, wav)
        
    Returns:
        Path to the extracted audio file
        
    Raises:
        ImportError: If MoviePy is not installed
        FileNotFoundError: If video file doesn't exist
    """
    if VideoFileClip is None:
        raise ImportError("MoviePy is required for audio extraction. Install with: pip install moviepy")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video_path = os.path.abspath(video_path)
    
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp_file:
        audio_path = tmp_file.name
    
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        clip.close()
        return audio_path
    except Exception as e:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        raise RuntimeError(f"Failed to extract audio: {str(e)}") from e


def capture_frame_at_timestamp(video_path: str, timestamp_seconds: float) -> np.ndarray | None:
    """
    Capture a single frame from video at the given timestamp.
    
    Args:
        video_path: Path to the video file
        timestamp_seconds: Time in seconds to capture frame
        
    Returns:
        Frame as numpy array (RGB) or None if failed
    """
    if VideoFileClip is None:
        return None
    
    if not os.path.exists(video_path):
        return None
    
    try:
        clip = VideoFileClip(video_path)
        # Ensure timestamp is within video duration
        timestamp = min(max(0, timestamp_seconds), clip.duration - 0.01)
        frame = clip.get_frame(timestamp)
        clip.close()
        return frame
    except Exception:
        return None


def capture_key_frames(video_path: str, timestamps: list[float], output_dir: str | None = None) -> list[tuple[float, str]]:
    """
    Capture frames at multiple timestamps for visual highlights.
    
    Args:
        video_path: Path to the video file
        timestamps: List of timestamps in seconds
        output_dir: Optional directory to save frame images
        
    Returns:
        List of (timestamp, image_path) tuples
    """
    results = []
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, ts in enumerate(timestamps):
        frame = capture_frame_at_timestamp(video_path, ts)
        if frame is not None:
            img = Image.fromarray(np.uint8(frame))
            img_path = os.path.join(output_dir, f"frame_{i}_{int(ts)}s.png")
            img.save(img_path, "PNG")
            results.append((ts, img_path))
    
    return results


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of the video in seconds.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds, or 0 if failed
    """
    if VideoFileClip is None or not os.path.exists(video_path):
        return 0.0
    
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        return 0.0


def get_video_info(video_path: str) -> dict:
    """
    Get basic video metadata.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dict with duration, fps, size info
    """
    if VideoFileClip is None or not os.path.exists(video_path):
        return {"duration": 0, "fps": 0, "width": 0, "height": 0}
    
    try:
        clip = VideoFileClip(video_path)
        info = {
            "duration": clip.duration,
            "fps": clip.fps,
            "width": clip.w,
            "height": clip.h,
        }
        clip.close()
        return info
    except Exception:
        return {"duration": 0, "fps": 0, "width": 0, "height": 0}


# Supported video extensions - comprehensive list
SUPPORTED_VIDEO_FORMATS = {
    ".mp4", ".avi", ".mkv", ".mov", ".webm",
    ".wmv", ".flv", ".m4v", ".mpg", ".mpeg",
    ".3gp", ".3g2", ".ogv", ".mts", ".m2ts",
    ".vob", ".ts", ".divx", ".f4v", ".asf",
}


def is_supported_video(filename: str) -> bool:
    """Check if the file has a supported video format."""
    return Path(filename).suffix.lower() in SUPPORTED_VIDEO_FORMATS
