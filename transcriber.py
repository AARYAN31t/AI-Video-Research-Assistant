"""
Transcriber Module - Speech-to-Text using OpenAI Whisper
Converts extracted audio to text with timestamps for each spoken segment.
"""

import os

try:
    import whisper
except ImportError:
    whisper = None


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Transcribe audio file to text using Whisper.
    
    Args:
        audio_path: Path to the audio file (mp3, wav, etc.)
        model_size: Whisper model size - tiny, base, small, medium, large
        language: Optional language code (e.g., 'en' for English)
        
    Returns:
        Tuple of (full_transcription_text, segments_list)
        Each segment: {"start": float, "end": float, "text": str}
        
    Raises:
        ImportError: If Whisper is not installed
        FileNotFoundError: If audio file doesn't exist
    """
    if whisper is None:
        raise ImportError("OpenAI Whisper is required. Install with: pip install openai-whisper")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = whisper.load_model(model_size)
    
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=False,
        verbose=False,
    )
    
    full_text = result["text"].strip()
    segments = []
    
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })
    
    return full_text, segments


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_timestamp_long(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
