"""
Summarizer Module - AI-powered text summarization using OpenAI API
Generates summaries, key points, keywords, and identifies important moments.
"""

import json
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _get_client() -> "OpenAI":
    if OpenAI is None:
        raise ImportError("OpenAI package is required. Install with: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it with your OpenAI API key."
        )
    
    return OpenAI(api_key=api_key)


def summarize_transcription(
    transcription: str,
    segments: list[dict],
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Generate comprehensive summary using OpenAI API.
    
    Args:
        transcription: Full transcription text
        segments: List of timestamped segments
        model: OpenAI model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
        
    Returns:
        Dict with: summary, key_points, keywords, topics, timestamped_highlights
    """
    client = _get_client()
    
    segments_text = "\n".join(
        f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
        for s in segments
    )
    
    system_prompt = """You are an expert AI Research Assistant. Your goal is to analyze video transcriptions with high academic rigor and clarity.
Given a video transcription with timestamps, you must produce a structured academic analysis in JSON format:
1. main_purpose: Identify the primary objective or thesis of the video (1-2 sentences).
2. key_insights: Extract 5-8 significant takeaways or findings from the discourse.
3. important_concepts: Highlight 3-5 core theoretical or practical concepts discussed.
4. structured_summary: Generate a well-organized, multi-paragraph summary (3-5 paragraphs) covering the introduction, core arguments, and conclusion.
5. keywords: List 5-10 academic keywords or technical terms.
6. timestamped_highlights: Identify 3-6 critical moments with their exact timestamps in seconds.
   Format: {"timestamp": <seconds>, "title": "<academic title>", "description": "<detailed analysis of the moment>"}

Maintain an academic, objective tone. Output as valid JSON with the specified keys."""

    user_prompt = f"""As an AI Research Assistant, analyze this video transcription:

TRANSCRIPTION WITH TIMESTAMPS:
{segments_text}

FULL TEXT:
{transcription}

Provide the structured JSON response."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2, # Lower temperature for more academic/consistent results
        )
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            elif content.startswith("JSON"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        
        # Ensure all expected keys exist (mapping old keys to new ones for compatibility if needed, but here we update the whole flow)
        result.setdefault("main_purpose", "")
        result.setdefault("key_insights", [])
        result.setdefault("important_concepts", [])
        result.setdefault("structured_summary", "")
        result.setdefault("keywords", [])
        result.setdefault("timestamped_highlights", [])
        
        # Backward compatibility for any existing code expecting 'summary' or 'key_points'
        if "summary" not in result:
            result["summary"] = result["main_purpose"] + "\n\n" + result["structured_summary"]
        if "key_points" not in result:
            result["key_points"] = result["key_insights"]
            
        return result
        
    except json.JSONDecodeError as e:
        return {
            "summary": "Summary generation completed but parsing failed.",
            "key_points": [],
            "keywords": [],
            "topics": [],
            "timestamped_highlights": [],
            "_parse_error": str(e),
        }
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}") from e


def extract_keywords_fallback(transcription: str) -> list[str]:
    """Simple fallback keyword extraction if OpenAI is unavailable."""
    import re
    words = re.findall(r"\b[a-zA-Z]{4,}\b", transcription.lower())
    from collections import Counter
    return [w for w, _ in Counter(words).most_common(10)]


def refine_transcription(raw_transcription: str, model: str = "gpt-4o-mini") -> str:
    """
    Refine a raw transcription into clear text with proper sentences and punctuation.
    
    Args:
        raw_transcription: The original Whisper output
        model: OpenAI model to use
        
    Returns:
        Refined transcription text
    """
    client = _get_client()
    
    system_prompt = """You are an expert transcription editor. Your task is to take a raw, potentially 
noisy speech-to-text transcription and refine it into clear, professional text.

Rules:
1. Fix punctuation, capitalization, and sentence structure.
2. Remove filler words (um, uh, like, etc.) unless they are critical for meaning.
3. Correct obvious misspellings based on context.
4. DO NOT change the meaning or intent of the speaker.
5. Provide the output as clear, well-structured paragraphs.
6. Return ONLY the refined text."""

    user_prompt = f"Please refine this transcription:\n\n{raw_transcription}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Refine transcription error: {e}")
        return raw_transcription # Fallback to raw if API fails
