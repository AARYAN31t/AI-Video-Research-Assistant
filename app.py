"""
AI Video Summarization Platform
Full-stack application for video upload, transcription, and AI-powered summarization.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import tempfile
from pathlib import Path

import streamlit as st

from video_processor import (
    extract_audio,
    capture_key_frames,
    get_video_info,
    is_supported_video,
)
from transcriber import transcribe_audio, format_timestamp_long, format_timestamp
from summarizer import summarize_transcription, refine_transcription
from document_exporter import (
    export_summary_markdown,
    export_summary_word,
    export_summary_pdf_enhanced,
)

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .highlight-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .timestamp-btn {
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 8px;
        background: #1E88E5;
        color: white !important;
        border: none;
        cursor: pointer;
        display: inline-block;
        text-decoration: none;
    }
    .timestamp-btn:hover {
        background: #1565C0;
    }
    .keyword-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: #E3F2FD;
        color: #1565C0;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
    }
    .frame-thumbnail {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def download_youtube_video(url: str) -> tuple[str | None, str]:
    """Download video from YouTube URL using yt-dlp.
    Returns (video_path, error_message). On success, error_message is empty."""
    try:
        import yt_dlp
    except ImportError:
        return None, "yt-dlp is not installed. Run: pip install yt-dlp"

    try:
        output_dir = tempfile.mkdtemp()
        out_tmpl = os.path.join(output_dir, "video.%(ext)s")
        ydl_opts = {
            "format": "best[ext=mp4]/best[ext=webm]/best",
            "outtmpl": out_tmpl,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url.strip()])
        for f in os.listdir(output_dir):
            p = os.path.join(output_dir, f)
            if os.path.isfile(p) and f.lower().endswith((".mp4", ".mkv", ".webm", ".m4a")):
                return p, ""
        return None, "No video file was downloaded. The video may be private or restricted."
    except Exception as e:
        err = str(e)
        if "FFmpeg" in err or "ffmpeg" in err:
            return None, "FFmpeg is required. Install from https://ffmpeg.org/download.html"
        if "Private video" in err or "private" in err.lower():
            return None, "This video is private or unavailable."
        if "Video unavailable" in err or "unavailable" in err.lower():
            return None, "Video is unavailable (deleted, age-restricted, or region-blocked)."
        return None, f"Download failed: {err[:200]}"


def _is_youtube_url(url: str) -> bool:
    """Check if URL is YouTube or similar."""
    if not url or not isinstance(url, str):
        return False
    url_lower = url.strip().lower()
    return "youtube.com" in url_lower or "youtu.be" in url_lower


def export_summary_pdf(summary_data: dict) -> bytes | None:
    """Export summary as PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        import io

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y = height - inch

        c.setFont("Helvetica-Bold", 16)
        c.drawString(inch, y, "Video Summary Report")
        y -= 0.5 * inch

        c.setFont("Helvetica", 12)
        c.drawString(inch, y, "Summary:")
        y -= 0.3 * inch
        c.setFont("Helvetica", 10)
        for line in summary_data.get("summary", "").split(". "):
            if line.strip():
                c.drawString(inch + 0.2 * inch, y, line.strip() + ".")
                y -= 0.25 * inch
        y -= 0.3 * inch

        c.setFont("Helvetica", 12)
        c.drawString(inch, y, "Key Points:")
        y -= 0.3 * inch
        c.setFont("Helvetica", 10)
        for pt in summary_data.get("key_points", []):
            c.drawString(inch + 0.2 * inch, y, "• " + pt[:80] + ("..." if len(pt) > 80 else ""))
            y -= 0.25 * inch
        y -= 0.3 * inch

        c.setFont("Helvetica", 12)
        c.drawString(inch, y, "Keywords: " + ", ".join(summary_data.get("keywords", [])[:15]))
        y -= 0.5 * inch

        c.save()
        return buffer.getvalue()
    except Exception:
        return None


def main():
    st.markdown('<p class="main-header">🎬 AI Video Summarization Platform</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a video to generate AI-powered transcriptions, summaries, keywords, and timestamps.</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_size = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="Larger models = better accuracy, slower processing",
        )
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Model for summarization",
        )
        capture_visual = st.checkbox("Capture Visual Highlights", value=True, help="Extract key frames at timestamps")
        st.divider()
        st.markdown("### Supported Formats")
        st.markdown("- MP4, AVI, MKV, MOV, WEBM")
        st.markdown("- Max 5 GB supported")
        st.divider()
        # Theme toggle
        dark_mode = st.toggle("Dark Mode", value=False)
        if dark_mode:
            st.markdown(
                """<style>
                .stApp { background-color: #1a1a2e; }
                </style>""",
                unsafe_allow_html=True,
            )

    # Initialize session state
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "current_time" not in st.session_state:
        st.session_state.current_time = 0

    # Upload section
    st.subheader("📤 Video Upload")
    col_upload, col_yt = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mkv", "mov", "webm", "wmv", "flv", "m4v", "mpg", "mpeg", "3gp", "3g2", "ogv", "mts", "m2ts", "vob", "ts"],
            help="MP4, AVI, MKV, MOV, WEBM, WMV, FLV, M4V, MPG, MPEG, 3GP, OGV, MTS, etc. (up to 5 GB)",
        )

    with col_yt:
        yt_url = st.text_input("Or paste YouTube URL", placeholder="https://youtube.com/... or youtu.be/...")
        if yt_url and _is_youtube_url(yt_url):
            if st.button("▶️ **Run** – Download Video", type="primary", use_container_width=True, key="yt_run_btn"):
                st.session_state.run_yt_url = yt_url

    video_path = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            video_path = tmp.name
        st.session_state.video_name = Path(uploaded_file.name).stem
        st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
    elif st.session_state.get("run_yt_url"):
        url_to_download = st.session_state.run_yt_url
        with st.spinner("⬇️ Downloading from YouTube..."):
            video_path, err_msg = download_youtube_video(url_to_download)
        st.session_state.run_yt_url = None
        if video_path:
            st.session_state.video_name = "YouTube_Video"
            st.success("✅ Video downloaded – click **Run Summary** below to generate AI summary")
        else:
            st.error(f"❌ {err_msg}")

    if video_path and is_supported_video(video_path):
        st.session_state.video_path = video_path

        # Video preview
        st.subheader("🎥 Video Preview")
        st.video(video_path)

        info = get_video_info(video_path)
        st.caption(f"Duration: {format_timestamp_long(info.get('duration', 0))} | {info.get('width', 0)}x{info.get('height', 0)}")

        # Prominent RUN Summary button
        st.markdown("---")
        st.markdown("### ▶️ Run Summary")
        st.caption("Click the button below to generate AI summary, transcription, keywords & timestamps.")
        if st.button("🚀 **Run Summary**", type="primary", use_container_width=True, key="run_summary_btn"):
            progress = st.progress(0, text="Initializing...")
            status = st.empty()

            try:
                # Step 1: Extract audio
                status.info("🔊 Extracting audio from video...")
                progress.progress(15, text="Extracting audio...")
                audio_path = extract_audio(video_path)
                progress.progress(30, text="Audio extracted. Transcribing...")

                # Step 2: Transcribe
                status.info("📝 Converting speech to text (Whisper)...")
                full_text, segments = transcribe_audio(audio_path, model_size=model_size)
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                progress.progress(55, text="Transcription complete. Summarizing...")

                # Step 3: Summarize
                status.info("🤖 Generating AI research analysis (OpenAI)...")
                summary_data = summarize_transcription(full_text, segments, model=openai_model)
                
                # Step 4: Refine Transcription (New feature for clear text)
                status.info("✍️ Refining transcription for clarity...")
                refined_text = refine_transcription(full_text, model=openai_model)
                progress.progress(80, text="Analysis and refinement complete.")

                # Step 5: Visual highlights
                frames_data = []
                if capture_visual and summary_data.get("timestamped_highlights"):
                    status.info("🖼️ Capturing key frames...")
                    timestamps = []
                    for h in summary_data["timestamped_highlights"]:
                        t = h.get("timestamp")
                        try:
                            timestamps.append(float(t))
                        except (TypeError, ValueError):
                            pass
                    if timestamps:
                        frames_data = capture_key_frames(video_path, timestamps[:6])

                progress.progress(100, text="Done!")
                status.success("✅ Processing complete!")

                st.session_state.results = {
                    "transcription": full_text,
                    "refined_transcription": refined_text,
                    "segments": segments,
                    "summary_data": summary_data,
                    "frames": frames_data,
                    "video_path": video_path,
                    "duration": info.get("duration", 0),
                }

            except Exception as e:
                status.error(f"❌ Error: {str(e)}")
                st.exception(e)

    # Display results
    if st.session_state.results:
        r = st.session_state.results
        sd = r["summary_data"]
        video_path = r["video_path"]

        st.divider()
        st.subheader("🔬 AI Research Analysis")

        col1, col2 = st.columns([1.2, 0.8])

        with col1:
            st.markdown("#### 🎯 Main Purpose")
            st.markdown(f'<div class="highlight-box">{sd.get("main_purpose", "N/A")}</div>', unsafe_allow_html=True)

            st.markdown("#### 💡 Key Insights")
            for insight in sd.get("key_insights", []):
                st.markdown(f"- {insight}")

            st.markdown("#### 🧠 Important Concepts")
            concepts = sd.get("important_concepts", [])
            if concepts:
                concept_html = " ".join(f'<span class="keyword-tag" style="background: #E8F5E9; color: #2E7D32;">{c}</span>' for c in concepts)
                st.markdown(f'<div>{concept_html}</div>', unsafe_allow_html=True)

            st.markdown("#### 📋 Structured Summary")
            st.markdown(sd.get("structured_summary", "N/A"))

            st.markdown("#### 🏷️ Keywords")
            keywords = sd.get("keywords", [])
            if keywords:
                kw_html = " ".join(f'<span class="keyword-tag">{k}</span>' for k in keywords)
                st.markdown(f'<div>{kw_html}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### ⏱️ Clickable Timestamps")
            highlights = sd.get("timestamped_highlights", [])
            if highlights:
                for h in highlights:
                    ts = h.get("timestamp", 0)
                    title = h.get("title", "Highlight")
                    desc = h.get("description", "")
                    # Use Streamlit button with callback to seek video
                    if st.button(
                        f"▶️ {format_timestamp(ts)} - {title}",
                        key=f"ts_{ts}",
                        use_container_width=True,
                    ):
                        st.session_state.current_time = ts
                        st.rerun()
                    st.caption(desc)
            else:
                # Fallback to segments
                for seg in r["segments"][:15]:
                    ts = seg["start"]
                    if st.button(
                        f"▶️ {format_timestamp(ts)} - {seg['text'][:50]}...",
                        key=f"seg_{ts}",
                        use_container_width=True,
                    ):
                        st.session_state.current_time = ts
                        st.rerun()

        st.markdown("---")
        st.markdown("#### 📄 Transcription")
        tab_raw, tab_refined = st.tabs(["Refined (Clear Text)", "Raw Whisper Output"])
        
        with tab_refined:
            st.markdown(r.get("refined_transcription", "No refined transcription available."))
            
        with tab_raw:
            with st.expander("View full raw transcription", expanded=False):
                for seg in r["segments"]:
                    st.markdown(f"**{format_timestamp(seg['start'])}** - {seg['text']}")

        # Visual highlights
        if r.get("frames"):
            st.markdown("#### 🖼️ Visual Highlights")
            cols = st.columns(min(3, len(r["frames"])))
            for i, (ts, img_path) in enumerate(r["frames"]):
                with cols[i % len(cols)]:
                    st.image(img_path, caption=format_timestamp(ts), use_container_width=True)

        # Document Export Section
        st.markdown("---")
        st.markdown("#### 📥 Export Summary as Document")
        st.caption("Download your video summary in your preferred format")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        # Get video name for export
        video_name = st.session_state.get("video_name", "Video")
        full_transcription = r.get("transcription", "")
        refined_transcription = r.get("refined_transcription", "")
        
        with col_export1:
            # Enhanced PDF Export
            pdf_bytes = export_summary_pdf_enhanced(sd, full_transcription, refined_transcription, video_name)
            if pdf_bytes:
                st.download_button(
                    "📄 PDF Notes",
                    data=pdf_bytes,
                    file_name=f"{video_name}_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Download as PDF document with full formatting"
                )
            else:
                st.button("📄 PDF Notes", disabled=True, use_container_width=True)
        
        with col_export2:
            # Word Document Export
            word_bytes = export_summary_word(sd, full_transcription, refined_transcription, video_name)
            if word_bytes:
                st.download_button(
                    "📝 Word Document",
                    data=word_bytes,
                    file_name=f"{video_name}_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    help="Download as Word .docx file (editable)"
                )
            else:
                st.info("Install python-docx for Word export: pip install python-docx", icon="ℹ️")
        
        with col_export3:
            # Markdown Export
            md_content = export_summary_markdown(sd, full_transcription, refined_transcription, video_name)
            st.download_button(
                "📋 Markdown Notes",
                data=md_content,
                file_name=f"{video_name}_summary.md",
                mime="text/markdown",
                use_container_width=True,
                help="Download as Markdown file (plain text with formatting)"
            )

    elif not video_path and not st.session_state.results:
        st.info("👆 Upload a video or paste a YouTube URL to get started.")

    # Footer
    st.divider()
    st.caption("AI Video Summarization Platform • Powered by Whisper & OpenAI • Built with Streamlit")


if __name__ == "__main__":
    main()
