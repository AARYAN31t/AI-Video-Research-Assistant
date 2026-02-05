# 🎬 AI Video Summarization Platform

A full-stack AI-based video summarization platform that allows users to upload videos and automatically generates text summaries, timestamps, keywords, and visual highlights using Artificial Intelligence.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Workflow](#-workflow)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Why NLP, Audio Processing & AI?](#-why-nlp-audio-processing--ai)
- [Future Scope](#-future-scope)
- [Real-World Use Cases](#-real-world-use-cases)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### Core Features
- **Video Upload** – Upload MP4, AVI, MKV, MOV, WEBM, WMV, FLV, M4V, MPG, MPEG, 3GP, OGV, MTS, and more (up to 5 GB)
- **Video Preview** – Preview uploaded videos before processing
- **Audio Extraction** – Extract audio from video using MoviePy
- **Speech-to-Text** – Convert audio to text with Whisper (OpenAI)
- **Timestamped Transcription** – Each spoken segment includes start/end timestamps
- **AI Summarization** – OpenAI API generates:
  - Concise summary (2-4 sentences)
  - Bullet-point key highlights (5-8 points)
  - Important keywords (5-10 terms)
  - Major topics discussed
- **Timestamped Highlights** – Clickable timestamps to jump to important moments
- **Visual Summary** – Key frames captured at highlight timestamps
- **PDF Export** – Download summary as PDF

### Optional Enhancements
- **YouTube Support** – Paste YouTube URLs to summarize (requires `yt-dlp`)
- **Dark/Light Mode** – Toggle in sidebar
- **Multi-language** – Whisper supports 99+ languages

---

## 🔄 Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Video       │────▶│  2. Audio       │────▶│  3. Whisper     │
│     Upload      │     │     Extraction  │     │     Speech-to-  │
│  (MP4/AVI/MKV)  │     │  (MoviePy)      │     │     Text        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  6. Display     │◀────│  5. Key Frame   │◀────│  4. OpenAI      │
│     Results     │     │     Capture     │     │     Summarize   │
│  (Streamlit UI) │     │  (Visual Thumbs)│     │  (GPT)          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Step-by-Step Process:**

1. **Video Upload** – User uploads a video file or pastes a YouTube URL
2. **Audio Extraction** – MoviePy extracts the audio track as MP3
3. **Speech-to-Text** – Whisper transcribes the audio with word-level timestamps
4. **AI Summarization** – OpenAI GPT analyzes the transcription and produces:
   - Summary, key points, keywords, topics, and important timestamps
5. **Key Frame Capture** – Frames are extracted at highlighted timestamps
6. **Display** – Streamlit renders all outputs in a clean, interactive UI

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Web UI** | Streamlit |
| **Video/Audio** | MoviePy, FFmpeg |
| **Speech-to-Text** | OpenAI Whisper |
| **AI Summarization** | OpenAI API (GPT-4o-mini, GPT-4o) |
| **PDF Export** | ReportLab |
| **YouTube** | yt-dlp (optional) |

---

## 📦 Installation

### Prerequisites
- **Python 3.9+**
- **FFmpeg** – Required for video/audio processing  
  - Windows: `winget install FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Setup

1. **Clone or navigate to the project:**
   ```bash
   cd "New folder"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key:**
   - Copy `.env.example` to `.env`
   - Add your key: `OPENAI_API_KEY=sk-your-key-here`
   - Or set environment variable: `set OPENAI_API_KEY=sk-...` (Windows)

---

## 🚀 Usage

### Run on localhost

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

### Steps

1. **Upload** a video (MP4, AVI, MKV, MOV, WEBM, WMV, FLV, etc. – up to 5 GB) or paste a YouTube URL
2. Optionally adjust **Whisper model** (base recommended) and **OpenAI model**
3. Click **Generate Summary**
4. Wait for processing (audio extraction → transcription → summarization)
5. View results: summary, key points, keywords, timestamps, visual highlights
6. Download summary as PDF if needed

---

## 📂 Project Structure

```
├── app.py              # Streamlit application (main entry point)
├── video_processor.py  # Audio extraction & frame capture (MoviePy)
├── transcriber.py      # Whisper speech-to-text
├── summarizer.py       # OpenAI summarization logic
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
└── README.md           # Documentation
```

### Module Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI, file upload, progress display, results rendering |
| `video_processor.py` | Extract audio from video, capture frames at timestamps |
| `transcriber.py` | Load Whisper model, transcribe audio with timestamps |
| `summarizer.py` | Call OpenAI API, parse JSON response (summary, keywords, etc.) |

---

## 🧠 Why NLP, Audio Processing & AI?

### Natural Language Processing (NLP)
- **Text understanding** – GPT analyzes the transcription to identify main ideas, themes, and structure
- **Keyword extraction** – Identifies important terms and concepts
- **Summarization** – Condenses long transcripts into readable summaries

### Audio Processing
- **Audio extraction** – Videos are primarily visual; audio contains the spoken content
- **Format conversion** – Whisper expects audio files; MoviePy bridges video → audio

### Artificial Intelligence
- **Whisper** – State-of-the-art speech recognition, handles accents and noise
- **OpenAI GPT** – Understands context, generates coherent summaries and highlights
- **Automation** – Replaces manual transcription and summarization

---

## 🔮 Future Scope

- **Batch processing** – Process multiple videos in a queue
- **Speaker diarization** – Identify who said what
- **Sentiment analysis** – Detect tone (positive, neutral, negative)
- **Multi-language UI** – Localize the interface
- **Cloud deployment** – Deploy on Streamlit Cloud, AWS, or GCP
- **Database storage** – Save summaries for later retrieval
- **API mode** – REST API for programmatic access

---

## 🌍 Real-World Use Cases

| Use Case | Description |
|----------|-------------|
| **Education** | Summarize lectures for students; create study guides |
| **Meetings** | Turn meeting recordings into actionable notes |
| **Content Creators** | Generate video descriptions and timestamps for YouTube |
| **Legal/Medical** | Transcribe depositions, consultations; extract key points |
| **Research** | Quickly scan conference talks and seminars |
| **Accessibility** | Provide text alternatives for video content |
| **Compliance** | Document training videos with searchable transcripts |

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **FFmpeg not found** | Install FFmpeg and add to PATH |
| **OPENAI_API_KEY error** | Set key in `.env` or environment variables |
| **Out of memory** | Use smaller Whisper model (tiny/base); process shorter videos |
| **Slow transcription** | Use `tiny` or `base` Whisper model; ensure GPU if available |
| **YouTube download fails** | Install/update yt-dlp: `pip install -U yt-dlp` |
| **Large video timeout** | Increase Streamlit timeout or process in chunks |

---

## 📄 License

MIT License – feel free to use and modify for your projects.

---

**Built with ❤️ using Python, Whisper, OpenAI, and Streamlit**
