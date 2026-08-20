<p align="center">
  <img src="assets/hero.svg" alt="🎬 AI Video Director Hero Banner" width="100%" />
</p>

<h1 align="center">🎬 AI Video Director</h1>

<p align="center">
  <strong>Reverse-compiling visual storytelling, cinematic camera work, lighting, and directorial intent using Vision-Language Models (VLM).</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python" /> <img src="https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" /> <img src="https://img.shields.io/badge/OpenCV-4.8+-5c3ee8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" /> <img src="https://img.shields.io/badge/OpenRouter-Multi-VLM-6366f1?style=for-the-badge&logo=openai&logoColor=white" alt="OpenRouter" />
</p>

---

## ✨ Features (Key Outcomes & Capabilities)

| Icon | Feature | Outcome & Real Proof |
| :---: | :--- | :--- |
| 🎥 | **Smart Keyframe Extraction** | Extracts optimal keyframes using OpenCV scene variance, preserving narrative context while minimizing token cost |
| 🧠 | **State-of-the-Art Multi-VLM** | Direct integration with Gemini 3.1 Pro, Qwen 2.5 VL, and Claude 3.5 Sonnet via OpenRouter |
| 📐 | **Cinematography Deconstruction** | Analyzes shot scale, angle, camera dynamics (pan, tilt, dolly), key/fill lighting, color grading, and emotional arc |
| 📊 | **Structured Director Report** | Generates production-ready breakdown tables and directorial critique instantly in Markdown |

---

## 📊 Architecture & Flow

```mermaid
graph LR
  Video[📹 Input Video .mp4] --> CV[👁️ OpenCV Keyframe Extractor]
  CV --> Frames[🖼️ Selected Keyframes]
  Frames --> VLM[🧠 Multi-VLM via OpenRouter]
  VLM --> Analysis[📊 Cinematography & Director Intent Analysis]
  Analysis --> UI[💻 Streamlit Interactive Dashboard]
  
  classDef primary fill:#ec4899,stroke:#be185d,stroke-width:2px,color:#fff;
  classDef accent fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff;
  class CV,VLM primary;
  class Analysis,UI accent;
```

---

## 📁 Project Structure

```bash
ai-video-director/
├── 📁 assets/                 # High-resolution SVG banners & media
├── 📄 app.py                  # Streamlit application entry point
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites
- Check language runtimes (Python / Node.js) and system dependencies.

```bash
# 1. Clone & enter repository
git clone https://github.com/LoNebula/ai-video-director.git
cd ai-video-director

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run app.py
```

---

## 💡 Usage Notes & Tips

> [!TIP]
> Ensure all required environment variables and dependencies are properly configured before execution.

---

<p align="center">
  Released under the <a href="LICENSE">MIT License</a>. Made with ❤️ by <a href="https://github.com/LoNebula">LoNebula</a>
</p>
