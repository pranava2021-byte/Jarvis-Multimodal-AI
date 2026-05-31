# 🤖 J.A.R.V.I.S. v7.0 — Just A Rather Very Intelligent System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-7.0-blue?style=for-the-badge)

**A fully agentic, voice-enabled AI assistant built in Python.**  
Powered by Groq's ultra-fast Llama 3.3 70B — works offline for most features.

</div>

---

## ✨ Features

| Category | Capabilities |
|----------|-------------|
| 🧠 **AI Brain** | Groq Llama 3.3 70B · streaming responses · persistent memory |
| 🎙 **Voice** | Wake-word ("Hey JARVIS") · speech input · TTS output |
| 🍕 **Food Ordering** | Domino's · Pizza Hut · Zomato · Swiggy · McDonald's |
| 🚗 **Cab Booking** | Ola · Uber · Rapido · Google Maps |
| 🗺 **Navigation** | Google Maps directions from/to anywhere |
| ▶️ **YouTube** | Search and open any video instantly |
| 📈 **Stocks** | Live stock prices via Yahoo Finance |
| 💱 **Currency** | Real-time exchange rates for 150+ currencies |
| 📧 **Email** | Open Gmail drafts with pre-filled content |
| 🔍 **File Analysis** | AI-powered analysis of .py, .csv, .txt, images, and more |
| 📰 **News** | Live headlines (world, tech, India, science, sports) |
| 🌤 **Weather** | Real-time weather for any city |
| 📖 **Wikipedia** | Instant summaries for any topic |
| ⏰ **Reminders** | Timed reminders and alarms |
| 📊 **Plots** | Matplotlib graphs (sin, cos, tan, x²) |
| 🐍 **Python Sandbox** | Execute Python code directly |
| 🔌 **Plugins** | Drop custom `.py` files in `/plugins` to extend |
| 🌐 **Network** | IP, connectivity, hostname diagnostics |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/jarvis-ai.git
cd jarvis-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key (FREE)
Get a free Groq API key at: **https://console.groq.com**

Create a `.env` file in the project folder:
```
GROQ_API_KEY=gsk_your_key_here
```

> ⚠️ **Never commit your `.env` to GitHub!** It's already in `.gitignore`.

### 4. Run JARVIS
```bash
# Normal mode (text + voice on demand)
python jarvis.py

# Always-on voice mode (wake word activated)
python jarvis.py --voice
```

---

## 🎯 Example Commands

```
order pizza                       → Shows Domino's, Pizza Hut, Zomato options
book a cab to Connaught Place     → Shows Ola, Uber, Rapido options
navigate to India Gate            → Opens Google Maps
play Bohemian Rhapsody on youtube → Opens YouTube search
stock RELIANCE                    → Live NSE stock price
convert 1000 INR to USD          → Real-time currency conversion
send email to boss@company.com   → Opens Gmail draft
analyze data.csv                 → AI-powered CSV analysis
weather in Mumbai                → Live weather
who is APJ Abdul Kalam           → Wikipedia summary
remind me in 10 min to drink water
set alarm for 6:30 am
take a note: Buy groceries
news india                       → Top India headlines
plot sin                         → Sine wave graph
run print("Hello World")         → Python execution
screenshot                       → Captures your screen
```

---

## 🔌 Plugin System

Drop any `.py` file in the `plugins/` folder:

```python
# plugins/my_custom_skill.py

TRIGGER_WORDS = ["flip a coin", "heads or tails"]

def run(query: str) -> str:
    import random
    result = random.choice(["Heads 🪙", "Tails 🪙"])
    return f"## 🪙 Coin Flip\n\n**Result: {result}**"
```

JARVIS auto-loads all plugins on startup.

---

## 📁 Project Structure

```
jarvis-ai/
├── jarvis_v6.py          # Main application
├── requirements.txt      # Dependencies
├── .env                  # ← YOUR API KEY (never commit this)
├── .gitignore
├── README.md
├── plugins/              # Drop custom skill .py files here
├── jarvis_memory.json    # Auto-created: conversation memory
└── jarvis_notes.txt      # Auto-created: saved notes
```

---

## 🔒 Security

- API keys are loaded from `.env` using `python-dotenv`
- `.env` is listed in `.gitignore` — **never pushed to GitHub**
- No keys are hardcoded anywhere in the source code
- Safe to open-source the entire project except `.env`

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq · Llama 3.3 70B Versatile |
| Vision | Groq · Llama 3.2 11B Vision |
| UI | Rich (terminal) |
| Voice In | SpeechRecognition + PyAudio |
| Voice Out | pyttsx3 (offline TTS) |
| Data | pandas · numpy |
| Charts | matplotlib |
| Browser | webbrowser + selenium (optional) |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ by a Class 12 student with big dreams.**  
*"The best way to predict the future is to invent it."*

</div>
