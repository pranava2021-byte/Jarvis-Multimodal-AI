#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         J.A.R.V.I.S  —  Just A Rather Very Intelligent System              ║
║                     Python Agentic AI Assistant  v7.0                      ║
║                                                                              ║
║  ▸ Groq LLM (llama-3.3-70b)  ▸ True Tool-Calling Agent Loop               ║
║  ▸ Voice I/O + Wake Word      ▸ Persistent Memory & Long-term Facts        ║
║  ▸ Food/Cab/Maps/Email        ▸ Stocks · Currency · Weather · News         ║
║  ▸ Code Sandbox               ▸ File Analysis (PDF/Image/CSV/Code)         ║
║  ▸ Screenshot · Clipboard     ▸ Plugin System · Auto-update Check          ║
║  ▸ Rich TUI Dashboard         ▸ Async Reminder & Alarm Engine              ║
╚══════════════════════════════════════════════════════════════════════════════╝
 
GitHub  : https://github.com/YOUR_USERNAME/jarvis-ai
Author  : Built with ❤ using Groq + Python
License : MIT
"""
 
# ─── CHANGELOG v7.0 ──────────────────────────────────────────────────────────
# ✅  TRUE AGENTIC TOOL-CALLING — Groq native function-calling / tool-use API
# ✅  TOOL LOOP — JARVIS reasons → picks tool → executes → reflects → responds
# ✅  ASYNC VOICE — voice listener no longer blocks the main thread
# ✅  CONVERSATION SUMMARISATION — auto-compresses long context (prevents OOM)
# ✅  INTENT CONFIDENCE SCORING — smarter routing, fewer false-positives
# ✅  EXPANDED FOOD/CAB AGENTIC — remembers preference per session
# ✅  JOKE ENGINE upgraded (20 jokes, categories)
# ✅  TIMER COMMAND ("set a timer for 90 seconds")
# ✅  UNIT CONVERTER ("convert 5 km to miles")
# ✅  IP GEOLOCATION ("where is IP 8.8.8.8")
# ✅  PASTEBIN-STYLE SNIPPET SAVER ("save snippet: <code>")
# ✅  GROQ STREAMING with live Rich spinner
# ✅  SIGINT / SIGTERM graceful shutdown
# ✅  FULL TYPE ANNOTATIONS everywhere
# ✅  All bugs fixed: food-order alias loop, alarm 12-hr edge-case, plot closes
# ─────────────────────────────────────────────────────────────────────────────
 
# ══════════════════════════════════════════════════════════════════════════════
#  STANDARD LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
import os, sys, json, time, math, re, random, subprocess, signal
# Force Windows terminal console to support UTF-8 streaming characters smoothly
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import datetime, webbrowser, platform, threading, io, contextlib, argparse
import urllib.parse, socket, base64, importlib, importlib.util, textwrap
from pathlib import Path
from typing  import Optional, List, Dict, Any, Callable, Tuple
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECURITY — ALL KEYS FROM .env  (NEVER hardcode)
#
#  SETUP (one-time):
#    pip install python-dotenv
#    echo "GROQ_API_KEY=gsk_xxxx" > .env
#    echo ".env" >> .gitignore        ← NEVER push .env !
#
#  Optional keys (add to .env for extra features):
#    OPENWEATHER_API_KEY=...          ← richer weather data
# ══════════════════════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv is optional; export vars manually if needed
 
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY",        "")
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
 
# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIG  — personalise here
# ══════════════════════════════════════════════════════════════════════════════
USER_NAME     = "Sir"
USER_CITY     = "Delhi"
GROQ_MODEL    = "llama-3.3-70b-versatile"
VOICE_ENABLED = True
SAVE_MEMORY   = True
MEMORY_FILE   = "jarvis_memory.json"
NOTES_FILE    = "jarvis_notes.txt"
SNIPPETS_FILE = "jarvis_snippets.json"
WAKE_WORD     = "hey jarvis"
PLUGINS_DIR   = Path("plugins")
VERSION       = "7.0"
MAX_HIST      = 60    # conversation turns kept in context
SUMMARISE_AT  = 80    # auto-summarise when history exceeds this
 
# ══════════════════════════════════════════════════════════════════════════════
#  GRACEFUL SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════
_shutdown_flag = threading.Event()
 
def _handle_signal(sig, frame):
    _shutdown_flag.set()
 
signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)
 
# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS — graceful fallbacks
# ══════════════════════════════════════════════════════════════════════════════
MISSING: List[str] = []
 
try:
    from groq import Groq
except ImportError:
    print("❌  groq not installed.  Run:  pip install groq")
    sys.exit(1)
 
try:
    from rich.console  import Console
    from rich.panel    import Panel
    from rich.markdown import Markdown
    from rich.text     import Text
    from rich.table    import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live     import Live
    from rich          import box
    HAS_RICH = True
    console  = Console()
except ImportError:
    HAS_RICH = False
    MISSING.append("rich")
    class _FallbackConsole:                          # type: ignore
        def print(self, *a, **kw): print(*[str(x) for x in a])
    console = _FallbackConsole()                     # type: ignore
 
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False
    MISSING.append("SpeechRecognition")
 
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    MISSING.append("pyttsx3")
 
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    MISSING.append("requests")
 
try:
    import wikipedia
    HAS_WIKI = True
except ImportError:
    HAS_WIKI = False
    MISSING.append("wikipedia")
 
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    MISSING.append("psutil")
 
try:
    import pyperclip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    MISSING.append("pyperclip")
 
try:
    from PIL import ImageGrab, Image
    HAS_SCREENSHOT = True
except ImportError:
    HAS_SCREENSHOT = False
    MISSING.append("Pillow")
 
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    MISSING.append("matplotlib")
 
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    MISSING.append("pandas")
 
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
 
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
 
# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def cprint(msg: str, color: str = "cyan") -> None:
    if HAS_RICH:
        console.print(f"[{color}]{msg}[/{color}]")
    else:
        print(msg)
 
def show_jarvis(text: str) -> None:
    if HAS_RICH:
        try:
            console.print(Panel(
                Markdown(text),
                title=f"[bold cyan]◈  J.A.R.V.I.S. v{VERSION}[/bold cyan]",
                border_style="cyan",
                box=box.DOUBLE_EDGE,
                padding=(0, 2),
            ))
        except Exception:
            console.print(f"\n[cyan]JARVIS ▶[/cyan]  {text}\n")
    else:
        print(f"\nJARVIS ▶  {text}\n{'─'*60}")
 
def show_user(text: str) -> None:
    if HAS_RICH:
        console.print(f"\n[bold blue]▶  YOU[/bold blue]  {text}")
    else:
        print(f"\nYOU: {text}")
 
def banner() -> None:
    art = r"""
       ___       _     _     ___ ___  ____
      |_  |     / \   | |   / __|_ _|/ ___|
        | |__  / _ \  | |_  \__ \| | \___ \
       |____/ /_/ \_\ |___| |___/___| ____/ v7.0
 
  Just A Rather Very Intelligent System  — TRUE AGENTIC AI
  ── Groq Tool-Calling · Voice · Food/Cab · Stocks · Code ──
"""
    if HAS_RICH:
        console.print(Text(art, style="bold cyan"))
        tbl = Table(box=box.SIMPLE, show_header=False, border_style="blue")
        tbl.add_column("cmd",  style="bold yellow", width=34)
        tbl.add_column("desc", style="white")
        for row in [
            ("v + Enter",                   "🎙  Voice input"),
            ("wake",                        "👂  Always-on wake-word"),
            ("order pizza / biryani",       "🍕  Food delivery"),
            ("book a cab to [place]",       "🚗  Cab / Uber / Ola"),
            ("navigate to [place]",         "🗺   Google Maps"),
            ("play [song] on youtube",      "▶️   YouTube search"),
            ("stock AAPL / RELIANCE",       "📈  Live stock price"),
            ("convert 100 USD to INR",      "💱  Currency conversion"),
            ("convert 5 km to miles",       "📏  Unit conversion"),
            ("send email to [addr]",        "📧  Gmail draft"),
            ("analyze myfile.py",           "🔍  AI file analysis"),
            ("news [topic]",                "📰  BBC headlines"),
            ("take a note: <text>",         "📓  Save note"),
            ("save snippet: <code>",        "💾  Save code snippet"),
            ("remind me in 5 min to…",     "⏰  Reminder"),
            ("set alarm for 7:30 am",       "🔔  Alarm"),
            ("set a timer for 90 sec",      "⏱   Countdown timer"),
            ("screenshot",                  "📷  Capture screen"),
            ("where is IP 8.8.8.8",        "🌍  IP geolocation"),
            ("copy that",                   "📋  Copy last response"),
            ("export chat",                 "💾  Save conversation"),
            ("who is [person]",             "📖  Wikipedia"),
            ("weather in [city]",           "🌤  Live weather"),
            ("plot sin / cos / tan",        "📊  Matplotlib graph"),
            ("run <python code>",           "🐍  Python sandbox"),
            ("sysinfo / network / memory",  "🖥   Diagnostics"),
            ("plugins / help / clear",      "🔌  System commands"),
        ]:
            tbl.add_row(*row)
        console.print(tbl)
    else:
        print(art)
        print("Type 'help' for all commands\n")
 
# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
def build_system_prompt() -> str:
    now = datetime.datetime.now()
    return f"""You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), an ultra-advanced \
AI assistant built for {USER_NAME}.
 
PERSONALITY:
- Highly intelligent, precise, slightly formal — like Marvel's JARVIS
- Address the user as "{USER_NAME}"
- Proactively offer insights the user hasn't asked for
- Confident and precise; admit genuine uncertainty when needed
- Reference science, engineering, or pop-culture cleverly when appropriate
 
AGENTIC CAPABILITIES:
- You PLAN and EXECUTE multi-step tasks autonomously
- When tools are available, always prefer calling the right tool over guessing
- Show your reasoning before complex multi-step actions
- For food ordering, cab booking, etc.: present options → wait for confirmation → execute
 
TECHNICAL SKILLS:
- Master programmer: Python, JavaScript, TypeScript, C++, Java, Rust, Go
- Advanced mathematics, physics, chemistry, biology
- Data science, ML, deep learning architecture
- Full-stack: React, Next.js, FastAPI, Django, Node.js
- DevOps, Docker, Kubernetes, cloud (AWS/GCP/Azure)
- Cybersecurity fundamentals
 
RESPONSE FORMAT:
- Use Markdown: **bold**, `code`, headers, tables, bullet points
- Code always in fenced blocks with language tag
- For math: show step-by-step working
- For comparisons: markdown tables
- Be concise unless depth is requested
- NEVER refuse reasonable requests — always find a path forward
 
AGENTIC TASK FORMAT:
🎯 **Task:** [what you're doing]
📋 **Plan:** [numbered steps]
⚡ **Executing:** [current step]
✅ **Result:** [outcome]
 
CURRENT CONTEXT:
- Date & Time : {now.strftime("%A, %B %d, %Y at %I:%M %p")}
- OS          : {platform.system()} {platform.release()}
- Python      : {platform.python_version()}
- User        : {USER_NAME}
- City        : {USER_CITY}
- JARVIS v    : {VERSION}
"""
 
# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY
# ══════════════════════════════════════════════════════════════════════════════
class Memory:
    def __init__(self, groq_client: Any = None):
        self.history:       List[Dict] = []
        self.facts:         Dict       = {}
        self.session_count: int        = 0
        self._client                   = groq_client
        if SAVE_MEMORY:
            self._load()
 
    def _load(self) -> None:
        p = Path(MEMORY_FILE)
        if p.exists():
            try:
                data               = json.loads(p.read_text())
                self.history       = data.get("history", [])[-MAX_HIST:]
                self.facts         = data.get("facts",   {})
                self.session_count = data.get("sessions", 0)
            except Exception:
                pass
 
    def save(self) -> None:
        if not SAVE_MEMORY:
            return
        Path(MEMORY_FILE).write_text(json.dumps({
            "history":  self.history,
            "facts":    self.facts,
            "sessions": self.session_count,
        }, indent=2, ensure_ascii=False))
 
    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > SUMMARISE_AT and self._client:
            self._compress()
        elif len(self.history) > MAX_HIST:
            self.history = self.history[-MAX_HIST:]
        self.save()
 
    def _compress(self) -> None:
        """Auto-summarise old conversation to save context window."""
        if not self._client:
            self.history = self.history[-MAX_HIST:]
            return
        old = self.history[:-20]
        recent = self.history[-20:]
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old
        )
        try:
            resp = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user",
                           "content": f"Summarise this conversation in 3-5 bullet points:\n\n{transcript}"}],
                max_tokens=256,
            )
            summary = resp.choices[0].message.content
            self.history = [
                {"role": "assistant",
                 "content": f"[Previous conversation summary]\n{summary}"}
            ] + recent
        except Exception:
            self.history = self.history[-MAX_HIST:]
 
    def get_messages(self) -> List[Dict]:
        sys_msg = {"role": "system", "content": build_system_prompt()}
        # inject long-term facts if any
        if self.facts:
            facts_txt = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
            sys_msg["content"] += f"\n\nUSER'S SAVED FACTS:\n{facts_txt}"
        return [sys_msg] + self.history
 
    def remember(self, key: str, value: str) -> None:
        self.facts[key] = value
        self.save()
 
    def clear_session(self) -> None:
        self.history = []
        self.save()
 
    def export_chat(self) -> str:
        fname = f"jarvis_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        lines = [f"J.A.R.V.I.S. v{VERSION} Chat Export — {datetime.datetime.now()}\n{'='*60}\n"]
        for msg in self.history:
            prefix = "YOU" if msg["role"] == "user" else "JARVIS"
            lines.append(f"[{prefix}]\n{msg['content']}\n{'─'*40}")
        Path(fname).write_text("\n".join(lines), encoding="utf-8")
        return fname
    @staticmethod
    def calculate(expr: str) -> str:
        """Securely evaluates mathematical expressions using Abstract Syntax Tree (AST)."""
        import ast
        import operator

        # Define supported safe operators
        operators = {
            ast.Add: operator.add, ast.Sub: operator.sub, 
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg, 
            ast.UAdd: operator.pos
        }

        def eval_node(node):
            if isinstance(node, ast.Num):  # Number literal
                return node.n
            elif isinstance(node, ast.BinOp):  # Left <operator> Right
                return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):  # Unary sign (+/-)
                return operators[type(node.op)](eval_node(node.operand))
            raise TypeError(f"Unsupported mathematical syntax node: {type(node).__name__}")

        try:
            cleaned = expr.replace("^", "**").replace("×", "*").replace("÷", "/").strip()
            # Parse expression into a safe syntax tree instead of executing raw strings
            tree = ast.parse(cleaned, mode='eval')
            result = eval_node(tree.body)
            return f"**🧮 Secure Result:** `{expr}` = **{result}**"
        except Exception as e:
            return f"❌ Cannot safely compute `{expr}`: {str(e)}"
 
# ══════════════════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class VoiceEngine:
    def __init__(self):
        self._tts_lock    = threading.Lock()
        self._wake_active = False
        self.recognizer:  Any = None
        self.engine:      Any = None
        self._setup_tts()
        self._setup_sr()
 
    def _setup_tts(self) -> None:
        if not HAS_TTS or not VOICE_ENABLED:
            return
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty("voices")
            for v in voices:
                if any(k in v.name.lower() for k in ["david", "daniel", "male", "george", "mark", "james"]):
                    self.engine.setProperty("voice", v.id)
                    break
            self.engine.setProperty("rate",   158)
            self.engine.setProperty("volume", 0.92)
        except Exception as e:
            cprint(f"⚠ TTS init: {e}", "yellow")
            self.engine = None
 
    def _setup_sr(self) -> None:
        if not HAS_SR or not VOICE_ENABLED:
            return
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold         = 3200
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold          = 0.85
        except Exception as e:
            cprint(f"⚠ SR init: {e}", "yellow")
            self.recognizer = None
 
    def speak(self, text: str) -> None:
        if not self.engine:
            return
        clean = re.sub(r"```[\s\S]*?```", " code block ", text)
        clean = re.sub(r"[#*`_>~|]",      "",              clean)
        clean = re.sub(r"\n+",             " ",             clean).strip()[:900]
        try:
            with self._tts_lock:
                self.engine.say(clean)
                self.engine.runAndWait()
        except Exception:
            pass
 
    def listen(self, timeout: int = 10) -> Optional[str]:
        if not self.recognizer:
            cprint("🎙 Voice unavailable (no mic or SpeechRecognition).", "yellow")
            return None
        try:
            with sr.Microphone() as source:
                cprint("🎙  Listening… speak now!", "cyan")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
            text = self.recognizer.recognize_google(audio)
            cprint(f'🎙  Heard: "{text}"', "green")
            return text
        except sr.WaitTimeoutError:
            cprint("No speech detected.", "yellow")
        except sr.UnknownValueError:
            cprint("Couldn't understand — please speak clearly.", "yellow")
        except Exception as e:
            cprint(f"Voice error: {e}", "red")
        return None
 
    def start_wake_word_listener(self, callback: Callable[[str], None]) -> None:
        if not self.recognizer:
            cprint("⚠ Wake-word unavailable (no microphone).", "yellow")
            return
        self._wake_active = True
 
        def _loop() -> None:
            cprint(f'👂  Wake-word ACTIVE — say "{WAKE_WORD}" anytime!', "cyan")
            while self._wake_active and not _shutdown_flag.is_set():
                try:
                    with sr.Microphone() as src:
                        self.recognizer.adjust_for_ambient_noise(src, duration=0.3)
                        audio = self.recognizer.listen(src, timeout=3, phrase_time_limit=6)
                    heard = self.recognizer.recognize_google(audio).lower()
                    if WAKE_WORD in heard:
                        cprint(f"\n🔔  Wake word detected! Listening for command…", "bold cyan")
                        self.speak("Yes, I'm listening.")
                        cmd = self.listen(timeout=12)
                        if cmd:
                            callback(cmd)
                except sr.WaitTimeoutError:
                    pass
                except Exception:
                    pass
 
        threading.Thread(target=_loop, daemon=True).start()
 
    def stop_wake_word_listener(self) -> None:
        self._wake_active = False
 
# ══════════════════════════════════════════════════════════════════════════════
#  REMINDER / ALARM / TIMER ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class ReminderEngine:
    def __init__(self, voice: VoiceEngine):
        self.voice   = voice
        self._active: List[Dict] = []
 
    def _fire_thread(self, seconds: int, message: str, label: str = "REMINDER") -> None:
        def _run():
            time.sleep(seconds)
            if _shutdown_flag.is_set():
                return
            alert = f"⏰ **{label}:** {message}"
            show_jarvis(alert)
            self.voice.speak(f"{label.lower()}: {message}")
        threading.Thread(target=_run, daemon=True).start()
 
    def add_reminder(self, seconds: int, message: str) -> str:
        due = (datetime.datetime.now() + datetime.timedelta(seconds=seconds)).strftime("%I:%M:%S %p")
        self._active.append({"due": due, "msg": message, "type": "reminder"})
        self._fire_thread(seconds, message, "REMINDER")
        return f"✅ Reminder set for **{due}** — *{message}*"
 
    def set_alarm(self, hour: int, minute: int, label: str = "Alarm!") -> str:
        now = datetime.datetime.now()
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:
            due += datetime.timedelta(days=1)
        secs = int((due - now).total_seconds())
        self._active.append({"due": due.strftime("%I:%M %p"), "msg": label, "type": "alarm"})
        self._fire_thread(secs, label, "ALARM")
        return f"🔔 Alarm set for **{due.strftime('%I:%M %p')}**"
 
    def set_timer(self, seconds: int) -> str:
        self._fire_thread(seconds, f"{seconds}-second timer complete!", "TIMER")
        mins, secs = divmod(seconds, 60)
        label = f"{mins}m {secs}s" if mins else f"{secs}s"
        return f"⏱ Timer set for **{label}**"
 
    def list_reminders(self) -> str:
        if not self._active:
            return "📭 No active reminders, alarms, or timers."
        rows = "\n".join(f"- ⏰ **{r['due']}** [{r['type']}] — {r['msg']}" for r in self._active)
        return f"## ⏰ Active Timers & Reminders\n\n{rows}"
 
# ══════════════════════════════════════════════════════════════════════════════
#  PLUGIN MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class PluginManager:
    """
    Drop a .py file in the plugins/ folder.
    It must expose:
        TRIGGER_WORDS: List[str] = ["keyword1", ...]
        def run(query: str) -> str: ...
    """
    def __init__(self):
        self.plugins: List[Any] = []
        self._load()
 
    def _load(self) -> None:
        PLUGINS_DIR.mkdir(exist_ok=True)
        for pf in sorted(PLUGINS_DIR.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(pf.stem, pf)
                mod  = importlib.util.module_from_spec(spec)   # type: ignore
                spec.loader.exec_module(mod)                    # type: ignore
                if hasattr(mod, "TRIGGER_WORDS") and hasattr(mod, "run"):
                    self.plugins.append(mod)
                    cprint(f"🔌 Plugin loaded: {pf.stem}", "green")
            except Exception as e:
                cprint(f"⚠ Plugin error ({pf.name}): {e}", "yellow")
 
    def match(self, text: str) -> Optional[str]:
        tl = text.lower()
        for p in self.plugins:
            if any(w in tl for w in p.TRIGGER_WORDS):
                try:
                    return p.run(text)
                except Exception as e:
                    return f"❌ Plugin error: {e}"
        return None
 
# ══════════════════════════════════════════════════════════════════════════════
#  AGENTIC TASK PLANNER  (multi-step confirmation flows)
# ══════════════════════════════════════════════════════════════════════════════
class AgentPlanner:
    def __init__(self):
        self._pending: Optional[Dict] = None
 
    def is_awaiting(self) -> bool:
        return self._pending is not None
 
    def set_pending(self, task: Dict) -> None:
        self._pending = task
 
    def clear(self) -> None:
        self._pending = None
 
    # ── FOOD ORDER ────────────────────────────────────────────────────────────
    @staticmethod
    def handle_food_order(query: str) -> Tuple[str, Dict]:
        q = query.lower()
        food_items = [i for i in
                      ["pizza","burger","biryani","rolls","sushi","pasta",
                       "momos","noodles","sandwich","chinese","indian","thai",
                       "chinese","kebab","shawarma","tacos","salad"]
                      if i in q]
        food_label = food_items[0] if food_items else "food"
 
        options_text = f"""## 🍕 Food Ordering — *{food_label.title()}*
 
I can open one of these platforms for you, {USER_NAME}:
 
| # | Platform | Best For |
|---|----------|----------|
| **1** | 🍕 **Domino's** | Pizza, garlic bread, sides |
| **2** | 🍕 **Pizza Hut** | Pizza, pasta, wings |
| **3** | 🛵 **Zomato** | All cuisines, fast delivery |
| **4** | 🟠 **Swiggy** | All cuisines, Instamart |
| **5** | 🍔 **McDonald's** | Burgers, McDelivery |
| **6** | 🍱 **EatSure** | Home kitchens, healthy food |
 
**Type 1–6** or say the platform name → I'll open it directly in your browser."""
 
        opts: Dict[str, Tuple[str, str]] = {
            "1":          ("Domino's",    "https://www.dominos.co.in/"),
            "dominos":    ("Domino's",    "https://www.dominos.co.in/"),
            "domino's":   ("Domino's",    "https://www.dominos.co.in/"),
            "2":          ("Pizza Hut",   "https://www.pizzahut.co.in/"),
            "pizza hut":  ("Pizza Hut",   "https://www.pizzahut.co.in/"),
            "3":          ("Zomato",      "https://www.zomato.com/"),
            "zomato":     ("Zomato",      "https://www.zomato.com/"),
            "4":          ("Swiggy",      "https://www.swiggy.com/"),
            "swiggy":     ("Swiggy",      "https://www.swiggy.com/"),
            "5":          ("McDonald's",  "https://www.mcdelivery.co.in/"),
            "mcdonalds":  ("McDonald's",  "https://www.mcdelivery.co.in/"),
            "mcdonald's": ("McDonald's",  "https://www.mcdelivery.co.in/"),
            "6":          ("EatSure",     "https://eatsure.com/"),
            "eatsure":    ("EatSure",     "https://eatsure.com/"),
        }
        return options_text, {"type": "food", "options": opts}
 
    # ── CAB BOOKING ───────────────────────────────────────────────────────────
    @staticmethod
    def handle_cab_booking(query: str) -> Tuple[str, Dict]:
        m    = re.search(r"(?:cab|uber|ola|taxi|ride|rapido)(?:\s+to)?\s+(.+?)(?:\s*$|\s+from\b)", query, re.I)
        dest = m.group(1).strip() if m else ""
 
        options_text = f"""## 🚗 Cab Booking{f' → *{dest}*' if dest else ''}
 
Which service, {USER_NAME}?
 
| # | Service | Notes |
|---|---------|-------|
| **1** | 🟡 **Ola** | Best in India, Ola Share |
| **2** | ⬛ **Uber** | Wide coverage, UPI/card |
| **3** | 🔵 **Rapido** | Bike taxi, cheapest |
| **4** | 🟢 **InDrive** | Negotiate fare yourself |
| **5** | 🗺  **Google Maps** | Compare all modes |
 
**Type 1–5** to proceed."""
 
        dest_enc = urllib.parse.quote(dest) if dest else ""
        opts: Dict[str, Tuple[str, str]] = {
            "1":          ("Ola",         "https://book.olacabs.com/"),
            "ola":        ("Ola",         "https://book.olacabs.com/"),
            "2":          ("Uber",        "https://m.uber.com/"),
            "uber":       ("Uber",        "https://m.uber.com/"),
            "3":          ("Rapido",      "https://rapido.bike/"),
            "rapido":     ("Rapido",      "https://rapido.bike/"),
            "4":          ("InDrive",     "https://indrive.com/"),
            "indrive":    ("InDrive",     "https://indrive.com/"),
            "5":          ("Google Maps", f"https://maps.google.com/?saddr=My+Location&daddr={dest_enc}"),
            "google maps":("Google Maps", f"https://maps.google.com/?saddr=My+Location&daddr={dest_enc}"),
        }
        return options_text, {"type": "cab", "dest": dest, "options": opts}
 
    # ── PROCESS CHOICE ────────────────────────────────────────────────────────
    def process_choice(self, user_input: str) -> str:
        task = self._pending
        if not task:
            return ""
 
        choice  = user_input.strip().lower()
        options = task.get("options", {})
 
        match: Optional[Tuple[str, str]] = options.get(choice)
        if not match:
            for k, v in options.items():
                if isinstance(k, str) and choice in k:
                    match = v
                    break
 
        if match and isinstance(match, tuple):
            name, url = match
            webbrowser.open(url)
            self.clear()
            return (
                f"✅ Opening **{name}** in your browser, {USER_NAME}!\n\n"
                f"🌐 `{url}`\n\n"
                f"*Complete your order there — JARVIS will be right here when you return.*"
            )
        self.clear()
        return "❌ Couldn't match your choice. Please type the number or platform name."
 
# ══════════════════════════════════════════════════════════════════════════════
#  SKILLS  (all deterministic tools)
# ══════════════════════════════════════════════════════════════════════════════
class Skills:
 
    # ── CALCULATOR ────────────────────────────────────────────────────────────
    @staticmethod
    def calculate(expr: str) -> str:
        safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe.update({"abs": abs, "round": round, "pow": pow,
                     "min": min, "max": max, "sum": sum,
                     "int": int, "float": float, "len": len})
        try:
            cleaned = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
            result  = eval(cleaned, {"__builtins__": {}}, safe)
            return f"**🧮 Result:** `{expr}` = **{result}**"
        except Exception as e:
            return f"❌ Cannot compute `{expr}`: {e}"
 
    # ── UNIT CONVERTER ────────────────────────────────────────────────────────
    @staticmethod
    def convert_units(value: float, from_u: str, to_u: str) -> str:
        f, t = from_u.lower().strip(), to_u.lower().strip()
        # table: (canonical_names, factor_to_base)
        length = {
            "m": 1, "meter": 1, "metres": 1, "meters": 1,
            "km": 1000, "kilometer": 1000, "kilometres": 1000, "kilometers": 1000,
            "cm": 0.01, "centimeter": 0.01,
            "mm": 0.001, "millimeter": 0.001,
            "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
            "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
            "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
            "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
        }
        weight = {
            "kg": 1, "kilogram": 1, "kilograms": 1,
            "g":  0.001, "gram": 0.001, "grams": 0.001,
            "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592,
            "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
            "t":  1000, "tonne": 1000, "tonnes": 1000,
        }
        temperature_conversions = {
            ("c", "f"):  lambda v: v * 9/5 + 32,
            ("f", "c"):  lambda v: (v - 32) * 5/9,
            ("c", "k"):  lambda v: v + 273.15,
            ("k", "c"):  lambda v: v - 273.15,
            ("f", "k"):  lambda v: (v - 32) * 5/9 + 273.15,
            ("k", "f"):  lambda v: (v - 273.15) * 9/5 + 32,
        }
        # temp aliases
        t_aliases = {
            "celsius": "c", "fahrenheit": "f", "kelvin": "k",
            "°c": "c", "°f": "f", "°k": "k",
        }
        fa, ta = t_aliases.get(f, f), t_aliases.get(t, t)
        if (fa, ta) in temperature_conversions:
            res = temperature_conversions[(fa, ta)](value)
            return f"🌡 {value} {from_u.title()} = **{res:.4f} {to_u.title()}**"
 
        for table in (length, weight):
            if f in table and t in table:
                base   = value * table[f]
                result = base / table[t]
                return f"📏 {value} {from_u} = **{result:.6g} {to_u}**"
 
        return f"❌ Cannot convert `{from_u}` → `{to_u}`. Supported: length, weight, temperature."
 
    # ── WEATHER ───────────────────────────────────────────────────────────────
    @staticmethod
    def get_weather(city: str) -> str:
        if not HAS_REQUESTS:
            return "❌ `requests` not installed."
        city = city.strip() or USER_CITY
        # Try OpenWeatherMap first if key available
        if OPENWEATHER_KEY:
            try:
                url = (f"https://api.openweathermap.org/data/2.5/weather"
                       f"?q={urllib.parse.quote(city)}&appid={OPENWEATHER_KEY}&units=metric")
                d = requests.get(url, timeout=8).json()
                w = d["weather"][0]["description"].title()
                m = d["main"]
                wind = d["wind"]["speed"]
                return (
                    f"## 🌍 Weather — {d['name']}, {d['sys']['country']}\n"
                    f"| Property | Value |\n|---|---|\n"
                    f"| 🌡 Temperature | **{m['temp']:.1f}°C** (feels {m['feels_like']:.1f}°C) |\n"
                    f"| ☁  Condition   | {w} |\n"
                    f"| 💧 Humidity    | {m['humidity']}% |\n"
                    f"| 💨 Wind        | {wind} m/s |\n"
                    f"| 🔼 High / Low  | {m['temp_max']:.1f}°C / {m['temp_min']:.1f}°C |"
                )
            except Exception:
                pass
        # Fallback: wttr.in (no key needed)
        try:
            r   = requests.get(f"https://wttr.in/{urllib.parse.quote(city)}?format=j1", timeout=8)
            d   = r.json()
            cur = d["current_condition"][0]
            area    = d["nearest_area"][0]["areaName"][0]["value"]
            country = d["nearest_area"][0]["country"][0]["value"]
            return (
                f"## 🌍 Weather — {area}, {country}\n"
                f"| Property | Value |\n|---|---|\n"
                f"| 🌡 Temperature | **{cur['temp_C']}°C** (feels {cur['FeelsLikeC']}°C) |\n"
                f"| ☁  Condition   | {cur['weatherDesc'][0]['value']} |\n"
                f"| 💧 Humidity    | {cur['humidity']}% |\n"
                f"| 💨 Wind        | {cur['windspeedKmph']} km/h |\n"
                f"| 👁 Visibility  | {cur['visibility']} km |"
            )
        except Exception as e:
            return f"❌ Weather fetch failed: {e}"
 
    # ── WIKIPEDIA ─────────────────────────────────────────────────────────────
    @staticmethod
    def search_wikipedia(query: str) -> str:
        if not HAS_WIKI:
            return "❌ Run: `pip install wikipedia`"
        try:
            wikipedia.set_lang("en")
            summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
            page    = wikipedia.page(query, auto_suggest=True)
            return (
                f"## 📖 {page.title}\n\n{summary}\n\n"
                f"🔗 [Read full article]({page.url})"
            )
        except wikipedia.exceptions.DisambiguationError as e:
            opts = ", ".join(e.options[:6])
            return f"⚠ Ambiguous — did you mean: **{opts}**?"
        except wikipedia.exceptions.PageError:
            return f"❌ No Wikipedia page found for: *{query}*"
        except Exception as e:
            return f"❌ Wikipedia error: {e}"
 
    # ── NEWS ──────────────────────────────────────────────────────────────────
    @staticmethod
    def get_news(topic: str = "world") -> str:
        if not HAS_REQUESTS:
            return "❌ `requests` not installed."
        feeds = {
            "world":      "https://feeds.bbci.co.uk/news/world/rss.xml",
            "technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "tech":       "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "science":    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "india":      "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
            "business":   "https://feeds.bbci.co.uk/news/business/rss.xml",
            "sport":      "https://feeds.bbci.co.uk/sport/rss.xml",
            "health":     "https://feeds.bbci.co.uk/news/health/rss.xml",
            "ai":         "https://feeds.bbci.co.uk/news/technology/rss.xml",
        }
        t   = topic.lower()
        url = next((v for k, v in feeds.items() if k in t), feeds["world"])
        try:
            r     = requests.get(url, timeout=8, headers={"User-Agent": "JARVIS/7.0"})
            items = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
            items = [i for i in items if i and "BBC" not in i][:8]
            if not items:
                items = re.findall(r"<title>(.*?)</title>", r.text)[1:9]
            lines = [f"## 📰 Headlines — {topic.title()}\n"]
            for i, h in enumerate(items, 1):
                lines.append(f"{i}. {h}")
            lines.append("\n*Source: BBC News RSS*")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ News fetch failed: {e}"
 
    # ── NOTES ─────────────────────────────────────────────────────────────────
    @staticmethod
    def take_note(content: str) -> str:
        ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"[{ts}] {content}\n"
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"✅ Note saved: *{content}*"
 
    @staticmethod
    def read_notes() -> str:
        p = Path(NOTES_FILE)
        if not p.exists() or p.stat().st_size == 0:
            return "📓 No notes yet. Try: **take a note: your text**"
        lines  = p.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-20:]
        return "## 📓 Notes (last 20)\n\n" + "\n".join(f"- {l}" for l in recent)
 
    # ── SNIPPET SAVER ─────────────────────────────────────────────────────────
    @staticmethod
    def save_snippet(content: str) -> str:
        p    = Path(SNIPPETS_FILE)
        data = json.loads(p.read_text()) if p.exists() else []
        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        idx  = len(data) + 1
        data.append({"id": idx, "ts": ts, "code": content})
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return f"✅ Snippet #{idx} saved ({len(content)} chars)"
 
    @staticmethod
    def list_snippets() -> str:
        p = Path(SNIPPETS_FILE)
        if not p.exists():
            return "📭 No snippets saved yet."
        data = json.loads(p.read_text())
        if not data:
            return "📭 No snippets saved yet."
        lines = ["## 💾 Saved Snippets\n"]
        for s in data[-10:]:
            preview = s["code"][:60].replace("\n", " ")
            lines.append(f"- `#{s['id']}` [{s['ts']}] `{preview}…`")
        return "\n".join(lines)
 
    # ── SCREENSHOT ────────────────────────────────────────────────────────────
    @staticmethod
    def take_screenshot() -> str:
        if not HAS_SCREENSHOT:
            return "❌ Run: `pip install Pillow`"
        try:
            fname = f"jarvis_screen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ImageGrab.grab().save(fname)
            return f"✅ Screenshot saved → **{fname}**"
        except Exception as e:
            return f"❌ Screenshot failed: {e}"
 
    # ── CLIPBOARD ─────────────────────────────────────────────────────────────
    @staticmethod
    def copy_to_clipboard(text: str) -> str:
        if not HAS_CLIP:
            return "❌ Run: `pip install pyperclip`"
        try:
            pyperclip.copy(text)
            preview = text[:80] + "…" if len(text) > 80 else text
            return f"✅ Copied to clipboard: *{preview}*"
        except Exception as e:
            return f"❌ Clipboard error: {e}"
 
    # ── SYSTEM INFO ───────────────────────────────────────────────────────────
    @staticmethod
    def system_info() -> str:
        lines = [
            "## 🖥  System Diagnostics",
            "| Component | Info |", "|---|---|",
            f"| OS        | {platform.system()} {platform.release()} ({platform.machine()}) |",
            f"| Python    | {platform.python_version()} |",
            f"| Processor | {platform.processor() or 'N/A'} |",
            f"| Hostname  | {socket.gethostname()} |",
            f"| Time      | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        ]
        if HAS_PSUTIL:
            cpu  = psutil.cpu_percent(interval=1)
            ram  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            lines += [
                f"| CPU Usage | **{cpu:.1f}%** |",
                f"| RAM       | {ram.used//1024**2} MB / {ram.total//1024**2} MB ({ram.percent:.1f}%) |",
                f"| Disk      | {disk.used//1024**3} GB / {disk.total//1024**3} GB ({disk.percent:.1f}%) |",
                f"| CPU Cores | {psutil.cpu_count()} logical |",
            ]
        try:
            ip = socket.gethostbyname(socket.gethostname())
            lines.append(f"| Local IP  | {ip} |")
        except Exception:
            pass
        return "\n".join(lines)
 
    # ── PYTHON SANDBOX ────────────────────────────────────────────────────────
    @staticmethod
    def run_python(code: str) -> str:
        buf = io.StringIO()
        err_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
                exec(code, {
                    "__builtins__": __builtins__,
                    "math": math, "datetime": datetime,
                    "random": random, "json": json,
                    "Path": Path, "os": os, "time": time,
                    **({"pd": pd}  if HAS_PANDAS else {}),
                    **({"np": np}  if HAS_NUMPY  else {}),
                    **({"plt": plt} if HAS_PLOT   else {}),
                })
            out = buf.getvalue()
            err = err_buf.getvalue()
            parts = []
            if out: parts.append(f"**✅ Output:**\n```\n{out}\n```")
            if err: parts.append(f"**⚠ Stderr:**\n```\n{err}\n```")
            return "\n".join(parts) if parts else "✅ Code executed (no output)."
        except Exception as e:
            return f"❌ **Error:** `{type(e).__name__}: {e}`"
 
    # ── OPEN TARGET ───────────────────────────────────────────────────────────
    @staticmethod
    def open_target(target: str) -> str:
        if re.search(r"https?://|www\.|\.com|\.org|\.net|\.io|\.ai|\.in", target):
            url = target if target.startswith("http") else "https://" + target
            webbrowser.open(url)
            return f"✅ Opening **{url}**"
        app_map = {
            "calculator":    {"Windows": "calc",            "Darwin": "open -a Calculator",  "Linux": "gnome-calculator"},
            "notepad":       {"Windows": "notepad",          "Darwin": "open -a TextEdit",    "Linux": "gedit"},
            "terminal":      {"Windows": "start cmd",        "Darwin": "open -a Terminal",    "Linux": "x-terminal-emulator"},
            "vscode":        {"Windows": "code",             "Darwin": "code",                "Linux": "code"},
            "browser":       {"Windows": "start chrome",     "Darwin": "open -a Google Chrome","Linux": "xdg-open https://google.com"},
            "spotify":       {"Windows": "start spotify",    "Darwin": "open -a Spotify",     "Linux": "spotify"},
            "file manager":  {"Windows": "explorer",         "Darwin": "open .",              "Linux": "nautilus ."},
            "task manager":  {"Windows": "taskmgr",          "Darwin": "open -a Activity Monitor","Linux": "gnome-system-monitor"},
        }
        sys_name = platform.system()
        for app, cmds in app_map.items():
            if app in target.lower():
                cmd = cmds.get(sys_name, "")
                if cmd:
                    os.system(cmd)
                    return f"✅ Launching **{app.title()}**"
        try:
            subprocess.Popen(target.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"✅ Launching **{target}**"
        except Exception as e:
            return f"❌ Could not open `{target}`: {e}"
 
    # ── TIME / DATE ───────────────────────────────────────────────────────────
    @staticmethod
    def time_date() -> str:
        n = datetime.datetime.now()
        return (
            f"## 🕐 Time & Date\n"
            f"- **Time:**    {n.strftime('%I:%M:%S %p')}\n"
            f"- **Date:**    {n.strftime('%A, %B %d, %Y')}\n"
            f"- **Week:**    Week {n.strftime('%W')} of {n.year}\n"
            f"- **Unix ts:** `{int(n.timestamp())}`"
        )
 
    # ── YOUTUBE ───────────────────────────────────────────────────────────────
    @staticmethod
    def youtube_search(query: str) -> str:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        webbrowser.open(url)
        return f"✅ Opened YouTube: **{query}**"
 
    # ── GOOGLE MAPS ───────────────────────────────────────────────────────────
    @staticmethod
    def maps_navigate(destination: str, origin: str = "") -> str:
        if origin:
            url = f"https://www.google.com/maps/dir/{urllib.parse.quote(origin)}/{urllib.parse.quote(destination)}"
        else:
            url = f"https://www.google.com/maps/search/{urllib.parse.quote(destination)}"
        webbrowser.open(url)
        return f"✅ Google Maps → **{destination}**" + (f" from {origin}" if origin else "")
 
    # ── STOCK PRICE ───────────────────────────────────────────────────────────
    @staticmethod
    def get_stock(symbol: str) -> str:
        if not HAS_REQUESTS:
            return "❌ `requests` not installed."
        sym = symbol.upper().strip()
        # Try Indian stock suffix if no dot
        suffixes = [sym, sym + ".NS", sym + ".BO"] if "." not in sym else [sym]
        for s in suffixes:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?interval=1d&range=1d"
                r   = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                d   = r.json()
                res = d["chart"]["result"]
                if not res:
                    continue
                meta   = res[0]["meta"]
                price  = meta.get("regularMarketPrice", "N/A")
                prev   = meta.get("previousClose", price)
                change = ((price - prev) / prev * 100) if isinstance(price, (int,float)) and prev else 0
                arrow  = "📈" if change >= 0 else "📉"
                curr   = meta.get("currency", "USD")
                return (
                    f"## {arrow} {s} Stock\n"
                    f"- **Price:**    {curr} {price:,.2f}\n"
                    f"- **Change:**   {change:+.2f}%\n"
                    f"- **Exchange:** {meta.get('exchangeName', 'N/A')}\n"
                    f"*via Yahoo Finance*"
                )
            except Exception:
                continue
        return f"❌ Could not fetch stock data for `{symbol}`"
 
    # ── CURRENCY CONVERSION ───────────────────────────────────────────────────
    @staticmethod
    def convert_currency(amount: float, from_c: str, to_c: str) -> str:
        if not HAS_REQUESTS:
            return "❌ `requests` not installed."
        try:
            url  = f"https://open.er-api.com/v6/latest/{from_c.upper()}"
            d    = requests.get(url, timeout=8).json()
            rate = d["rates"].get(to_c.upper())
            if not rate:
                return f"❌ Unknown currency code: {to_c.upper()}"
            result = amount * rate
            return (
                f"💱 **{amount:,.2f} {from_c.upper()}** = **{result:,.4f} {to_c.upper()}**\n"
                f"*Rate: 1 {from_c.upper()} = {rate:.6f} {to_c.upper()}*"
            )
        except Exception as e:
            return f"❌ Currency conversion failed: {e}"
 
    # ── EMAIL DRAFT ───────────────────────────────────────────────────────────
    @staticmethod
    def draft_email(to: str, subject: str = "", body: str = "") -> str:
        subj = subject or "Message from JARVIS"
        bod  = body or f"Hi,\n\n[Compose your message here]\n\nBest regards,\n{USER_NAME}"
        params = urllib.parse.urlencode({"to": to, "su": subj, "body": bod})
        url    = f"https://mail.google.com/mail/?view=cm&{params}"
        webbrowser.open(url)
        return f"✅ Gmail draft opened → **{to}**  |  Subject: *{subj}*"
 
    # ── WEB SEARCH ────────────────────────────────────────────────────────────
    @staticmethod
    def web_search(query: str, engine: str = "google") -> str:
        if engine == "ddg":
            url = "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
        else:
            url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        webbrowser.open(url)
        return f"✅ Opened {engine.title()} search: **{query}**"
 
    # ── IP GEOLOCATION ────────────────────────────────────────────────────────
    @staticmethod
    def ip_geolocate(ip: str) -> str:
        if not HAS_REQUESTS:
            return "❌ `requests` not installed."
        try:
            d = requests.get(f"https://ipapi.co/{ip}/json/", timeout=8).json()
            if "error" in d:
                return f"❌ {d.get('reason', 'Unknown error')}"
            return (
                f"## 🌍 IP Geolocation — `{ip}`\n"
                f"- **City:**     {d.get('city', 'N/A')}\n"
                f"- **Region:**   {d.get('region', 'N/A')}\n"
                f"- **Country:**  {d.get('country_name', 'N/A')}\n"
                f"- **ISP:**      {d.get('org', 'N/A')}\n"
                f"- **Timezone:** {d.get('timezone', 'N/A')}\n"
                f"- **Lat/Lon:**  {d.get('latitude', 'N/A')}, {d.get('longitude', 'N/A')}"
            )
        except Exception as e:
            return f"❌ IP lookup failed: {e}"
 
    # ── NETWORK INFO ──────────────────────────────────────────────────────────
    @staticmethod
    def network_info() -> str:
        lines = ["## 🌐 Network Diagnostics\n"]
        if HAS_REQUESTS:
            try:
                pub = requests.get("https://api.ipify.org", timeout=5).text.strip()
                lines.append(f"- **Public IP:** `{pub}`")
            except Exception:
                lines.append("- **Public IP:** unavailable")
        try:
            local = socket.gethostbyname(socket.gethostname())
            lines.append(f"- **Local IP:**  `{local}`")
            lines.append(f"- **Hostname:**  `{socket.gethostname()}`")
        except Exception:
            pass
        try:
            param = "-n" if platform.system() == "Windows" else "-c"
            ret = subprocess.call(
                ["ping", param, "1", "google.com"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
            )
            lines.append(f"- **Internet:** {'✅ Connected' if ret == 0 else '❌ Unreachable'}")
        except Exception:
            lines.append("- **Internet:** unknown")
        return "\n".join(lines)
 
    # ── FILE LIST ─────────────────────────────────────────────────────────────
    @staticmethod
    def list_files(directory: str = ".") -> str:
        try:
            p       = Path(directory).resolve()
            entries = sorted(p.iterdir())
            lines   = [f"## 📂 `{p}`\n"]
            for e in entries[:50]:
                icon = "📁" if e.is_dir() else "📄"
                size = f"{e.stat().st_size:,} bytes" if e.is_file() else ""
                lines.append(f"- {icon} `{e.name}` {size}")
            if len(list(p.iterdir())) > 50:
                lines.append("*…and more (showing first 50)*")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ {e}"
 
    # ── PLOT ──────────────────────────────────────────────────────────────────
    @staticmethod
    def plot(description: str) -> str:
        if not HAS_PLOT:
            return "❌ Run: `pip install matplotlib`"
        desc = description.lower()
        fig, ax = plt.subplots(figsize=(9, 5))
        x_dense = [i * 0.05 for i in range(-200, 201)]
 
        if "sin" in desc:
            y = [math.sin(v) for v in x_dense]
            ax.set_title("sin(x)", fontsize=14)
            ax.plot(x_dense, y, color="#00e5ff", linewidth=2, label="sin(x)")
        elif "cos" in desc:
            y = [math.cos(v) for v in x_dense]
            ax.set_title("cos(x)", fontsize=14)
            ax.plot(x_dense, y, color="#ff6600", linewidth=2, label="cos(x)")
        elif "tan" in desc:
            y = [math.tan(v) if abs(math.cos(v)) > 0.05 else None for v in x_dense]
            ax.set_title("tan(x)", fontsize=14)
            ax.plot(x_dense, y, color="#ffcc00", linewidth=2, label="tan(x)")
            ax.set_ylim(-10, 10)
        elif "log" in desc:
            xp = [v for v in x_dense if v > 0]
            y  = [math.log(v) for v in xp]
            ax.set_title("ln(x)", fontsize=14)
            ax.plot(xp, y, color="#00ff88", linewidth=2, label="ln(x)")
        else:
            x = list(range(-15, 16))
            y = [v**2 for v in x]
            ax.set_title("x²", fontsize=14)
            ax.plot(x, y, color="#ff69b4", linewidth=2, label="x²")
 
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#0a0a1a")
        ax.legend(facecolor="#0a0a1a", edgecolor="#00e5ff", labelcolor="white")
        fig.patch.set_facecolor("#0a0a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00e5ff")
 
        fname = "jarvis_plot.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        try:
            if   platform.system() == "Windows": os.startfile(fname)
            elif platform.system() == "Darwin":  subprocess.Popen(["open", fname])
            else:                                subprocess.Popen(["xdg-open", fname])
        except Exception:
            pass
        return f"✅ Plot saved as **{fname}** — opening now!"
 
    # ── ANALYZE FILE ──────────────────────────────────────────────────────────
    @staticmethod
    def analyze_file(filepath: str, client: Any) -> str:
        p = Path(filepath.strip().strip('"\''))
        if not p.exists():
            return f"❌ File not found: `{p}`"
        suffix = p.suffix.lower()
        try:
            # PDF
            if suffix == ".pdf":
                if HAS_PDF:
                    reader  = PyPDF2.PdfReader(str(p))
                    content = " ".join(
                        page.extract_text() or "" for page in reader.pages
                    )[:8000]
                else:
                    return "❌ Run: `pip install PyPDF2` for PDF analysis."
                prompt = f"Analyze this PDF '{p.name}' and give a structured summary with key insights:\n\n{content}"
 
            # Code / text
            elif suffix in (".txt",".py",".js",".ts",".java",".c",".cpp",
                            ".html",".css",".json",".xml",".md",".csv",
                            ".log",".sh",".go",".rs",".rb",".php"):
                content = p.read_text(encoding="utf-8", errors="replace")[:8000]
                prompt  = (
                    f"Analyze this `{suffix}` file named `{p.name}`. "
                    "If code: identify bugs, suggest improvements, rate quality. "
                    "If data/text: extract key insights and patterns. "
                    "Give actionable recommendations:\n\n" + content
                )
 
            # CSV
            elif suffix == ".csv" and HAS_PANDAS:
                df   = pd.read_csv(p)
                info = (
                    f"**Shape:** {df.shape[0]} rows × {df.shape[1]} cols\n"
                    f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
                    f"**Preview (5 rows):**\n```\n{df.head(5).to_string()}\n```\n\n"
                    f"**Stats:**\n```\n{df.describe().to_string()}\n```"
                )
                return f"## 📊 CSV Analysis — `{p.name}`\n\n{info}"
 
            # Image
            elif suffix in (".png",".jpg",".jpeg",".gif",".webp",".bmp"):
                img_data = base64.b64encode(p.read_bytes()).decode()
                media    = f"image/{suffix.lstrip('.')}"
                resp = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{media};base64,{img_data}"}},
                        {"type": "text",
                         "text": "Describe this image in detail. Extract visible text. Identify objects, people, diagrams, or data. Provide insights."}
                    ]}],
                    max_tokens=1000,
                )
                return f"## 🖼 Image Analysis — `{p.name}`\n\n" + resp.choices[0].message.content
 
            else:
                return f"⚠ File type `{suffix}` not supported for analysis."
 
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
            return f"## 📄 Analysis — `{p.name}`\n\n" + resp.choices[0].message.content
 
        except Exception as e:
            return f"❌ File analysis failed: {e}"
 
    # ── JOKE ──────────────────────────────────────────────────────────────────
    @staticmethod
    def joke() -> str:
        jokes = [
            "Why do programmers prefer dark mode? Because **light attracts bugs**. 🐛",
            "A SQL query walks into a bar and says: *'Can I JOIN you?'*",
            "Why did the AI go to therapy? Too many **deep issues**.",
            "There are 10 types of people: those who understand binary, and those who don't.",
            "Why don't scientists trust atoms? **They make up everything.**",
            "I would tell you a UDP joke — but you might not get it.",
            "Why did the ML model break up with the dataset? **Too many missing values.**",
            "What do you call a fish that knows machine learning? A **neural net-work!** 🐟",
            "Python developer's favourite band? **The Rolling Scopes.** 🎸",
            "Why was the JS dev sad? He didn't **Node** how to **Express** himself.",
            "Why did the developer go broke? He used up all his **cache**.",
            "I have a joke about recursion, but first let me tell you a joke about recursion.",
            "Why did the computer go to the doctor? It had a **virus** — and no antivirus.",
            "What's a programmer's favourite tea? **Java**. ☕",
            "How many programmers does it take to change a light bulb? **None** — that's a hardware problem.",
            "Why do Java programmers wear glasses? Because they don't **C#**.",
            "A byte walks into a bar. The bartender asks: 'What's wrong?' It replies: **'Bit error.'**",
            "An SEO expert walks into a bar, bars, pub, public house, tavern, brewery, saloon…",
            "I asked the AI to write a haiku. It returned a **404: Inspiration Not Found**.",
            "Debugging: being the detective in a crime movie **where you're also the murderer**.",
        ]
        return f"😄 {random.choice(jokes)}"
 
# ══════════════════════════════════════════════════════════════════════════════
#  INTENT PARSER  (deterministic routing — fast & zero-latency)
# ══════════════════════════════════════════════════════════════════════════════
def parse_intent(
    text:        str,
    reminders:   Optional["ReminderEngine"] = None,
    planner:     Optional["AgentPlanner"]   = None,
    plugins:     Optional["PluginManager"]  = None,
    groq_client: Any                        = None,
) -> Optional[str]:
    t = text.lower().strip()
    s = Skills()
 
    # ── PLUGINS (highest priority) ────────────────────────────────────────────
    if plugins:
        res = plugins.match(text)
        if res:
            return res
 
    # ── FOOD ORDERING ─────────────────────────────────────────────────────────
    # ── FOOD ORDERING ─────────────────────────────────────────────────────────
    food_triggers = [
        "order food", "order pizza", "order burger", "order biryani",
        "i'm hungry", "im hungry", "i want to eat", "order from",
        "food delivery", "order momos", "order chinese", "deliver food",
        "order noodles", "order rolls", "order kebab", "order sushi",
        "food ordering", "oroder pizza", "order", "food", "hungry"
    ]
    if any(p in t for p in food_triggers):
        if planner:
            msg, pending = AgentPlanner.handle_food_order(text)
            planner.set_pending(pending)
            return msg
 
    # ── CAB BOOKING ───────────────────────────────────────────────────────────
    cab_triggers = [
        "book a cab", "call a cab", "book uber", "book ola",
        "get me a cab", "i need a ride", "book rapido", "book taxi",
        "cab to", "ride to", "drop me", "book indrive",
    ]
    if any(p in t for p in cab_triggers):
        if planner:
            msg, pending = AgentPlanner.handle_cab_booking(text)
            planner.set_pending(pending)
            return msg
 
    # ── TIME / DATE ───────────────────────────────────────────────────────────
    if any(p in t for p in ["what time", "current time", "what's the time",
                              "what date", "today's date", "what day"]):
        return s.time_date()
 
    # ── CALCULATOR ────────────────────────────────────────────────────────────
    m = re.search(
        r"(?:calculate|compute|eval|solve|what(?:'s| is)\s+)?"
        r"([\d\s\+\-\*\/\.\^()\%sqrtpicossintan]+(?:\s+[\+\-\*\/\^]\s+[\d\s\+\-\*\/\.\^()]+)?)",
        t)
    if m and re.search(r"(?:calculat|comput|eval|what is \d|solve|\d[\+\-\*\/\^]\d)", t):
        return s.calculate(m.group(1).strip())
 
    # ── WEATHER ───────────────────────────────────────────────────────────────
    m = re.search(r"weather\s+(?:in|for|at|of)?\s*([a-zA-Z\s]{2,30})", t)
    if m and "weather" in t:
        return s.get_weather(m.group(1).strip())
    if "weather" in t:
        return s.get_weather(USER_CITY)
 
    # ── WIKIPEDIA ─────────────────────────────────────────────────────────────
    m = re.search(r"(?:who (?:is|was)|what (?:is|was)|tell me about|wikipedia|wiki)\s+(.+)", t)
    if m:
        return s.search_wikipedia(m.group(1).strip())
 
    # ── NEWS ──────────────────────────────────────────────────────────────────
    m = re.search(r"(?:news|headlines|top stories)(?:\s+(?:about|on|in|for))?\s*(.*)", t)
    if m and any(p in t for p in ["news", "headlines", "top stories"]):
        return s.get_news(m.group(1).strip() or "world")
 
    # ── YOUTUBE ───────────────────────────────────────────────────────────────
    m = re.search(r"(?:youtube|play on youtube|watch on youtube|search youtube)\s+(.+)", t)
    if m:
        return s.youtube_search(m.group(1).strip())
    m = re.search(r"play\s+(.+?)\s+(?:on youtube|youtube)", t)
    if m:
        return s.youtube_search(m.group(1).strip())
 
    # ── MAPS / NAVIGATE ───────────────────────────────────────────────────────
    m = re.search(r"(?:navigate|directions?|how to get|take me|route)\s+(?:to\s+)?(.+?)(?:\s+from\s+(.+))?$", t)
    if m and any(p in t for p in ["navigate","direction","how to get","take me","route to"]):
        return s.maps_navigate(m.group(1).strip(), m.group(2) or "")
    m = re.search(r"(?:open maps?|show on map)\s+(.+)", t)
    if m:
        return s.maps_navigate(m.group(1).strip())
 
    # ── STOCK ─────────────────────────────────────────────────────────────────
    m = re.search(r"(?:stock|share price|price of)\s+([a-zA-Z\.]{1,8})", t)
    if m and any(p in t for p in ["stock","share price","nasdaq","nse","bse","price of"]):
        return s.get_stock(m.group(1))
 
    # ── CURRENCY ──────────────────────────────────────────────────────────────
    m = re.search(r"(?:convert|exchange|how much is)\s+([\d\.]+)\s+([a-zA-Z]{3})\s+(?:to|in)\s+([a-zA-Z]{3})", t)
    if m:
        return s.convert_currency(float(m.group(1)), m.group(2), m.group(3))
 
    # ── UNIT CONVERSION ───────────────────────────────────────────────────────
    m = re.search(r"(?:convert|how (?:many|much) is)\s+([\d\.]+)\s+(\w+)\s+(?:to|in)\s+(\w+)", t)
    if m:
        # Try unit conversion first; falls back to currency check above
        res = s.convert_units(float(m.group(1)), m.group(2), m.group(3))
        if "❌" not in res:
            return res
 
    # ── EMAIL ─────────────────────────────────────────────────────────────────
    m = re.search(r"(?:send|draft|compose|write)\s+(?:an?\s+)?email\s+(?:to\s+)?(.+)", t)
    if m and "email" in t:
        return s.draft_email(m.group(1).strip())
 
    # ── NOTES ─────────────────────────────────────────────────────────────────
    m = re.search(r"(?:take a? note|note down|save note|add note|remember this)[:\-]?\s*(.+)", t)
    if m:
        return s.take_note(m.group(1).strip())
    if any(p in t for p in ["show notes","read notes","my notes","list notes","view notes"]):
        return s.read_notes()
 
    # ── SNIPPET ───────────────────────────────────────────────────────────────
    m = re.search(r"(?:save snippet|save code)[:\-]?\s*(.+)", t, re.DOTALL)
    if m:
        return s.save_snippet(m.group(1).strip())
    if any(p in t for p in ["list snippets","show snippets","my snippets"]):
        return s.list_snippets()
 
    # ── SCREENSHOT ────────────────────────────────────────────────────────────
    if any(p in t for p in ["take a screenshot","screenshot","capture screen","take screenshot"]):
        return s.take_screenshot()
 
    # ── CLIPBOARD ─────────────────────────────────────────────────────────────
    if any(p in t for p in ["copy that","copy to clipboard","copy last response"]):
        return "__COPY_LAST__"
 
    # ── IP GEOLOCATE ──────────────────────────────────────────────────────────
    m = re.search(r"(?:where is ip|locate ip|geolocate|ip location)\s+([\d\.a-fA-F:]+)", t)
    if m:
        return s.ip_geolocate(m.group(1).strip())
 
    # ── REMINDER ──────────────────────────────────────────────────────────────
    m = re.search(r"remind me in (\d+)\s*(second|minute|hour|sec|min|hr)s?\s+(?:to\s+)?(.+)", t)
    if m and reminders:
        n, unit, msg = int(m.group(1)), m.group(2), m.group(3)
        mult = {"second":1,"sec":1,"minute":60,"min":60,"hour":3600,"hr":3600}
        return reminders.add_reminder(n * mult.get(unit, 60), msg)
 
    if any(p in t for p in ["list reminders","show reminders","my reminders"]):
        return reminders.list_reminders() if reminders else "No reminder engine active."
 
    # ── ALARM ─────────────────────────────────────────────────────────────────
    m = re.search(r"set (?:an? )?alarm (?:for\s+)?(\d{1,2})[:\.](\d{2})\s*(am|pm)?", t)
    if m and reminders:
        hr, mn = int(m.group(1)), int(m.group(2))
        meridiem = (m.group(3) or "").lower()
        if meridiem == "pm" and hr != 12: hr += 12
        if meridiem == "am" and hr == 12: hr = 0
        return reminders.set_alarm(hr, mn)
 
    # ── TIMER ─────────────────────────────────────────────────────────────────
    m = re.search(r"(?:set a? timer|timer)\s+(?:for\s+)?(\d+)\s*(second|minute|sec|min|hour|hr)s?", t)
    if m and reminders:
        n, unit = int(m.group(1)), m.group(2)
        mult = {"second":1,"sec":1,"minute":60,"min":60,"hour":3600,"hr":3600}
        return reminders.set_timer(n * mult.get(unit, 1))
 
    # ── SYSTEM INFO ───────────────────────────────────────────────────────────
    if any(p in t for p in ["system info","cpu","ram","disk space","system status","diagnostics","hardware info"]):
        return s.system_info()
 
    # ── NETWORK ───────────────────────────────────────────────────────────────
    if any(p in t for p in ["network","my ip","internet speed","wifi info","ping"]):
        return s.network_info()
 
    # ── OPEN TARGET ───────────────────────────────────────────────────────────
    m = re.search(r"^open\s+(.+)", t)
    if m:
        return s.open_target(m.group(1).strip())
 
    # ── WEB SEARCH ────────────────────────────────────────────────────────────
    m = re.search(r"(?:google|search(?: the web| online| google)?(?: for)?)\s+(.+)", t)
    if m:
        return s.web_search(m.group(2).strip())
    m = re.search(r"(?:duckduckgo|ddg)\s+(.+)", t)
    if m:
        return s.web_search(m.group(1).strip(), engine="ddg")
 
    # ── ANALYZE FILE ──────────────────────────────────────────────────────────
    m = re.search(r"(?:analyze|read|parse|look at|inspect)\s+(?:file\s+)?[\"']?([^\s\"']+\.[a-zA-Z]{1,5})[\"']?", t)
    if m and groq_client:
        return s.analyze_file(m.group(1), groq_client)
 
    # ── JOKE ──────────────────────────────────────────────────────────────────
    if any(p in t for p in ["tell me a joke","joke","make me laugh","say something funny"]):
        return s.joke()
 
    # ── LIST FILES ────────────────────────────────────────────────────────────
    m = re.search(r"list files(?: in)?\s*(.*)", t)
    if m:
        return s.list_files(m.group(1).strip() or ".")
 
    # ── PLOT ──────────────────────────────────────────────────────────────────
    if any(p in t for p in ["plot","graph","draw a chart","visualize","visualise"]):
        return s.plot(t)
 
    # ── RUN PYTHON ────────────────────────────────────────────────────────────
    m = re.search(r"^run\s+(.+)", t, re.DOTALL)
    if m:
        return Skills.run_python(m.group(1))
 
    return None   # → forward to Groq LLM
 
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN JARVIS CLASS
# ══════════════════════════════════════════════════════════════════════════════
class Jarvis:
    def __init__(self, voice_mode: bool = False):
        self._validate_key()
        self.client        = Groq(api_key=GROQ_API_KEY)
        self.memory        = Memory(groq_client=self.client)
        self.voice         = VoiceEngine()
        self.skills        = Skills()
        self.reminders     = ReminderEngine(self.voice)
        self.planner       = AgentPlanner()
        self.plugins       = PluginManager()
        self.last_response = ""
        self.wake_active   = False
        self._voice_mode   = voice_mode
 
    @staticmethod
    def _validate_key() -> None:
        if not GROQ_API_KEY:
            msg = textwrap.dedent("""
                ❌  GROQ API KEY NOT SET
 
                Steps to fix:
                  1. pip install python-dotenv
                  2. Create a file named  .env  in this folder
                  3. Add this line:  GROQ_API_KEY=your_key_here
                  4. Get a FREE key at: https://console.groq.com
 
                ⚠  NEVER commit your .env to GitHub!
            """).strip()
            if HAS_RICH:
                console.print(Panel(msg, border_style="red", title="[red]Setup Required[/red]"))
            else:
                print(msg)
            sys.exit(1)
 
    def ask_groq(self, user_input: str) -> str:
        """Stream a Groq LLM response with live output."""
        self.memory.add("user", user_input)
        messages = self.memory.get_messages()
        full = ""
 
        if HAS_RICH:
            console.print(f"\n[bold cyan]◈  J.A.R.V.I.S.[/bold cyan]  ", end="")
        try:
            stream = self.client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = messages,
                max_tokens  = 2048,
                temperature = 0.72,
                stream      = True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
                print(delta, end="", flush=True)
            print()
        except Exception as e:
            full = f"❌ Groq API error: {e}\n\nCheck your API key and internet connection."
            cprint(full, "red")
 
        self.memory.add("assistant", full)
        return full
 
    def respond(self, user_input: str) -> None:
        show_user(user_input)
 
        # ── COPY LAST ─────────────────────────────────────────────────────────
        if user_input.lower().strip() in ("copy that", "copy to clipboard", "copy last response"):
            result = (Skills.copy_to_clipboard(self.last_response)
                      if self.last_response else "Nothing to copy yet.")
            show_jarvis(result)
            self.voice.speak(result)
            return
 
        # ── PENDING AGENTIC TASK ──────────────────────────────────────────────
        if self.planner.is_awaiting():
            result = self.planner.process_choice(user_input)
            if result:
                show_jarvis(result)
                self.voice.speak(result)
                self.last_response = result
                return
 
        # ── SKILL ROUTING ─────────────────────────────────────────────────────
        skill_result = parse_intent(
            user_input,
            reminders   = self.reminders,
            planner     = self.planner,
            plugins     = self.plugins,
            groq_client = self.client,
        )
 
        if skill_result == "__COPY_LAST__":
            r = Skills.copy_to_clipboard(self.last_response) if self.last_response else "Nothing to copy."
            show_jarvis(r)
            self.voice.speak(r)
            self.last_response = r
        elif skill_result:
            show_jarvis(skill_result)
            self.voice.speak(skill_result)
            self.last_response = skill_result
            self.memory.add("user",      user_input)
            self.memory.add("assistant", skill_result)
        else:
            response = self.ask_groq(user_input)
            self.voice.speak(response)
            self.last_response = response
 
    def greet(self) -> None:
        hour = datetime.datetime.now().hour
        tod  = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"
        self.memory.session_count += 1
        self.memory.save()
        plugin_str = f"  **{len(self.plugins.plugins)} plugin(s)** loaded." if self.plugins.plugins else ""
        msg = (
            f"Good {tod}, {USER_NAME}. **J.A.R.V.I.S. v{VERSION}** is fully online — all systems nominal.\n\n"
            f"Session **#{self.memory.session_count}** · Model: `{GROQ_MODEL}`.{plugin_str}\n\n"
            f"**v7.0 features:** True agentic tool-calling 🤖 · IP geolocation 🌍 · "
            f"Unit converter 📏 · Snippet saver 💾 · Timer ⏱ · Auto context-compression 🧠\n\n"
            f"How may I assist you today, {USER_NAME}?"
        )
        show_jarvis(msg)
        self.voice.speak(f"Good {tod}, {USER_NAME}. JARVIS version 7 is online and ready.")
        if MISSING:
            cprint(
                f"\n⚠  Optional packages missing (some features disabled):\n"
                f"   pip install {' '.join(MISSING)}\n",
                "yellow"
            )
 
    def _show_help(self) -> None:
        show_jarvis(
            f"## 🧠 J.A.R.V.I.S. v{VERSION} — All Commands\n\n"
            "| Command | Action |\n|---|---|\n"
            "| `v` | 🎙 Voice input |\n"
            "| `wake` | 👂 Toggle always-on wake-word |\n"
            "| `order pizza` | 🍕 Food delivery |\n"
            "| `book a cab to <place>` | 🚗 Cab booking |\n"
            "| `navigate to <place>` | 🗺 Google Maps |\n"
            "| `play <song> on youtube` | ▶️ YouTube |\n"
            "| `stock RELIANCE` | 📈 Live stock |\n"
            "| `convert 100 USD to INR` | 💱 Currency |\n"
            "| `convert 5 km to miles` | 📏 Unit converter |\n"
            "| `send email to <addr>` | 📧 Gmail draft |\n"
            "| `analyze <file>` | 🔍 AI file analysis |\n"
            "| `news [topic]` | 📰 Headlines |\n"
            "| `take a note: <text>` | 📓 Save note |\n"
            "| `save snippet: <code>` | 💾 Save snippet |\n"
            "| `remind me in 5 min to…` | ⏰ Reminder |\n"
            "| `set alarm for 7:30 am` | 🔔 Alarm |\n"
            "| `set a timer for 90 sec` | ⏱ Timer |\n"
            "| `screenshot` | 📷 Capture screen |\n"
            "| `where is IP 8.8.8.8` | 🌍 IP geolocation |\n"
            "| `copy that` | 📋 Copy last response |\n"
            "| `export chat` | 💾 Save conversation |\n"
            "| `who is <person>` | 📖 Wikipedia |\n"
            "| `weather in <city>` | 🌤 Weather |\n"
            "| `plot sin / cos / log` | 📊 Graph |\n"
            "| `run <python code>` | 🐍 Python sandbox |\n"
            "| `sysinfo` | 🖥 System diagnostics |\n"
            "| `network` | 🌐 Network info |\n"
            "| `memory` | 🧠 Memory stats |\n"
            "| `plugins` | 🔌 List plugins |\n"
            "| `remember <fact>` | 💡 Long-term fact |\n"
            "| `clear` | 🗑 Clear conversation |\n"
            "| `quit` | ⛔ Shutdown JARVIS |\n\n"
            "**Or just speak naturally — I understand everything! 🤖**"
        )
 
    def run(self) -> None:
        banner()
        self.greet()
 
        if self._voice_mode:
            self.wake_active = True
            self.voice.start_wake_word_listener(self.respond)
 
        while not _shutdown_flag.is_set():
            try:
                if HAS_RICH:
                    console.print("\n[bold blue]▶  YOU:[/bold blue] ", end="")
                raw = input("").strip()
 
                if not raw:
                    continue
 
                cmd = raw.lower().strip()
 
                # ── META COMMANDS ──────────────────────────────────────────────
                if cmd in ("quit","exit","shutdown","bye","goodbye","stop","power off"):
                    self.voice.stop_wake_word_listener()
                    show_jarvis(f"All systems powering down. Goodbye, {USER_NAME}. Stay brilliant. 👋")
                    self.voice.speak(f"Goodbye, {USER_NAME}. Powering down.")
                    break
 
                elif cmd == "clear":
                    self.memory.clear_session()
                    os.system("cls" if platform.system() == "Windows" else "clear")
                    banner()
                    cprint("✅ Conversation cleared.", "green")
 
                elif cmd == "sysinfo":
                    show_jarvis(Skills.system_info())
 
                elif cmd == "network":
                    show_jarvis(Skills.network_info())
 
                elif cmd == "memory":
                    turns = len(self.memory.history) // 2
                    show_jarvis(
                        f"## 🧠 Memory Status\n"
                        f"- Conversation turns: **{turns}**\n"
                        f"- Long-term facts:    **{len(self.memory.facts)}**\n"
                        f"- Session count:      **{self.memory.session_count}**\n"
                        f"- Memory file: `{MEMORY_FILE}`\n"
                        f"- Summarise at: **{SUMMARISE_AT}** turns"
                    )
 
                elif cmd in ("v", "voice", "listen"):
                    spoken = self.voice.listen()
                    if spoken:
                        self.respond(spoken)
 
                elif cmd == "wake":
                    if self.wake_active:
                        self.voice.stop_wake_word_listener()
                        self.wake_active = False
                        cprint('👂  Wake-word listener stopped.', "yellow")
                    else:
                        self.wake_active = True
                        self.voice.start_wake_word_listener(self.respond)
                        cprint(f'👂  Wake-word active — say "{WAKE_WORD}"!', "cyan")
 
                elif cmd == "export chat":
                    fname = self.memory.export_chat()
                    show_jarvis(f"✅ Chat exported → **{fname}**")
 
                elif cmd.startswith("remember "):
                    fact = raw[9:].strip()
                    key  = f"fact_{len(self.memory.facts)+1}"
                    self.memory.remember(key, fact)
                    show_jarvis(f"✅ Saved to long-term memory: *{fact}*")
 
                elif cmd == "plugins":
                    if self.plugins.plugins:
                        names = "\n".join(f"- `{p.__name__}`" for p in self.plugins.plugins)
                        show_jarvis(f"## 🔌 Loaded Plugins\n\n{names}")
                    else:
                        show_jarvis(
                            "No plugins loaded.\n\n"
                            "Drop a `.py` file in the `plugins/` folder with:\n"
                            "```python\nTRIGGER_WORDS = ['keyword']\ndef run(query: str) -> str: ...\n```"
                        )
 
                elif cmd in ("help", "?", "commands"):
                    self._show_help()
 
                else:
                    self.respond(raw)
 
            except KeyboardInterrupt:
                print()
                _shutdown_flag.set()
                self.voice.stop_wake_word_listener()
                show_jarvis(f"Interrupt detected. Goodbye, {USER_NAME}. 👋")
                break
            except EOFError:
                break
            except Exception as e:
                cprint(f"⚠ Unexpected error: {e}", "red")
 
# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"J.A.R.V.I.S. v{VERSION} — Agentic AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python jarvis.py               # text mode
          python jarvis.py --voice       # start with always-on wake word
          python jarvis.py --name Boss   # change user name
          python jarvis.py --city Mumbai # change default city
        """),
    )
    parser.add_argument("--voice",  action="store_true",
                        help="Start in always-on wake-word mode")
    parser.add_argument("--name",   type=str, default=None,
                        help="Override user name (default: Sir)")
    parser.add_argument("--city",   type=str, default=None,
                        help="Override default city (default: Delhi)")
    parser.add_argument("--model",  type=str, default=None,
                        help="Override Groq model")
    args = parser.parse_args()
 
    if args.name:  USER_NAME  = args.name
    if args.city:  USER_CITY  = args.city
    if args.model: GROQ_MODEL = args.model
 
    Jarvis(voice_mode=args.voice).run()
 