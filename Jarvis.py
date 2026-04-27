import os
import time
import tempfile
import torch
import pyttsx3
import whisper
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image, ImageDraw, ImageFont
import textwrap

# -----------------------------
# CONFIG - Upgraded to Phi-3 (Much smarter than DistilGPT2)
# -----------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" 
WHISPER_MODEL = "tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODELS
# -----------------------------
print(f"🚀 Loading Elite Brain ({MODEL_NAME})...")
# Using pipeline for faster and better inference
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float32 if device == "cpu" else torch.float16, "trust_remote_code": True},
    device_map="auto" if device == "cuda" else None
)

print("🎤 Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL)

# -----------------------------
# UPGRADED CHAT LOGIC (With Short-Term Memory)
# -----------------------------
messages = [
    {"role": "system", "content": "You are Jarvis, a highly intelligent AI built by Pranava Bhatia. You are helpful, witty, and concise."}
]

def generate_reply(text):
    global messages
    messages.append({"role": "user", "content": text})
    
    # Keeping only last 5 exchanges to save memory
    context = messages[-6:] 
    
    output = pipe(context, max_new_tokens=150, temperature=0.7, do_sample=True)
    reply = output[0]['generated_text'][-1]['content']
    
    messages.append({"role": "assistant", "content": reply})
    return reply

# -----------------------------
# TTS & IMAGE GEN (Optimized)
# -----------------------------
def speak_to_file(text):
    tmp_path = os.path.join(tempfile.gettempdir(), f"jarvis_{int(time.time()*1000)}.wav")
    tts = pyttsx3.init()
    tts.save_to_file(text, tmp_path)
    tts.runAndWait()
    return tmp_path

def make_text_image(text):
    img = Image.new("RGB", (800, 400), (10, 10, 25)) # Deep Blue theme
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    wrapped = textwrap.fill(text, width=50)
    draw.text((40, 60), f"JARVIS SAYS:\n\n{wrapped}", fill=(0, 255, 255), font=font)
    img_path = os.path.join(tempfile.gettempdir(), f"jarvis_ui_{int(time.time()*1000)}.png")
    img.save(img_path)
    return img_path

# -----------------------------
# GRADIO PIPELINE
# -----------------------------
def jarvis_pipeline(audio):
    if not audio: return "No voice detected.", None, None
    
    try:
        # Transcription
        res = whisper_model.transcribe(audio)
        user_input = res["text"].strip()
        if not user_input: return "Speak clearly, sir.", None, None

        # Logic
        reply = generate_reply(user_input)
        return reply, speak_to_file(reply), make_text_image(reply)
    except Exception as e:
        return f"System Error: {e}", None, None

# -----------------------------
# UI INTERFACE
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 JARVIS: Advanced Multimodal AI")
    gr.Markdown("Created by **Pranava Bhatia** | Class 12 AI Innovator")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Command Jarvis")
            submit_btn = gr.Button("Send Voice Command", variant="primary")
        
        with gr.Column():
            text_out = gr.Textbox(label="Jarvis Response")
            audio_out = gr.Audio(label="Voice Output", autoplay=True)
            img_out = gr.Image(label="System Dashboard")

    submit_btn.click(jarvis_pipeline, inputs=audio_input, outputs=[text_out, audio_out, img_out])

if __name__ == "__main__":
    demo.launch()