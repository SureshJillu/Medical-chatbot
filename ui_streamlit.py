import streamlit as st
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import tempfile
import os
import re
import time
import uuid
from gtts import gTTS
from medical_chatbot import get_response  # Your backend function

# Configure page
st.set_page_config(page_title="AI Medical Chatbot", layout="wide")

# Sidebar with heading/instructions
with st.sidebar:
    st.title("AI Medical Chatbot")
    st.markdown("""
    **Instructions:**
    - Type your query in the box or click the mic (🎤) to record.
    - When recording, the mic button shows "⏹ Stop".
    - Each assistant response will include a speaker (🔊) button; click it to hear that response.
    """)

# Inject minimal CSS for message alignment (user right, assistant left)
st.markdown(
    """
    <style>
    [data-testid="stChatMessage-user"] {
        text-align: right !important;
        margin-left: auto !important;
    }
    [data-testid="stChatMessage-assistant"] {
        text-align: left !important;
        margin-right: auto !important;
    }
    .stChatMessage {
        max-width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for messages and recording
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of dicts: {"role": "user"/"assistant", "content": str, "audio_file": optional str}
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "record_start" not in st.session_state:
    st.session_state.record_start = None

# --------------------
# AUDIO HELPERS
# --------------------
def start_recording(fs=16000, max_duration=60):
    st.session_state.audio_data = sd.rec(int(max_duration * fs), samplerate=fs, channels=1, dtype="int16")
    st.session_state.record_start = time.time()
    st.session_state.recording = True

def stop_recording(fs=16000):
    sd.stop()
    elapsed = time.time() - st.session_state.record_start
    samples = int(elapsed * fs)
    data = st.session_state.audio_data[:samples]
    st.session_state.recording = False
    st.session_state.audio_data = None
    st.session_state.record_start = None
    return data

def save_audio(data, fs=16000):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, data, fs)
    return temp_file.name

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        text = "Speech recognition service is unavailable."
    os.remove(file_path)
    return text

# --------------------
# TTS HELPER
# --------------------
def generate_tts(text, filename):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

# --------------------
# RESPONSE CLEANING
# --------------------
def clean_response(response: str) -> str:
    # Remove any chain-of-thought markers
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Reformat medical fact if present
    if "Medical Fact (from BioBERT):" in response:
        parts = response.split("Medical Fact (from BioBERT):")
        main = parts[0].strip()
        fact = parts[1].strip()
        response = f"{main}\n\n**Medical Fact:** {fact}"
    return response

# --------------------
# CONVERSATION DISPLAY
# --------------------
def display_conversation():
    st.markdown("### Conversation")
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
            # Display a speaker button only if audio_file exists for this message.
            if "audio_file" in msg and os.path.exists(msg["audio_file"]):
                if st.button("🔊", key=f"speaker_{idx}"):
                    with open(msg["audio_file"], "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")

# --------------------
# MESSAGE PROCESSING
# --------------------
def process_message(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)
    with st.spinner("Generating response..."):
        resp = get_response(user_text)
    resp = clean_response(resp)
    tts_filename = f"tts_{uuid.uuid4().hex}.mp3"
    generate_tts(resp, tts_filename)
    st.session_state.messages.append({"role": "assistant", "content": resp, "audio_file": tts_filename})
    st.chat_message("assistant").write(resp)

# --------------------
# LAYOUT: CONVERSATION & INPUT
# --------------------
display_conversation()
cols = st.columns([0.85, 0.15])
with cols[0]:
    # The text input is placed on the left
    user_input = st.chat_input("Type your message here...", key="chat_input")
with cols[1]:
    # The mic button is placed on the right; its label toggles based on recording state.
    mic_label = "⏹ Stop" if st.session_state.recording else "🎤"
    mic_pressed = st.button(mic_label, key="mic_button")

# Process text input if provided
if user_input:
    process_message(user_input)

# Process mic button press for toggle behavior
if mic_pressed:
    if not st.session_state.recording:
        # Start recording if not currently recording.
        start_recording()
        try:
            st.experimental_rerun()  # Update button label
        except Exception:
            pass
    else:
        # Stop recording, transcribe, and process the voice input.
        audio_data = stop_recording()
        temp_file = save_audio(audio_data)
        transcribed = transcribe_audio(temp_file)
        if transcribed in ["Sorry, I couldn't understand the audio.", "Speech recognition service is unavailable."]:
            st.chat_message("assistant").write("Voice input not recognized. Please try again or type your message.")
        else:
            st.session_state.messages.append({"role": "user", "content": transcribed})
            st.chat_message("user").write(f"*Voice Input:* {transcribed}")
            process_message(transcribed)
        try:
            st.experimental_rerun()
        except Exception:
            pass