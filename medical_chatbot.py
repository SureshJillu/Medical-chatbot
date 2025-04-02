import os
import logging
import ollama
import pyttsx3
import numpy as np
import sounddevice as sd
import wave

import speech_recognition as sr
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from biobert import get_biobert_answer  # Import BioBERT function

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Text-to-Speech (TTS)
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)  # Set speech speed

# Alternative Speech Recognition using sounddevice (instead of PyAudio)
def get_speech_input_alternative(duration=5, fs=16000):
    """
    Records audio for a given duration using sounddevice,
    saves it to a temporary WAV file, and uses SpeechRecognition to transcribe.
    """
    logging.info(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    temp_filename = "temp_audio.wav"
    
    # Save recording to WAV file
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio = 2 bytes per sample
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    
    # Use SpeechRecognition to transcribe the audio file
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_filename) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Sorry, I couldn't understand that."
    except sr.RequestError:
        text = "Speech Recognition service is unavailable."
    
    # Remove the temporary file
    os.remove(temp_filename)
    return text

def speak(text):
    """Convert text to speech."""
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        logging.error(f"TTS error: {e}")

def query_llm(prompt):
    """Query the DeepSeek LLM using Ollama."""
    try:
        response = ollama.chat(model="deepseek-r1:latest", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"LLM query error: {e}")
        return "Error querying the language model."

def split_paragraphs(raw_text):
    """Split large text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Optimal for medical text
        chunk_overlap=200,    # Ensures continuity between chunks
        separators=["\n\n", "\n", ". "]
    )
    return text_splitter.split_text(raw_text)

def load_pdfs(pdf_folder):
    """Extract text from all PDFs in the given folder."""
    text_chunks = []
    try:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    raw_text = page.extract_text()
                    if raw_text:
                        text_chunks.extend(split_paragraphs(raw_text))
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {e}")
    except Exception as e:
        logging.error(f"Error loading PDFs from folder {pdf_folder}: {e}")
    return text_chunks

def create_vector_store(pdf_folder):
    """Create a FAISS vector store from PDF text chunks."""
    text_chunks = load_pdfs(pdf_folder)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    return FAISS.from_texts(text_chunks, embeddings)

# Define the path for vector store persistence
VECTOR_STORE_PATH = "./medical-vectorstore"

# Initialize FAISS vector store (load existing or create new)
if os.path.exists(VECTOR_STORE_PATH):
    logging.info("Loading existing FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    logging.info("Creating new FAISS vector store...")
    vector_store = create_vector_store("medical_pdfs")
    vector_store.save_local(VECTOR_STORE_PATH)

def extract_relevant_medical_text(query):
    """Fetch relevant medical context from the vector store based on the query."""
    try:
        docs = vector_store.similarity_search(query, k=3)
        unique_docs = list(set([doc.page_content.strip() for doc in docs]))
        logging.info("Retrieved medical context:")
        for doc in unique_docs:
            logging.info(doc[:300])
        return " ".join(unique_docs).replace("\n", " ").strip() or "No relevant medical context found."
    except Exception as e:
        logging.error(f"Error extracting medical text: {e}")
        return "No relevant medical context found."

# Use ConversationBufferMemory to manage conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def is_medical_query(query, conversation_history=""):
    """
    Determines if the query is medical-related.
    It checks the current query for keywords, and if not found,
    it checks if the conversation history already indicates a medical context.
    """
    medical_keywords = ["disease", "diagnosis", "symptom", "treatment", "medicine", "health", "infection", "diabetes", "cure"]
    query_lower = query.lower()
    
    # Check the current query
    if any(keyword in query_lower for keyword in medical_keywords):
        return True
    
    # Check conversation history if provided
    if conversation_history:
        history_lower = conversation_history.lower()
        if any(keyword in history_lower for keyword in medical_keywords):
            return True
    
    return False

def get_response(user_question):
    """
    Get a chatbot response by checking if the query is medical-related.
    Uses conversation history to decide context.
    """
    past_conversation = memory.load_memory_variables({}).get("chat_history", [])
    conversation_history = "\n".join([msg.content for msg in past_conversation]) if past_conversation else ""
    
    # Determine if the query should be processed as medical
    if is_medical_query(user_question, conversation_history):
        # Process as a medical query
        context = extract_relevant_medical_text(user_question)
        biobert_fact = get_biobert_answer(context, user_question)
        prompt = f"""
You are an AI Medical Chatbot Assistant.
If the user query is medical-related, refer to the extracted BioBERT fact.
If it's non-medical, respond normally.

Recent Chat History:
{conversation_history if conversation_history else "No prior conversation."}

Medical Fact (from BioBERT): {biobert_fact}
User Query: {user_question}
        """
    else:
        # Process as a general query
        prompt = f"""
You are an AI Chatbot Assistant.
Recent Chat History:
{conversation_history if conversation_history else "No prior conversation."}
User Query: {user_question}
        """
    logging.info("Querying LLM with prompt...")
    response = query_llm(prompt)

    # Update conversation memory with the user query and the bot's response
    memory.save_context({"input": user_question}, {"output": response})
    return clean_response(response)



import re

def clean_response(response: str) -> str:
    """Remove any chain-of-thought text between <think> and </think> tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return cleaned





if __name__ == "__main__":
    logging.info("Starting AI Medical Chatbot (CLI mode)...")
    print("\n🩺 Welcome to the AI Medical Chatbot")
    print("Type your query or type 'voice' to use speech input. Type 'exit' to quit.")

    while True:
        user_input = input("\n📝 Enter your query (or 'voice' for speech input): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            logging.info("Exiting chatbot.")
            print("Goodbye!")
            speak("Goodbye! Have a great day.")
            break
        elif user_input.lower() == "voice":
            # Use the alternative speech input function
            user_input = get_speech_input_alternative()
            print(f"You (voice): {user_input}")

        bot_response = get_response(user_input)
        print(f"Bot: {bot_response}")
        speak(bot_response)

