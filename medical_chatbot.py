import ollama
import os
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize DeepSeek LLM (using Ollama)
def query_llm(prompt):
    response = ollama.chat(model="deepseek-r1:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Function to split large medical text into smaller chunks
def split_paragraphs(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Better for medical context
        chunk_overlap=200,  # Ensures related text stays together
        separators=["\n\n", "\n", ". "]
    )
    return text_splitter.split_text(raw_text)

# Function to extract text from medical PDFs
def load_pdfs(pdf_folder):
    text_chunks = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        reader = PdfReader(pdf_path)
        
        for page in reader.pages:
            raw_text = page.extract_text()
            if raw_text:
                chunks = split_paragraphs(raw_text)
                text_chunks.extend(chunks)
    
    return text_chunks

# Load and create FAISS vector store
def create_vector_store(pdf_folder):
    text_chunks = load_pdfs(pdf_folder)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5")  # Better for retrieval

    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

VECTOR_STORE_PATH = "./medical-vectorstore"

if os.path.exists(VECTOR_STORE_PATH):
    print("Loading existing FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5")

    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS vector store...")
    vector_store = create_vector_store("medical_pdfs")  # Folder where PDFs are stored
    vector_store.save_local(VECTOR_STORE_PATH)

# Function to fetch relevant medical context
def extract_relevant_medical_text(query):
    docs = vector_store.similarity_search(query, k=3)  # Retrieve 3 most relevant chunks
    filtered_docs = filter_relevant_docs(docs)  # Remove irrelevant results

    print("\n🔍 Retrieved medical context:")
    for doc in filtered_docs:
        print(doc.page_content[:300])  # Print first 300 characters for debugging

    return "\n".join([doc.page_content for doc in filtered_docs]) if filtered_docs else "No relevant medical context found."

# Function to filter out non-medical results
def filter_relevant_docs(docs):
    medical_keywords = ["disease", "infection", "treatment", "diagnosis", "medicine"]
    return [doc for doc in docs if any(word in doc.page_content.lower() for word in medical_keywords)]

# Memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chat history tracking
chat_history = []

def get_response(user_question):
    global chat_history

    # Retrieve only relevant medical context
    context = extract_relevant_medical_text(user_question)

    # Append user query to chat history
    chat_history.append(f"User: {user_question}")

    # Trim chat history (keep last 10 exchanges for context)
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # Generate prompt
    prompt = f"""
You are an AI Medical Chatbot Assistant. 
Look at the user query. If it's a medical query, use the provided medical context. 
If not, generate a response without even looking at the medical context.

**Recent Chat History:** 
{chr(10).join(chat_history)}

**Medical Context:** {context}

**User Query:** {user_question}
"""

    response = query_llm(prompt)

    chat_history.append(f"Bot: {response}")
    return response

def get_response_from_medical_chatbot(user_question):
    return get_response(user_question)  # Ensures compatibility with UI


# Interactive chatbot loop
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        bot_response = get_response(user_input)
        print(f"Bot: {bot_response}")
