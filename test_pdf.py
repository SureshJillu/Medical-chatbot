from pdf_processor import extract_text_from_pdfs
from pdf_processor import chunk_text
import pickle


pdf_folder = "medical_pdfs"  # Change this to the actual folder where your PDFs are stored
text = extract_text_from_pdfs(pdf_folder)

print("Extracted text preview:\n", text[:500])  # Show first 500 characters

# Import necessary libraries
from pdf_processor import extract_text_from_pdf, chunk_text

# Extract text
text = extract_text_from_pdf("medical_pdfs/medical-book-1.pdf")  # Replace with actual PDF name


# 🔹 Apply better filtering here
def clean_medical_text(text):
    """
    Removes metadata, disclaimers, and unwanted content.
    """
    lines = text.split("\n")
    filtered_lines = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ["editor", "copyright", "disclaimer", "gale group"]):
            continue  # Skip metadata and disclaimers
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()

clean_text = clean_medical_text(text)  # 🔹 Apply filtering

# Chunk the filtered text
chunks = chunk_text(clean_text)





print(f"✅ Chunking complete. {len(chunks)} chunks created.")
print("Sample chunk:\n", chunks[0])  # Show a preview


text = extract_text_from_pdfs("medical_pdfs")  # Your cleaned text
chunks = chunk_text(text)

print(f"✅ Chunking complete. {len(chunks)} chunks created.")
print(f"Sample chunk:\n{chunks[0]}")


with open("processed_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Processed chunks saved to processed_chunks.pkl")
