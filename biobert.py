from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BioBERT model & tokenizer
biobert_model_name = "dmis-lab/biobert-base-cased-v1.1"
try:
    biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
    biobert_model = AutoModelForQuestionAnswering.from_pretrained(biobert_model_name)
except Exception as e:
    logging.error(f"Error loading BioBERT model: {e}")
    raise e

def get_biobert_answer(context, question):
    """
    Extract a medical fact using BioBERT's question-answering capability.
    
    Args:
        context (str): The retrieved medical text.
        question (str): The user’s medical query.
    
    Returns:
        str: Extracted answer or a fallback message.
    """
    if not context.strip():
        return "No relevant medical context found."

    try:
        inputs = biobert_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        outputs = biobert_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = biobert_tokenizer.convert_tokens_to_string(
            biobert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        return answer if answer.strip() else "No specific answer found in the provided text."
    except Exception as e:
        logging.error(f"Error in BioBERT answer extraction: {e}")
        return "Error extracting answer using BioBERT."
