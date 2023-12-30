from transformers import pipeline, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer
import pdfplumber
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gc  # Import garbage collector interface

# Function to extract text from PDF
def pdfReader(NameOfPdfFile):
    text = ""
    with pdfplumber.open(NameOfPdfFile) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Load model and tokenizer once (Memory Optimization)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

def ask_question_googleFlanLargeMethod(question, context):
    # Truncate context if it's too long (Memory Optimization)
    max_context_length = 1024  # Adjust as needed
    if len(context) > max_context_length:
        context = context[:max_context_length]

    # Combine the question with the context
    input_text = f"answer the question: '{question}' based on the following context. {context}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate the answer from the model
    outputs = model.generate(**inputs, max_length=512, no_repeat_ngram_size=2, num_return_sequences=1)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Clear memory (Memory Optimization)
    del inputs, outputs
    gc.collect()

    # Check if the answer is relevant
    if answer.lower() in context.lower():
        return answer
    else:
        return "The answer was not found in the document."

# Assuming 'amazon-rainforest-fact-sheet.pdf' is in the same directory as your script
context = pdfReader("amazon-rainforest-fact-sheet.pdf")

# Ask for user input
user_question = input("Ask your question from your document: ")
result = ask_question_googleFlanLargeMethod(user_question, context)
print(result)

if result == "The answer was not found in the document.":
    print("OK, Now we try another way")

# Clear memory after processing (Memory Optimization)
del context, user_question, result
gc.collect()
