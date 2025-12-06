from flask import Flask, render_template, request
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import torch

app = Flask(__name__)

# Load model and tokenizer globally (cached in memory)
tokenizer = RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
model.eval()  # set to evaluation mode

# Use GPU if available for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load context from text file
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()


# Fast and accurate Q&A function with sliding window
def answer_question_fast(question, context, max_len=384, doc_stride=128):
    # Tokenize with sliding window for long contexts
    inputs = tokenizer(
        question,
        context,
        max_length=max_len,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move to device for inference
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    offset_mapping = inputs["offset_mapping"]
    
    # Get model predictions for all chunks
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
    
    # Find best answer across all chunks
    best_score = -float('inf')
    best_answer = ""
    
    num_chunks = input_ids.size(0)
    
    for chunk_idx in range(num_chunks):
        # Get top start and end positions for this chunk
        chunk_start_logits = start_logits[chunk_idx]
        chunk_end_logits = end_logits[chunk_idx]
        
        # Get best start and end indices
        start_idx = torch.argmax(chunk_start_logits).item()
        end_idx = torch.argmax(chunk_end_logits).item()
        
        # Skip if invalid
        if start_idx > end_idx or start_idx == 0 or end_idx == 0:
            continue
        
        # Get offsets for this chunk
        chunk_offset = offset_mapping[chunk_idx]
        
        # Check if offsets are valid
        if start_idx >= len(chunk_offset) or end_idx >= len(chunk_offset):
            continue
            
        start_char = chunk_offset[start_idx][0].item()
        end_char = chunk_offset[end_idx][1].item()
        
        # Skip padding tokens (offset = 0,0)
        if start_char == 0 and end_char == 0:
            continue
        
        # Extract answer
        answer = context[start_char:end_char].strip()
        
        # Calculate score
        score = chunk_start_logits[start_idx].item() + chunk_end_logits[end_idx].item()
        
        # Update best answer if score is better
        if score > best_score and answer and len(answer) > 2:
            best_score = score
            best_answer = answer
    
    return best_answer if best_answer else "No good answer found."

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = None
    if request.method == "POST":
        question = request.form.get("question", "")
        if question.strip():
            answer = answer_question_fast(question, context)
    return render_template("index.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(debug=True)
