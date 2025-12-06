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
    
    if not best_answer:
        return "No good answer found."
    
    # Convert to sentence form
    return format_answer_as_sentence(question, best_answer)


def format_answer_as_sentence(question, answer):
    """Convert answer to a complete sentence based on the question."""
    question_lower = question.lower().strip()
    
    # Remove question mark if present
    if question_lower.endswith('?'):
        question_lower = question_lower[:-1].strip()
    
    # Determine question type and format accordingly
    if question_lower.startswith('when '):
        # Extract subject from question
        subject = extract_subject_from_when(question_lower)
        # Ensure proper capitalization
        if subject:
            sentence = f"{subject} {answer}."
            # Capitalize first letter and clean up spaces
            sentence = sentence.strip()
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            # Remove double spaces
            sentence = ' '.join(sentence.split())
            return sentence
    
    elif question_lower.startswith('who '):
        # "Who founded..." -> "[Answer] founded..."
        verb_part = question_lower.replace('who ', '').strip()
        # Check if answer already contains the verb phrase
        if answer.lower() in question_lower:
            sentence = f"{answer.capitalize()}."
        else:
            sentence = f"{answer} {verb_part}."
        sentence = ' '.join(sentence.split())
        return sentence
    
    elif question_lower.startswith('what '):
        if 'what is' in question_lower or 'what was' in question_lower:
            subject = question_lower.replace('what is', '').replace('what was', '').strip()
            if subject:
                sentence = f"It is {answer}."
            else:
                sentence = f"The answer is {answer}."
        elif 'what did' in question_lower:
            parts = question_lower.split('what did')
            if len(parts) == 2:
                subject_verb = parts[1].strip()
                sentence = f"{answer.capitalize()}."
            else:
                sentence = f"The answer is {answer}."
        else:
            sentence = f"The answer is {answer}."
        sentence = ' '.join(sentence.split())
        return sentence
    
    elif question_lower.startswith('where '):
        # "Where did X develop" -> "X developed in [answer]"
        subject_part = question_lower.replace('where ', '').strip()
        if 'did' in subject_part:
            parts = subject_part.split('did')
            if len(parts) == 2:
                subject = parts[1].strip()
                # Handle verb conversion
                if 'develop' in subject:
                    verb = 'developed'
                    subject = subject.replace('develop', '').strip()
                elif 'live' in subject:
                    verb = 'lived'
                    subject = subject.replace('live', '').strip()
                else:
                    verb = 'was'
                sentence = f"{subject.capitalize()} {verb} in {answer}."
            else:
                sentence = f"It is in {answer}."
        else:
            sentence = f"It is in {answer}."
        sentence = ' '.join(sentence.split())
        return sentence
    
    elif question_lower.startswith('how '):
        sentence = f"{answer.capitalize()}."
        sentence = ' '.join(sentence.split())
        return sentence
    
    elif question_lower.startswith('why '):
        if not answer.lower().startswith('because'):
            sentence = f"Because {answer.lower()}."
        else:
            sentence = f"{answer.capitalize()}."
        sentence = ' '.join(sentence.split())
        return sentence
    
    else:
        # Default format for other question types
        sentence = f"The answer is {answer}."
        sentence = ' '.join(sentence.split())
        return sentence


def extract_subject_from_when(question):
    """Extract subject from 'when' questions."""
    # Remove 'when' and common question words
    subject = question.replace('when ', '').strip()
    
    # Handle different forms
    if 'did' in subject:
        # "when did India gain independence" -> "India gained independence in"
        parts = subject.split('did')
        if len(parts) == 2:
            noun = parts[0].strip().capitalize()
            verb_phrase = parts[1].strip()
            # Convert to past tense format
            if 'gain' in verb_phrase:
                verb_phrase = verb_phrase.replace('gain', 'gained')
            elif 'get' in verb_phrase:
                verb_phrase = verb_phrase.replace('get', 'got')
            elif 'start' in verb_phrase or 'begin' in verb_phrase:
                if 'start' in verb_phrase:
                    verb_phrase = verb_phrase.replace('start', 'started')
                else:
                    verb_phrase = verb_phrase.replace('begin', 'began')
            elif 'end' in verb_phrase:
                verb_phrase = verb_phrase.replace('end', 'ended')
            return f"{noun} {verb_phrase} in"
    
    elif 'was' in subject or 'were' in subject:
        # "when was the war" -> "The war was in"
        parts = subject.split('was') if 'was' in subject else subject.split('were')
        if len(parts) >= 1:
            noun = parts[0].strip().capitalize()
            verb = 'was' if 'was' in subject else 'were'
            return f"{noun} {verb} in"
    
    # Default format
    return "It was in"

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
