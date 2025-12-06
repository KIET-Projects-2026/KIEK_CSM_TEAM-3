from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import torch

# Load model and tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load context
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()

# Test question
question = "When did India gain independence?"

print(f"Question: {question}")
print(f"Using device: {device}")
print("-" * 50)

# Encode with sliding window
max_len = 384
doc_stride = 128

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

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
offset_mapping = inputs["offset_mapping"]

print(f"Number of chunks: {input_ids.size(0)}")

# Get predictions
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

# Find best answer
best_score = -float('inf')
best_answer = ""

for chunk_idx in range(input_ids.size(0)):
    chunk_start_logits = start_logits[chunk_idx]
    chunk_end_logits = end_logits[chunk_idx]
    
    start_idx = torch.argmax(chunk_start_logits).item()
    end_idx = torch.argmax(chunk_end_logits).item()
    
    if start_idx > end_idx or start_idx == 0 or end_idx == 0:
        continue
    
    chunk_offset = offset_mapping[chunk_idx]
    
    if start_idx >= len(chunk_offset) or end_idx >= len(chunk_offset):
        continue
        
    start_char = chunk_offset[start_idx][0].item()
    end_char = chunk_offset[end_idx][1].item()
    
    if start_char == 0 and end_char == 0:
        continue
    
    answer = context[start_char:end_char].strip()
    score = chunk_start_logits[start_idx].item() + chunk_end_logits[end_idx].item()
    
    print(f"\nChunk {chunk_idx}: score={score:.2f}, answer='{answer[:100]}...'")
    
    if score > best_score and answer and len(answer) > 2:
        best_score = score
        best_answer = answer

print(f"\n" + "=" * 50)
print(f"BEST ANSWER (score={best_score:.2f}): {best_answer}")
