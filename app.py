from flask import Flask, render_template, request
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import torch

app = Flask(__name__)

# Load model and tokenizer globally (cached in memory)
tokenizer = RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
model.eval()  # set to evaluation mode

# Load context from text file
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()


# Optimized Q&A function
def answer_question_fast(question, context, max_len=512, stride=128):
    encoding = tokenizer(
        question,
        context,
        max_length=max_len,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mappings = encoding["offset_mapping"]

    # Batch inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    best_answer = ""
    best_score = float("-inf")

    for i in range(input_ids.size(0)):
        start_index = torch.argmax(start_logits[i])
        end_index = torch.argmax(end_logits[i])

        if start_index <= end_index:
            start_char = offset_mappings[i][start_index][0].item()
            end_char = offset_mappings[i][end_index][1].item()
            answer = context[start_char:end_char].strip()
            score = start_logits[i][start_index].item() + end_logits[i][end_index].item()

            if score > best_score and answer:
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
