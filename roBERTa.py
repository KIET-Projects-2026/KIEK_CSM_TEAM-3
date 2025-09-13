
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch

tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

context ="India’s history spans millennia, beginning with early human settlements between 73,000–55,000 years ago. The earliest known villages date to around 7000 BCE. The Indus Valley Civilization (2600–1900 BCE), centered in present-day Pakistan and northwestern India, was among the world’s earliest urban cultures, known for advanced architecture and trade. After its decline, Indo-Aryan tribes introduced Vedic culture, composing the Vedas—foundational Hindu texts—and organizing society into varnas, which evolved into the caste system. Around 600 BCE, Mahajanapadas emerged, alongside Jainism and Buddhism, which challenged Brahmanical rituals and emphasized nonviolence and liberation. In 321 BCE, Chandragupta Maurya unified much of India. His grandson Ashoka, after the Kalinga War, embraced Buddhism and spread dharma across Asia. The Gupta Empire (320–550 CE) marked a Golden Age, with advances in science, mathematics (including zero), literature, and art. Between 600–1200 CE, regional powers like the Chalukyas, Cholas, and Palas rose. The Cholas expanded into Southeast Asia, promoting Tamil culture and maritime trade. Grand temples and regional languages flourished. Islamic rule began with the Delhi Sultanate in 1206, introducing Persian culture and administration. Despite conflict, Hindu-Muslim syncretism emerged. The Mughal Empire, founded by Babur in 1526, centralized power and left a legacy of architecture (e.g., Taj Mahal), literature, and art. Akbar promoted tolerance; Aurangzeb leaned toward orthodoxy. The empire declined by the 18th century. In 1757, the British East India Company gained control after the Battle of Plassey. India became a colony in 1858. Colonial rule brought railways, Western education, and exploitation. The freedom struggle, led by Gandhi, Nehru, and Ambedkar, emphasized nonviolence and reform. India gained independence in 1947, followed by partition and widespread displacement. A democratic constitution was adopted in 1950. Since then, India has grown into a global power, advancing in technology, space, and economics, while facing challenges like poverty, inequality, and communal tensions. Its history reflects resilience, innovation, and cultural diversity."

while True:
    question = input("Enter your question (or type 'exit' to quit):\n")
    if question.lower() == 'exit':
        break

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    start_scores, end_scores = model(**inputs).values()

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    )

    print("Answer:", answer)

