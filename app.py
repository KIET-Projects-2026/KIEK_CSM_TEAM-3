from flask import Flask, render_template, request
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import torch

app = Flask(__name__)

# Load model and tokenizer globally (cached in memory)
tokenizer = RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
model.eval()  # set to evaluation mode

# Hardcoded context
context = """India’s history spans tens of thousands of years, beginning with early human settlements between 73,000–55,000 years ago. Archaeological evidence from sites such as Bhimbetka caves suggests a long record of prehistoric life. The earliest known agricultural villages date to around 7000 BCE at Mehrgarh (in present-day Pakistan). The Indus Valley Civilization (2600–1900 BCE), centered in present-day Pakistan and northwestern India, was one of the world’s earliest and most sophisticated urban cultures. Cities like Harappa and Mohenjo-Daro were notable for their advanced urban planning, drainage systems, standardized weights, seals, and trade networks. After its decline, Indo-Aryan tribes introduced Vedic culture, composing the Vedas, which remain foundational Hindu scriptures. Vedic society was organized into varnas (social orders), which later evolved into the caste system. By around 600 BCE, the Mahajanapadas (large kingdoms) emerged. During this period, Jainism (founded by Mahavira) and Buddhism (founded by Siddhartha Gautama, the Buddha) challenged ritualistic Brahmanism, emphasizing nonviolence, meditation, and liberation from suffering. In 321 BCE, Chandragupta Maurya unified much of northern India, establishing the Maurya Empire. His grandson, Ashoka the Great, after the bloody Kalinga War, embraced Buddhism and became one of history’s greatest patrons of peace and dharma. He sent Buddhist missions across Asia, spreading Indian philosophy, art, and ethics. After centuries of fragmentation, the Gupta Empire (320–550 CE) ushered in India’s Golden Age. Mathematics (the concept of zero, decimal system, Aryabhata’s astronomy), science, medicine, Sanskrit literature (Kalidasa), and art flourished. Magnificent temples and sculptures reflected religious devotion and artistic mastery. Between 600–1200 CE, powerful regional dynasties rose, including the Chalukyas, Pallavas, Palas, and the Cholas. The Chola Empire became a naval power, expanding influence across Southeast Asia (Indonesia, Malaysia) and spreading Tamil culture, language, and temple architecture. Temples such as Brihadeeswarar in Thanjavur remain cultural icons. Islamic influence in India began through trade and later conquest. In 1206, the Delhi Sultanate was established, introducing Persian administration, architecture, and culture. Despite political conflict, India saw cultural syncretism, with Sufi traditions blending with Bhakti movements, creating shared spaces of devotion. In 1526, Babur founded the Mughal Empire, which became one of the most powerful empires in the world. Under emperors like Akbar, who promoted religious tolerance, and Shah Jahan, who built the Taj Mahal, Mughal India flourished in art, literature, architecture, and administration. However, Aurangzeb’s orthodoxy created tensions, and the empire declined by the 18th century. The British East India Company rose after the Battle of Plassey (1757), gradually expanding its rule. By 1858, after the Revolt of 1857, India came under direct British Crown control. British rule brought railways, Western education, law, and new industries, but also economic exploitation, famines, and cultural suppression. India’s freedom struggle in the late 19th and 20th centuries saw leaders like Mahatma Gandhi (nonviolence and satyagraha), Jawaharlal Nehru (modernization and democracy), B. R. Ambedkar (social justice and constitution-building), and many revolutionaries who fought colonial rule. India finally gained independence in 1947, but was partitioned into India and Pakistan, leading to mass migrations and violence. The Republic of India adopted its democratic Constitution in 1950, guaranteeing secularism, equality, and rights for all citizens. Since independence, India has emerged as a global power, making strides in science, space exploration (ISRO), IT, nuclear technology, and economic development. Yet, it continues to face challenges such as poverty, inequality, environmental issues, and communal tensions. India’s culture is one of the oldest and most diverse in the world. It is shaped by thousands of years of interaction among Hindu, Buddhist, Jain, Islamic, Sikh, and Christian traditions. Key cultural aspects include religion and philosophy, art and architecture, language and literature, science and knowledge, festivals, cuisine, and music and dance. India is the birthplace of Hinduism, Buddhism, Jainism, and Sikhism, and has absorbed influences from Islam and Christianity. Spirituality and philosophy have long shaped its identity. From Indus seals to Ajanta cave paintings, from temple architecture (Khajuraho, Konark, Meenakshi) to Mughal monuments (Taj Mahal, Red Fort, Fatehpur Sikri), India’s artistic heritage is vast. India has 22 official languages and thousands of dialects. Sanskrit literature, Tamil Sangam poetry, medieval Bhakti poetry, and modern writers (Tagore, Premchand, R. K. Narayan, Arundhati Roy) highlight its literary diversity. India contributed zero, decimal system, Ayurveda, yoga, astronomy, metallurgy, and surgery techniques. Ancient universities like Nalanda and Takshashila were global centers of learning. Diwali, Holi, Eid, Christmas, Vaisakhi, Pongal, Onam, and Navratri reflect India’s cultural pluralism. Regional cuisines, spices, and vegetarian traditions form a rich culinary heritage. Classical traditions (Hindustani, Carnatic), folk music, and dance forms (Bharatanatyam, Kathak, Odissi, Kathakali) coexist with modern cinema and Bollywood music.
Andhra Pradesh: Known for the Satavahana dynasty and Kakatiya empire, Andhra Pradesh has a rich history in Telugu culture, art, and literature. It plays a significant role in agriculture, technology (IT hubs like Visakhapatnam), and classical music and dance traditions.

Arunachal Pradesh: Historically home to various tribal kingdoms and influenced by Tibetan culture, Arunachal Pradesh is strategically important for India due to its location on the border with China and Bhutan. It contributes to biodiversity and tribal heritage.

Assam: Assam's history is marked by the Ahom dynasty, which resisted Mughal invasions. It is famous for tea plantations, oil reserves, and the Brahmaputra River, making it crucial for agriculture, trade, and biodiversity.

Bihar: Ancient Bihar was the center of the Maurya and Gupta empires and the birthplace of Buddhism and Jainism. It has historically played a pivotal role in politics, education (Nalanda University), and culture.

Chhattisgarh: Formed in 2000, Chhattisgarh was historically part of Madhya Pradesh with tribal kingdoms. Known for its minerals and forests, it plays an important role in steel production and energy resources.

Goa: Goa, a Portuguese colony until 1961, has a unique Indo-Portuguese culture. Its historical forts, churches, and beaches make it a major tourism hub. It also plays a role in maritime trade.

Gujarat: Known for the Indus Valley Civilization site of Lothal, Gujarat has a history of the Solanki and Maratha empires. It is an economic powerhouse with industries, ports, and cultural contributions like Mahatma Gandhi’s birthplace.

Haryana: Historically part of the Mahabharata era (Kurukshetra), Haryana was under the rule of the Mughals and Marathas. It is important for agriculture, manufacturing, and being close to Delhi.

Himachal Pradesh: Known for its Himalayan kingdoms and British hill stations, Himachal Pradesh has a rich cultural heritage of temples and monasteries. Tourism and hydropower are key roles.

Jharkhand: Created in 2000 from Bihar, Jharkhand has rich mineral resources and tribal culture. It is a key contributor to steel, coal, and other industries in India.

Karnataka: Historically ruled by the Vijayanagara Empire and Chalukyas, Karnataka has a rich heritage in architecture, literature, and classical music. Bangalore makes it a technological and IT hub.

Kerala: Known for the Chera dynasty, Kerala played a major role in the spice trade with Europe. It is famous for its backwaters, Ayurveda, literacy, and social reforms.

Madhya Pradesh: Called the “Heart of India,” it was home to the Maurya, Gupta, and Chandela dynasties. Known for temples, wildlife, and mineral resources, it has been central to politics and culture.

Maharashtra: Historically ruled by the Marathas under Shivaji, Maharashtra has been a center of trade, politics, and culture. Mumbai, the financial capital, contributes massively to economy, cinema, and industry.

Manipur: Known for its ancient kingdoms and classical dance forms like Manipuri, the state has a history of resisting invasions. It plays a role in sports and cultural preservation.

Meghalaya: Home to Khasi, Garo, and Jaintia tribes, Meghalaya has a rich tribal culture and unique matrilineal society. Its forests, waterfalls, and biodiversity are significant.

Mizoram: Historically ruled by tribal chiefs, Mizoram became a state in 1987. It is culturally rich with music and dance and contributes to biodiversity and border security.

Nagaland: Known for Naga tribes and resistance to British and Indian rule, Nagaland is significant for tribal culture, festivals, and as a strategically important northeastern state.

Odisha: Ancient Kalinga, ruled by Mauryas and later Eastern Ganga dynasty, Odisha is famous for temples, maritime trade, and the Jagannath culture. It contributes to industry, handicrafts, and classical dance.

Punjab: Known as the land of five rivers, historically ruled by Sikh Gurus and the Mughals, Punjab plays a major role in agriculture (Green Revolution) and industry. It is also a cultural and religious hub for Sikhs.

Rajasthan: Famous for Rajput kingdoms, forts, and desert culture, Rajasthan has historically been important in trade, defense, and art. Tourism and handicrafts are key contributions.

Sikkim: Formerly a monarchy, Sikkim merged with India in 1975. Known for monasteries and biodiversity, it plays a role in tourism, horticulture, and strategic Himalayan border security.

Tamil Nadu: Ancient Tamil kingdoms like Cholas and Pandyas left a rich cultural heritage. Tamil Nadu contributes significantly to classical arts, literature, temples, and industry.

Telangana: Formed in 2014, Telangana has historical ties with the Kakatiya dynasty. Hyderabad is a technology and pharmaceutical hub, contributing to IT and culture.

Tripura: Historically under Tripuri kings, it has a rich tribal culture. Agriculture and cultural heritage, along with strategic northeastern location, define its importance.

Uttar Pradesh: Ancient empires like Maurya, Gupta, Mughal, and later British influence mark UP’s history. It is significant for religion (Hinduism, Buddhism), politics, and agriculture.

Uttarakhand: Known as the land of the Himalayas and religious centers like Haridwar and Rishikesh, Uttarakhand has historical kingdoms and contributes through tourism, hydropower, and pilgrimage.

West Bengal: Historically part of Bengal Sultanate and British India, it played a central role in trade, education, and culture. Kolkata is a cultural and intellectual hub, contributing to industry and literature.
"""


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
