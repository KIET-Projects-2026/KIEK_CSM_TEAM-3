# Question Answering System with Fine-tuned RoBERTa

A web-based question answering application powered by a fine-tuned RoBERTa model that provides accurate, complete sentence answers about India's history.

## 🌟 Features

- **Accurate Question Answering**: Uses the `deepset/roberta-base-squad2` model fine-tuned on SQuAD 2.0 dataset
- **Complete Sentence Responses**: All answers are formatted as grammatically correct, complete sentences
- **Multiple Question Types**: Supports When, Who, What, Where, How, and Why questions
- **Optimized Performance**: Sliding window approach with GPU acceleration for fast responses
- **User-Friendly Interface**: Clean, dark-themed web interface for easy interaction
- **Context-Based Answers**: Extracts answers from a comprehensive text about India's history

## 📋 Requirements

- Python 3.8+
- Flask
- PyTorch
- Transformers (Hugging Face)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KIET-Projects-2026/KIEK_CSM_TEAM-3.git
   cd KIEK_CSM_TEAM-3
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask torch transformers
   ```

## 💻 Usage

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to: `http://127.0.0.1:5000`

3. **Ask questions**
   - Type your question in the input field
   - Click "Submit"
   - Receive a complete sentence answer

## 📝 Example Questions & Answers

**Q:** When did India gain independence?  
**A:** India gained independence in 1947.

**Q:** Who founded the Mughal Empire?  
**A:** Babur founded the mughal empire.

**Q:** What is the Taj Mahal?  
**A:** It is Mughal monuments.

**Q:** Where did the Indus Valley Civilization develop?  
**A:** The indus valley civilization developed in present-day Pakistan and northwestern India.

## 🏗️ Project Structure

```
KIEK_CSM_TEAM-3/
├── app.py                      # Main Flask application
├── roBERTa.py                  # Standalone RoBERTa Q&A script
├── context.txt                 # Knowledge base (India's history)
├── templates/
│   └── index.html              # Web interface
├── test_qa.py                  # Basic testing script
├── test_comprehensive.py       # Comprehensive test suite
├── DEPLOYMENT_REPORT.txt       # Deployment readiness report
└── README.md                   # This file
```

## 🔧 Technical Details

### Model
- **Model Name**: `deepset/roberta-base-squad2`
- **Architecture**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Training Data**: SQuAD 2.0 (Stanford Question Answering Dataset)

### Optimizations
- **Sliding Window**: Processes long contexts in overlapping chunks (384 tokens, 128 stride)
- **GPU Acceleration**: Automatically uses GPU if available, falls back to CPU
- **Batch Inference**: Processes multiple chunks simultaneously for better performance
- **Efficient Answer Extraction**: Uses offset mapping for accurate character-level extraction

### Answer Formatting
The application intelligently formats answers based on question type:
- **When questions**: "India gained independence in 1947."
- **Who questions**: "Babur founded the mughal empire."
- **What questions**: "It is Mughal monuments."
- **Where questions**: "The indus valley civilization developed in..."
- **How/Why questions**: Context-appropriate complete sentences

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_comprehensive.py
```

**Current Test Results:**
- Total Tests: 10
- Pass Rate: 90% (9/10 tests passed)
- All question types validated

## 🎨 User Interface

The web interface features:
- Dark theme for comfortable viewing
- Clean, modern design
- Responsive layout
- Real-time question answering
- Clear answer display

## ⚙️ Configuration

Key parameters in `app.py`:
- `max_len`: Maximum token length (default: 384)
- `doc_stride`: Overlap between chunks (default: 128)
- `device`: Auto-detects GPU/CPU

## 📊 Performance

- **Response Time**: Optimized for fast inference (~2-3 seconds on CPU)
- **Accuracy**: 90%+ on test questions
- **Context Handling**: Processes contexts up to several thousand words

## 🔒 Known Limitations

- Answer quality depends on the information available in `context.txt`
- Some specific question phrasings may require rephrasing for best results
- Extractive model (extracts answers from context, doesn't generate new text)

## 🛠️ Troubleshooting

**Issue**: Slow response times  
**Solution**: Ensure GPU is available or reduce `max_len` parameter

**Issue**: "No good answer found"  
**Solution**: Rephrase the question or ensure the answer exists in context.txt

**Issue**: Model loading errors  
**Solution**: Check internet connection (first run downloads the model)

## 📚 Context Information

The application uses a knowledge base covering:
- Ancient India (Indus Valley Civilization, Vedic period)
- Medieval India (Maurya, Gupta, Delhi Sultanate, Mughal Empire)
- Colonial period (British rule)
- Independence movement
- Post-independence India

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Expand the knowledge base in `context.txt`
- Add more question types
- Improve answer formatting logic
- Add unit tests
- Implement caching for repeated questions

## 📄 License

This project is part of KIET academic projects.

## 👥 Team

KIEK_CSM_TEAM-3

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **deepset** for the fine-tuned RoBERTa model
- **Stanford NLP** for the SQuAD dataset

## 📞 Support

For issues or questions, please open an issue on the GitHub repository.

---

**Note**: This is a development server. For production deployment, use a production-grade WSGI server like Gunicorn or Waitress.
