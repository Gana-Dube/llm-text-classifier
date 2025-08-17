# 🤖 LLM Text Classifier

A high-performance, scalable microservice for text classification using Large Language Models (LLMs). Built with Python, Gradio, and Google's Gemini API.

## 🎯 Overview

This microservice provides an API for classifying text inputs using pre-trained LLMs through prompt engineering. It supports multiple classification tasks and is designed with scalability, performance, and production-readiness in mind.

## ✨ Features

- **Multiple Classification Tasks**: Supports both binary and multi-class classification
- **Configurable Prompts**: Easy configuration through YAML files
- **Production Ready**: Includes logging, error handling, rate limiting, and retry mechanisms
- **Interactive UI**: Gradio-based web interface for testing and demonstration
- **Batch Processing**: Support for processing multiple texts at once
- **Real Datasets**: Uses actual IMDB movie reviews and Amazon product reviews datasets

## 🏗️ Architecture

```
├── app.py                          # Main Gradio application
├── classifier/
│   ├── __init__.py
│   ├── text_classifier.py          # Main classifier orchestrator
│   ├── config_manager.py           # Configuration management
│   └── gemini_service.py           # Gemini API integration
├── config/
│   └── classifier_config.yaml      # Configuration file
├── data/                           # Downloaded datasets
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-text-classifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```

4. **Download datasets** (optional - for testing):
   ```bash
   python download_datasets.py
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Open your browser** and navigate to `http://localhost:7860`

## 📊 Supported Classification Tasks

### 1. Movie Review Sentiment Analysis
- **Type**: Binary classification
- **Classes**: positive, negative
- **Dataset**: IMDB 50K movie reviews
- **Use Case**: Analyze sentiment in movie reviews

### 2. Amazon Product Review Rating
- **Type**: Multi-class classification  
- **Classes**: 1, 2, 3, 4, 5 (star ratings)
- **Dataset**: Amazon product reviews
- **Use Case**: Predict star ratings from review text

## ⚙️ Configuration

The system is configured through `config/classifier_config.yaml`:

```yaml
classifier:
  model_provider: "gemini"
  model_name: "gemini-2.0-flash"
  temperature: 0.1
  max_tokens: 150

classification_tasks:
  sentiment:
    name: "Movie Review Sentiment Analysis"
    description: "Classify movie reviews as positive or negative"
    type: "binary"
    classes: ["positive", "negative"]
    prompt_template: |
      You are a sentiment analysis expert...

api:
  rate_limit: 60  # requests per minute
  timeout: 30     # seconds
  retry_attempts: 3
```

## 🔧 Usage

### Web Interface

1. **Single Text Classification**: Enter text and select a task
2. **Batch Classification**: Upload a CSV file with text data
3. **Model Information**: View current model and task configurations

### Programmatic Usage

```python
from classifier import TextClassifier

# Initialize classifier
classifier = TextClassifier()

# Classify single text
result = classifier.classify(
    text="This movie was absolutely fantastic!",
    task="sentiment"
)

# Batch classification
texts = ["Great movie!", "Terrible film", "Average story"]
results = classifier.batch_classify(texts, "sentiment")
```

## 🏭 Production Features

### Rate Limiting
- Configurable requests per minute
- Automatic throttling to prevent API quota exhaustion

### Error Handling
- Retry mechanisms with exponential backoff
- Graceful degradation on API failures
- Comprehensive error logging

### Logging
- Structured logging with configurable levels
- Request/response tracking
- Performance monitoring

### Scalability
- Stateless design for horizontal scaling
- Configurable timeouts and retry policies
- Efficient batch processing

## 📈 Performance

- **Response Time**: ~1-3 seconds per classification
- **Throughput**: Up to 60 requests/minute (configurable)
- **Batch Processing**: Supports up to 100 texts per batch
- **Memory Usage**: Minimal footprint with efficient text processing

## 🔒 Security

- API keys managed through environment variables
- Input validation and sanitization
- Rate limiting to prevent abuse
- No sensitive data stored in logs

## 🧪 Testing

### Manual Testing
1. Use the Gradio interface for interactive testing
2. Upload sample CSV files for batch testing
3. Monitor logs for performance and error tracking

### Sample Test Cases
- **Positive Review**: "Amazing product, highly recommended!"
- **Negative Review**: "Terrible quality, waste of money"
- **Neutral Review**: "Average product, nothing special"

## 📦 Deployment

### Local Development
```bash
python app.py
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

### Cloud Deployment
- **Railway**: Direct deployment from GitHub
- **Render**: Web service deployment
- **Hugging Face Spaces**: Gradio app hosting
- **Heroku**: Container deployment

## 🛠️ Development

### Adding New Classification Tasks

1. **Update configuration**:
   ```yaml
   classification_tasks:
     new_task:
       name: "New Classification Task"
       description: "Description of the task"
       classes: ["class1", "class2"]
       prompt_template: "Your prompt here..."
   ```

2. **Test the new task** through the web interface

### Adding New LLM Providers

1. Create a new service class (e.g., `openai_service.py`)
2. Implement the same interface as `GeminiService`
3. Update the configuration to support the new provider

## 📋 Requirements

- Python 3.8+
- Internet connection for LLM API calls
- Minimum 512MB RAM
- 1GB disk space (including datasets)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for LLM capabilities
- Gradio team for the excellent UI framework
- Kaggle for providing the datasets
- Open source community for various dependencies

## 📞 Support

For issues and questions:
1. Check the logs for error details
2. Verify API key configuration
3. Ensure all dependencies are installed
4. Create an issue in the repository

---

**Built with ❤️ for high-performance text classification**
