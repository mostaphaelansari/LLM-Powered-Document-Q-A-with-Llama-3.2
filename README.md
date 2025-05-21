# Deepseek AI Document Q&A

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-0.0.300+-green.svg)
![LLAMA](https://img.shields.io/badge/LLAMA-3.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

An advanced document analysis and question-answering application powered by the LLAMA RAG (Retrieval-Augmented Generation) model. The system enables users to upload documents or paste text content and receive AI-powered insights and contextual answers through an intuitive web interface.

## Overview

Deepseek AI Document Q&A leverages cutting-edge natural language processing to transform how users interact with their documents. By combining document parsing, intelligent chunking, and retrieval-augmented generation, the application provides accurate, context-aware responses to user queries based on uploaded content.

## Technology Stack

### Core Technologies
- **Python 3.8+** - Primary development language
- **LLAMA 3.2** - Large language model for response generation
- **LangChain** - Framework for LLM application development
- **Streamlit** - Web application framework
- **Chroma/FAISS** - Vector database for document embedding storage

### Document Processing
- **PyPDF2/pdfplumber** - PDF document parsing
- **python-docx** - DOCX file processing
- **tiktoken** - Text tokenization and chunking
- **sentence-transformers** - Text embedding generation

### AI/ML Libraries
- **transformers** - Hugging Face model integration
- **torch** - PyTorch for deep learning operations
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis

## Features

- **Multi-format Support**: Seamlessly processes PDF, DOCX, and TXT documents
- **Intelligent Document Indexing**: Advanced text chunking with metadata preservation
- **Semantic Search**: Vector-based retrieval for relevant document sections
- **Context-aware Responses**: RAG-powered answers with source attribution
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Configurable Parameters**: Adjustable model settings and processing options
- **Real-time Processing**: Live document analysis and query processing

## Project Structure

```
├── app.py                          # Main Streamlit application
├── src/
│   ├── document_processor.py       # Document parsing and chunking
│   ├── embeddings.py              # Text embedding generation
│   ├── retriever.py               # Document retrieval logic
│   ├── llm_handler.py             # LLAMA model integration
│   └── utils.py                   # Utility functions
├── config/
│   ├── model_config.yaml          # Model configuration settings
│   └── app_config.yaml            # Application parameters
├── data/
│   ├── uploads/                   # Uploaded document storage
│   └── vector_store/              # Vector database storage
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── docker-compose.yml             # Multi-service orchestration
└── tests/
    ├── test_processor.py          # Document processing tests
    └── test_retriever.py          # Retrieval system tests
```

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (recommended for optimal model performance)
- GPU support (optional, for faster inference)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LLM-Powered-Document-Q-A-with-Llama-3.2.git
   cd LLM-Powered-Document-Q-A-with-Llama-3.2
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and model paths
   ```

## Dependencies

### Core Requirements
```
streamlit>=1.28.0
langchain>=0.0.300
transformers>=4.35.0
torch>=2.0.0
sentence-transformers>=2.2.2
chromadb>=0.4.0
```

### Document Processing
```
PyPDF2>=3.0.1
python-docx>=0.8.11
pdfplumber>=0.9.0
tiktoken>=0.5.1
```

### Additional Libraries
```
numpy>=1.24.0
pandas>=2.0.0
regex>=2023.8.8
tqdm>=4.65.0
```

## Usage

### Starting the Application

1. **Launch the Streamlit interface**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will be ready for document upload and queries

### Using the Interface

1. **Document Upload**: Use the file uploader to select PDF, DOCX, or TXT files
2. **Text Input**: Alternatively, paste text directly into the text area
3. **Query Processing**: Enter your questions in the query input field
4. **Configuration**: Adjust settings in the sidebar for optimal results

## Configuration

### Model Parameters

Configure the application through the Streamlit sidebar:

- **Model Selection**: Choose from available LLAMA model variants
- **Temperature** (0.0-1.0): Controls response creativity and randomness
- **Chunk Size** (100-2000): Sets document segmentation size
- **Overlap Size** (0-200): Defines chunk overlap for context preservation
- **Max Tokens**: Limits response length

### Advanced Settings

Edit `config/model_config.yaml` for advanced configurations:

```yaml
model:
  name: "llama-3.2-8b"
  temperature: 0.7
  max_tokens: 512

processing:
  chunk_size: 1000
  chunk_overlap: 100
  embedding_model: "all-MiniLM-L6-v2"

retrieval:
  similarity_threshold: 0.7
  max_results: 5
```

## Docker Deployment

For containerized deployment:

```bash
docker-compose up -d
```

This will start the application with all necessary services configured.

## Performance Optimization

- **GPU Acceleration**: Enable CUDA for faster model inference
- **Model Caching**: Pre-load models to reduce startup time
- **Batch Processing**: Process multiple documents simultaneously
- **Vector Store Optimization**: Use persistent storage for large document collections

## Contributing

We welcome contributions to improve the project! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new functionality
- Update documentation for API changes
- Ensure compatibility with Python 3.8+

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Meta AI** for the LLAMA 3.2 model
- **LangChain** community for the RAG framework
- **Streamlit** team for the web application framework
- **Hugging Face** for model hosting and transformers library

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the [documentation](docs/) for detailed guides
- Join our community discussions
