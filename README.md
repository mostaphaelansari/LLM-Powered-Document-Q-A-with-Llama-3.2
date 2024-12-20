# LLM-Powered-Document-Q-A-with-Llama-3.2
A Retrieval-Augmented Generation (RAG) application that enables intelligent document analysis and question answering using Llama 3.2. Built with Streamlit, Langchain, and Ollama.
## Features

- **Multi-format Document Support**: Process PDF, DOCX, TXT, MD, and CSV files with intelligent text extraction.
- **Advanced Text Processing**: Automatic text chunking and embedding using LangChain.
- **Vector Search**: FAISS-powered similarity search for accurate document retrieval.
- **Configurable Parameters**: Adjustable model settings, chunk sizes, and temperature.
- **Comprehensive Metadata Tracking**: Detailed processing statistics and response metrics.
- **Interactive UI**: Clean Streamlit interface with chat history and document statistics.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Required Python packages:
    - `ollama`
    - `streamlit`
    - `chardet`
    - `PyPDF2`
    - `python-docx`
    - `langchain`
    - `faiss-cpu`

## Installation

1. Clone the repository:
        ```bash
        git clone <repository-url>
        cd enhanced-rag-app
        ```
2. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
3. Ensure Ollama is running locally on port 11434.

## Usage

1. Start the application:
        ```bash
        streamlit run app.py
        ```
2. Access the web interface at [http://localhost:8501](http://localhost:8501).
3. Configure the application using the sidebar:
        - Select the LLM model.
        - Adjust temperature and chunk size.
        - Upload documents or paste text directly.
4. Enter questions and receive AI-powered responses based on your documents.

## Architecture

### Key Components

- **DocumentProcessor**
    - Handles multiple document formats.
    - Extracts text with metadata.
    - Provides error handling and encoding detection.

- **EnhancedRAGApplication**
    - Manages document indexing and retrieval.
    - Handles embedding generation.
    - Coordinates response generation.
    - Tracks performance metrics.

- **Streamlit UI**
    - Provides interactive interface.
    - Displays document statistics.
    - Maintains chat history.
    - Shows response metadata.

## Configuration Options

- `model_name`: LLM model selection (default: "llama3.2")
- `embedding_model`: Model for generating embeddings
- `chunk_size`: Text chunk size (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `temperature`: Response creativity (0.0-1.0)
- `max_tokens`: Maximum response length

## Error Handling

The application includes comprehensive error handling:

- Document processing validation
- Encoding detection and fallback
- Vector store initialization checks
- Response generation monitoring

## Logging

Detailed logging is implemented throughout:

- Document processing events
- Vector store operations
- Response generation metrics
- Error tracking

## Performance Considerations

- Document chunking is optimized for balance between context and performance.
- Vector similarity search uses efficient FAISS indexing.
- Response generation includes context length management.
- UI updates are optimized for responsiveness.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Implement changes with tests.
4. Submit a pull request.

## License

[Insert License Information]

## Acknowledgments

- Ollama team for the LLM integration.
- Streamlit for the UI framework.
- LangChain for document processing utilities.
- FAISS for vector search capabilities.
