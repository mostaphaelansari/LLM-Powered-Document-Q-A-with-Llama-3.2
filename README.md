# Deepseek AI Document Q&A

This project is an advanced document analysis and question-answering application powered by the LLAMA RAG model. It allows users to upload documents or paste text and get AI-powered insights and answers.

## Features

- Supports multiple document formats: PDF, DOCX, TXT
- Processes and indexes documents with metadata tracking
- Retrieves relevant document chunks for a given query
- Generates detailed and accurate responses based on the context
- Provides a Streamlit-based user interface for easy interaction

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LLM-Powered-Document-Q-A-with-Llama-3.2.git
    cd LLM-Powered-Document-Q-A-with-Llama-3.2
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload documents or paste text, enter your query, and get AI-powered insights and answers.

## Configuration

You can configure the application parameters from the sidebar in the Streamlit UI:

- **Model**: Select the model to use for generating responses.
- **Temperature**: Adjust the temperature for response generation.
- **Chunk Size**: Set the chunk size for text splitting.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
