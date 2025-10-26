# Q-A-System
This project is an experiment in creating a complete Retrieval-Augmented Generation (RAG) pipeline. It uses AI to ingest local documents, understand their content by converting them into vector embeddings, and answer questions based on the indexed data.

## Features
- **Multi-Format Document Ingestion**: Supports various file types including `.txt`, `.md`, `.pdf`, `.docx`, and `.csv`.
- **Efficient Data Processing**: Documents are split into manageable chunks, embedded into vectors, and stored in a vector database.
- **Intelligent Index Updates**: When a file is re-indexed, the system automatically removes the old data before inserting the new version, preventing stale results.
- **Local RAG Pipeline**: Utilizes local, open-source models from Hugging Face for both embedding and question-answering, ensuring privacy and no external API costs for the LLM.
- **Interactive Querying**: Provides a command-line interface to ask questions about your documents.

## Technology Stack
- **Vector Database**: Pinecone
- **LLM Framework**: LangChain
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- **Generative LLM**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (running locally via Hugging Face `transformers`)
- **Core Libraries**: `torch`, `langchain-huggingface`, `pdfplumber`, `python-docx`, `pandas`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Q-A-System
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create a virtual environment
    python -m venv venv
    # Activate it
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    # Install required packages
    pip install -r requirements.txt 
    ```

3.  **Configure Environment Variables:**
    Create a file named `.env` in the root of the project and add your Pinecone credentials:
    ```
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_INDEX_NAME="your-chosen-index-name"
    ```

## Usage

The application has two main commands: `index` and `query`.

### 1. Indexing Documents
To process your local files and store them in Pinecone, use the `index` command.

```bash
# Index a single file
python main.py index "path/to/your/document.pdf"

# Index all supported files in a directory (and its subdirectories)
python main.py index "path/to/your/data_folder" --recursive
```

### 2. Querying Documents
After indexing, you can start an interactive session to ask questions about your documents.

```bash
python main.py query
```

The system will load the local LLM and prompt you to ask questions. Type `exit` to end the session.
