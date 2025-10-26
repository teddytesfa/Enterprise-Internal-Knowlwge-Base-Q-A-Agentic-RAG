# Enterprise Internal Knowledge Base Q&A Agentic RAG - Document Ingestion and Indexing

This branch implements the document ingestion and indexing pipeline for an Enterprise Internal Knowledge Base Q&A Agentic RAG. The goal is to build a system that can answer questions based on a collection of internal documents.

## Overview

The core of this project branch is a pipeline that takes raw documents, processes them, and creates a searchable vector index. This index can then be used by a Retrieval-Augmented Generation (RAG) agent to find relevant information and answer user queries.

The pipeline consists of the following steps:

1.  **Document Ingestion**: Loads documents from a specified directory. It supports various file formats, including Markdown, PDF, and more.
2.  **Document Chunking**: Splits the loaded documents into smaller, manageable chunks using a sentence-aware splitter.
3.  **Embedding Generation**: Converts each document chunk into a vector embedding using a pre-trained sentence transformer model (`BAAI/bge-small-en-v1.5`).
4.  **Vector Indexing**: Stores the generated embeddings in a `ChromaDB` vector database, creating a semantic index that can be efficiently searched.

## Project Structure

```
/
├───.gitignore
├───README.md
├───data/
│   └───vector_db/
├───docs/
├───notebooks/
│   └───01_document_ingestion_indexing.ipynb
├───resources/
│   ├───sample-datasets/
│   │   ├───company_handbook.md
│   │   ├───project_nexus_onboarding_guide.md
│   │   └───troubleshooting_local_setup.md
│   └───static/
├───src/
│   └───ingestion/
│       ├───__init__.py
│       ├───config.py
│       ├───connector.py
│       ├───indexer.py
│       └───main.py
└───tests/
```

-   `data/vector_db`: Stores the ChromaDB vector index.
-   `notebooks`: Contains Jupyter notebooks for experimentation and prototyping.
-   `resources/sample-datasets`: Contains sample documents for the knowledge base.
-   `src/ingestion`: Contains the Python modules for the ingestion and indexing pipeline.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    _On Windows, use `.venv\Scripts\activate`_

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google API Key:**
    This project uses the Google Gemini LLM. You need to set up your Google API key as an environment variable.

    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    ```

## Usage

To run the document ingestion and indexing pipeline, execute the `main.py` script:

```bash
python src/ingestion/main.py
```

This will:
1.  Load the documents from the `resources/sample-datasets` directory.
2.  Chunk the documents.
3.  Generate embeddings and build the vector index in the `data/vector_db` directory.
4.  Run a few test queries to verify the pipeline.

## Notebook

The `notebooks/01_document_ingestion_indexing.ipynb` notebook provides a detailed, step-by-step walkthrough of the entire pipeline. It's a great resource for understanding the implementation details and for experimenting with different configurations.
