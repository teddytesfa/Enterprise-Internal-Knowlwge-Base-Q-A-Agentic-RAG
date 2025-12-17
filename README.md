# Enterprise Internal Knowledge Base Q&A Agentic RAG

An autonomous **ReAct Agent** designed to answer complex questions about internal enterprise documents. This system leverages **Retrieval-Augmented Generation (RAG)** to ground its responses in a private knowledge base, ensuring accuracy and relevance.

## 🚀 Key Features

-   **Agentic Workflow**: Implements a custom **ReAct (Reasoning + Acting)** loop, allowing the AI to "think" before it answers and dynamically query the knowledge base as needed.
-   **RAG Pipeline**: Robust ingestion system that chunks and indexes documents (Markdown, PDF, etc.) into a vector store.
-   **Multi-LLM Support**: Configurable to run with **Google Gemini** or **Groq** (powering Llama 3 models) for high-speed inference.
-   **API-First Design**: Exposes a clean FastAPI interface for easy integration with frontends or other services.

## 🛠️ Tech Stack

-   **Orchestration (The RAG Pipeline)**: [LlamaIndex](https://www.llamaindex.ai/)
-   **LLM Providers**: Google Gemini / Groq
-   **Vector Database**: ChromaDB
-   **API Framework**: FastAPI
-   **Language**: Python 3.10+

## ⚡ Quick Start

Follow these steps to get the agent running locally.

### 1. Clone the Repository
```bash
git clone https://github.com/teddytesfa/Enterprise-Internal-Knowlwge-Base-Q-A-Agentic-RAG.git
cd Enterprise-Internal-Knowlwge-Base-Q-A-Agentic-RAG
```

### 2. Set Up Environment
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys & LLM Provider
Copy the example environment file:
```bash
cp .env.example .env
```
Then, configure your preferred LLM provider in `.env`.


#### Option A: Use Groq (Default)
Set your Groq API key. You can explicitly set the provider to `groq` or leave it default.
```ini
GROQ_API_KEY="your_groq_cloud_key"
LLM_PROVIDER="groq"
```

#### Option B: Use Google Gemini
Set your Google API key and change the provider to `gemini`.
```ini
GOOGLE_API_KEY="your_google_ai_studio_key"
LLM_PROVIDER="gemini"
```

### 5. Run the Server
Start the API server. This will automatically ingest sample documents from `resources/sample-datasets` if the index doesn't exist.
```bash
python src/api/main.py
```

## 🔌 Usage

Once the server is running (default: `http://0.0.0.0:8000`), you can query the agent via the `/query` endpoint.

**Example Request:**
```bash
curl -X POST "http://0.0.0.0:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do i set up my local dev for nexus project?"}'
```

**Example Response:**
```json
"To set up your local development environment for Project Nexus, you need to install Docker, configure the .env file..."
```

## 📚 Documentation & Experimentation

-   **Notebooks**:
    -   `notebooks/01_document_ingestion_indexing.ipynb`: Step-by-step code walkthrough of the RAG pipeline.
    -   `notebooks/react_agent_llamaIndex_rag_test.ipynb`: Demonstration of the ReAct Agent interacting with the RAG pipeline.
-   **Source Code**:
    -   `src/agent/`: Core ReAct agent logic.
    -   `src/ingestion/`: Document processing and indexing.
    -   `src/api/`: FastAPI entry point.
