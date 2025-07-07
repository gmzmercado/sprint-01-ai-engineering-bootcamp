# AI Engineering Bootcamp: Multi-Provider LLM Chatbot UI (Prototype)

> **Disclaimer:** This repository is a **production/testing prototype** for educational and experimental purposes. It is not a final product and may change significantly. Use with caution in production environments.

A hands-on project for exploring and comparing Large Language Model (LLM) APIs from OpenAI, Google, and Groq. This repository features a Streamlit-based chatbot UI with a Retrieval-Augmented Generation (RAG) pipeline, Qdrant vector search integration, and evaluation tools for experimentation and learning.

---

## Features

- **Chatbot UI**: Interact with LLMs from OpenAI, Google, and Groq in a unified web interface.
- **Retrieval-Augmented Generation (RAG)**: Answers are generated using context retrieved from a Qdrant vector database, enabling product-aware responses.
- **Model Selection**: Choose from available models for each provider.
- **Configurable Parameters**: Adjust temperature and max tokens for each chat session within the interface.
- **API Key Management**: Securely manage API keys via environment variables or a `.env` file.
- **Reset Chat**: Easily clear the conversation and start fresh.
- **Automated Evaluation**: Evaluate the RAG pipeline using RAGAS metrics and LangSmith with provided scripts.

---

## Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for fast, reliable dependency management)
- API keys for [OpenAI](https://platform.openai.com/), [Google Gemini](https://ai.google.dev/), and [Groq](https://console.groq.com/)
- [Qdrant](https://qdrant.tech/) instance (local or remote)

### Install uv (if not already installed)
```bash
pip install uv
```

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd sprint-01-ai-engineering-bootcamp
   ```
2. **Install dependencies using uv:**
   ```bash
   uv pip install .
   ```
3. **Set up environment variables:**
   Create a `.env` file in the project root or export variables in your shell:
   ```env
   OPENAI_API_KEY=your-openai-key
   GOOGLE_API_KEY=your-google-key
   GROQ_API_KEY=your-groq-key
   QDRANT_URL=localhost
   QDRANT_COLLECTION_NAME=your-collection
   EMBEDDING_MODEL=text-embedding-3-small
   EMBEDDING_MODEL_PROVIDER=openai
   GENERATION_MODEL=gpt-4o
   GENERATION_MODEL_PROVIDER=openai
   LANGSMITH_API_KEY=your-langsmith-key  # (optional, for evaluation)
   LANGSMITH_PROJECT=your-project-name   # (optional)
   ```
4. **Run the Streamlit app:**
   ```bash
   make run-streamlit
   ```
   The app will be available at [http://localhost:8501](http://localhost:8501).

---

## Docker Setup (via Makefile)

You can use the provided Makefile for building and running the Docker container:

1. **Build the Docker image:**
   ```bash
   make build-docker-streamlit
   ```
2. **Run the container:**
   ```bash
   make run-docker-streamlit
   ```
   This will mount your local `.env` file and expose the app at [http://localhost:8501](http://localhost:8501).

---

## Configuration

API keys and configuration are required for each provider and for Qdrant. The app uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to load them from environment variables or a `.env` file:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`
- `EMBEDDING_MODEL`, `EMBEDDING_MODEL_PROVIDER`
- `GENERATION_MODEL`, `GENERATION_MODEL_PROVIDER`
- `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (for evaluation)

---

## Retrieval-Augmented Generation (RAG) Pipeline

The chatbot uses a RAG pipeline to answer questions about products:
1. **Embedding**: User queries are embedded using the specified model.
2. **Retrieval**: Relevant product data is retrieved from Qdrant based on vector similarity.
3. **Prompt Construction**: Retrieved context is formatted and combined with the user question.
4. **LLM Generation**: The prompt is sent to the selected LLM to generate a context-aware answer.

See [`src/chatbot_ui/retrieval.py`](src/chatbot_ui/retrieval.py) for implementation details.

---

## Data & Preprocessing

- The `data/` directory contains large product datasets in JSONL format for retrieval.
- Use the notebook [`notebooks/02-RAG-Preprocessing.ipynb`](notebooks/02-RAG-Preprocessing.ipynb) to preprocess data, generate embeddings, and upload to Qdrant.

---

## Evaluation

Automated evaluation of the RAG pipeline is provided via [`evals/eval_retriever.py`](evals/eval_retriever.py):
- Uses [RAGAS](https://github.com/explodinggradients/ragas) metrics and [LangSmith](https://smith.langchain.com/) for faithfulness, relevancy, and context precision/recall.
- Requires a LangSmith API key and a dataset for evaluation.
- Example usage:
  ```python
  # Set LANGSMITH_API_KEY in your environment
  python evals/eval_retriever.py
  ```

---

## Usage

- Use the sidebar to select the provider (OpenAI, Google, Groq) and model.
- Adjust temperature and max tokens as needed.
- Enter your prompt in the chat input box.
- Click "Reset Chat" to clear the conversation.
- Answers are generated using the RAG pipeline and product data.

---

## Example Notebooks

- [`notebooks/02-RAG-Preprocessing.ipynb`](notebooks/02-RAG-Preprocessing.ipynb):
  - Demonstrates preprocessing, embedding, and uploading product data to Qdrant for RAG.

---

## License

This project is licensed under the [MIT License](LICENSE). 