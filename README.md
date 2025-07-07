# AI Engineering Bootcamp: Multi-Provider LLM Chatbot UI

A hands-on project for exploring and comparing Large Language Model (LLM) APIs from OpenAI, Google, and Groq. This repository features a Streamlit-based chatbot UI for interactive experimentation, as well as example notebooks for direct API usage.

---

## Features

- **Chatbot UI**: Interact with LLMs from OpenAI, Google, and Groq in a unified web interface.
- **Model Selection**: Choose from available models for each provider.
- **Configurable Parameters**: Adjust temperature and max tokens for each chat session within the interface.
- **API Key Management**: Securely manage API keys via environment variables or a `.env` file.
- **Reset Chat**: Easily clear the conversation and start fresh.

---

## Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for fast, reliable dependency management)
- API keys for [OpenAI](https://platform.openai.com/), [Google Gemini](https://ai.google.dev/), and [Groq](https://console.groq.com/)

### Install uv (if not already installed)
```bash
pip install uv
```

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd 01-ai-engineering-bootcamp
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

API keys are required for each provider. The app uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to load them from environment variables or a `.env` file:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `GROQ_API_KEY`

---

## Usage

- Use the sidebar to select the provider (OpenAI, Google, Groq) and model.
- Adjust temperature and max tokens as needed.
- Enter your prompt in the chat input box.
- Click "Reset Chat" to clear the conversation.

---

## Example Notebooks

- [`notebooks/01-lllm-apis.ipynb.ipynb`](notebooks/01-lllm-apis.ipynb.ipynb):
  - Demonstrates direct usage of OpenAI, Google Gemini, and Groq APIs in Python.
  - Useful for learning how to authenticate and interact with each provider programmatically.

---

## License

This project is licensed under the [MIT License](LICENSE). 