# Conversational PDF RAG Assistant

This is a simple Streamlit app that lets you upload PDFs and chat with them. It uses LangChain for the RAG logic, Groq for fast LLM responses, and Ollama for local embeddings. It also keeps track of your chat history so you can ask follow-up questions.

## How it works
- 1. **Upload:** You drop one or more PDFs into the UI.
- 2. **Embed:** The app uses Ollama (nomic-embed-text) to turn your text into vectors.
- 3. **Chat:** You ask a question. The app looks at your history, rephrases the question if needed, finds the relevant parts of your PDFs, and gives you an answer via Groq's Llama 3.1 model.


## Setup
- **1. Clone and Install** 
```bash
git clone https://github.com/SangitaSingha073/conversational-pdf-rag.git
cd conversational-pdf-rag
pip install -r requirements.txt
```
- **2. Local Embeddings** You need Ollama running on your machine. Once installed, pull the embedding model:
```bash
ollama pull nomic-embed-text
```
- **3. Run it** 
```bash
streamlit run RAG_app.py
```

## What you'll need
- A Groq API Key ([Get your Groq API Key here](https://console.groq.com/))
- Ollama installed and running locally.

