# rag_groqapi_fastapi_streamlit# RAG Framework with Groq API, FastAPI, and Streamlit

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) framework** using **Groq API for inference**, **FastAPI as the backend**, **Streamlit as the frontend UI**, **ChromaDB as the vector store**, and **Llama 3.2 as the LLM**. The backend leverages the **LangChain framework** to streamline interactions with the vector database and LLM.

## Inspiration
This project has been inspired by [Ashish Sinha's RAG API repository](https://github.com/AshishSinha5/rag_api/tree/main), with modifications and enhancements to integrate **Groq API, FastAPI, and Streamlit** for an interactive user experience.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Vector Store:** ChromaDB
- **LLM:** Llama 3.2 via Groq API
- **Framework:** LangChain

## Features
- **Initialize LLM**: Users can select Llama 3.2 as the model and initialize it via Groq API.
- **Upload Documents**: Supports PDF and HTML file uploads, storing embeddings in ChromaDB.
- **Query LLM**: Retrieves relevant context from ChromaDB and queries Llama 3.2 for responses.
- **Conversational Memory**: Maintains session history for better interactions.

## Installation
```sh
# Clone the repository
git clone https://github.com/gousemd73/rag_groqapi_fastapi_streamlit.git>
cd rag_groqapi_fastapi_streamlit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Running the Application
### Start the Backend
```sh
uvicorn main:app --reload
```
### Start the Frontend
```sh
streamlit run ui.py
```

## API Endpoints
### Initialize LLM
`GET /init_llm`
```json
{
  "model_name": "llama3-8b-8192",
  "api_key": "your-groq-api-key"
}
```
### Upload File
`POST /upload`
```json
{
  "collection_name": "your-collection-name",
  "file": "your-pdf-or-html-file"
}
```
### Query LLM
`GET /query`
```json
{
  "query": "your-query",
  "n_results": 2,
  "collection_name": "your-collection-name"
}
```

## Acknowledgments
- Inspired by [Ashish Sinha's RAG API](https://github.com/AshishSinha5/rag_api/tree/main)
- Built using **Groq API, FastAPI, Streamlit, LangChain, ChromaDB, and Llama 3.2**.



