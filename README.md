# TN Agri RAG Assistant

A GenAI-powered RAG chatbot that answers user questions from Tamil Nadu Government agriculture scheme pages using LangChain, OpenAI embeddings, and FAISS.

## Project Description
This application collects TN scheme content from official links, splits it into chunks, creates embeddings, stores them in a FAISS vector database, and provides grounded answers through a Streamlit chat interface.

## Features
- RAG pipeline with LangChain
- Source-grounded responses from TN scheme pages
- FAISS vector search for fast retrieval
- Streamlit SaaS-style UI
- Tamil/English response option
- Quick keyword buttons for common queries
- Source citation display

## Tech Stack / Tools Used
- Python 3.13
- LangChain
- OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)
- FAISS (`faiss-cpu`)
- Streamlit
- BeautifulSoup / Requests
- LangSmith (tracing/observability)

## Folder Structure
```text
TN-AGRI-RAG-Buildathon/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── scheme_urls.txt
│   ├── raw_documents.jsonl
│   └── chunks.jsonl
├── vectorstore/
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
└── assets/
    ├── TN Logo.png
    └── farm_bg.jpg
