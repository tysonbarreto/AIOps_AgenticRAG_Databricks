
# 🤖 Agentic RAG Pipeline

A modular, agent-powered Retrieval-Augmented Generation (RAG) application built with LangChain, LangGraph, and Streamlit. This project demonstrates dynamic stateful reasoning, fast vector search, and real-time user interaction—deployed seamlessly via CI/CD.

---

## ⚙️ Architecture Overview

- **LangChain Agent**: Orchestrates tool usage and decision-making across retrieval and generation tasks  
- **FAISS Vector Store**: Enables fast semantic search over embedded documents  
- **LangGraph State Nodes**: Implements stateful logic and branching workflows for agentic control  
- **Streamlit UI**: Provides an interactive front-end for user queries and responses  
- **Gemini 2.5 Flash**: Powers fast, high-quality generation  
- **Google Embedding Model 001**: Converts documents into dense vector representations for retrieval

---

## 🚀 Deployment & CI/CD

- Automated deployment via **GitHub Actions**  
- Uses **bundle assets** to package and push updates to the hosting environment  
- Ensures consistent builds and rapid iteration across environments

---

## 🧠 Capabilities

- 🔍 Semantic search over custom knowledge base  
- 🧩 Agentic reasoning with branching workflows  
- 🖥️ Real-time user interaction via Streamlit  
- ⚡ Fast generation with Gemini 2.5 Flash  
- 🔄 Continuous deployment with GitHub Actions
