from dataclasses import dataclass
from typing import List
from pathlib import Path

from aiops_rag_databricksapp.config import AIConfigSettings, AIConfig
settings = AIConfig()
_ = settings.activate_LLM_environment

from aiops_rag_databricksapp.store import VectorStore
from aiops_rag_databricksapp.ingest import DocumentProcessor
from aiops_rag_databricksapp.rag_graph import RAGGraphBuilder
from aiops_rag_databricksapp.rag_node import RAGNodes
from aiops_rag_databricksapp.rag_react_node import ReActRAGNodes

from langchain_openai import ChatOpenAI

import streamlit as st

import os, sys, time

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic RAG Search",
    page_icon="🔍",
    layout="centered"
)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        config = AIConfig()
        llm = ChatOpenAI(model=config.llm_model)
        doc_processor = DocumentProcessor(
            chunk_size=config.settings.chunk_size,
            chunk_overlap=config.settings.chunk_overlap
        )
        vector_store = VectorStore()
        
        # Use default URLs
        urls = config.settings.default_urls
        
        # Process documents
        documents = doc_processor.process_urls(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = RAGGraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0
    
st.title("🔍 Agentic RAG Document Search")
st.markdown("Ask questions about the loaded documents")

def main():
    init_session_state()
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
                
    st.markdown("---")

    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")


    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")
                
if __name__=="__main__":
    main()