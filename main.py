from dataclasses import dataclass
from typing import List
from pathlib import Path

from aiops_rag_databricksapp.config import AIConfigSettings, AIConfig
settings = AIConfig()
settings.activate_LLM_environment

from aiops_rag_databricksapp.store import VectorStore
from aiops_rag_databricksapp.ingest import DocumentProcessor
from aiops_rag_databricksapp.rag_graph import RAGGraphBuilder
from aiops_rag_databricksapp.rag_node import RAGNodes
from aiops_rag_databricksapp.rag_react_node import ReActRAGNodes

from langchain_openai import ChatOpenAI

@dataclass
class AgenticRAG:
    """Main Agentic RAG application"""
    urls:List[str]
    
    def __post_init__(self):
        self.config = AIConfig()

        self.urls = self.urls or self.config.settings.default_urls
        self.llm_model:str = self.config.llm_model
        self.llm:ChatOpenAI = ChatOpenAI(model=self.llm_model)
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.settings.chunk_size,
            chunk_overlap=self.config.settings.chunk_overlap
        )
        self.vector_store = VectorStore()
    
    def initialise_vectorestore(self):
        """Setup vector store with processed documents"""
        print(f"📄 Processing {len(self.urls)} URLs...")
        documents = self.doc_processor.process_urls(self.urls)
        print(f"📊 Created {len(documents)} document chunks")
        print("🔍 Creating vector store...")
        self.vector_store.create_vectorstore(documents)
        print("🔍 Vector store initialised...")
        
    
    def build_agentic_rag_graph(self, question:str):
        self.graph_builder=RAGGraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()
        print("✅ System initialized successfully!\n")
        return self.graph_builder.run(question=question)
    
    def ask(self, question:str)->str:
        """
        Ask a question to the RAG system
        
        Args:
            question: User question
            
        Returns:
            Generated answer
        """
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")
        result = self.build_agentic_rag_graph(question=question)
        answer = result['answer']
        print(f"✅ Answer: {answer}\n")
        return answer   
    
    def agentic_chat(self):
        print("💬 Agentic Chat Mode - Type 'quit' to exit\n")
        
        while True:
            question = input("Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question:
                self.ask(question)
                print("-" * 80 + "\n")
    
def main():
    """Main function"""
    # Example: Load URLs from file if exists
    urls_file = Path("data/urls.txt")
    urls = None
    
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    # Initialize RAG system
    rag = AgenticRAG(urls=urls)
    rag.initialise_vectorestore()
    
    # Example questions
    example_questions = [
        "What is the concept of agent loop in autonomous agents?",
        "What are the key components of LLM-powered agents?",
        "Explain the concept of diffusion models for video generation."
    ]
    
    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")
    
    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")
    
    # Optional: Run interactive mode
    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.agentic_chat()

if __name__ == "__main__":
    main()