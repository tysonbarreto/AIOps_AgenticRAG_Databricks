from typing import Dict, List

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from dataclasses import dataclass


@dataclass
class VectorStore:
    embedding = OpenAIEmbeddings()
    vectorstore = None
    retriever = None

    def create_vectorstore(self, documents:List[Document]):
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to embed
        """
        self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embedding)
        self.retriever = self.vectorstore.as_retriever()
        
    def get_retriever(self):
        """
        Get the retriever instance
        
        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)