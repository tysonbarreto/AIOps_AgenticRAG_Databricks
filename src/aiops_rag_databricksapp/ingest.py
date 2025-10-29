from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.text import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from dataclasses import dataclass

@dataclass
class DocumentProcessor:
    """
    Initialize document processor
    ---
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    chunk_size:int
    chunk_overlap:int
    
    def __post_init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_from_url(self, url:str)->List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources:List[str])->List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
           
            path = Path("data")
            if path.is_dir():  # PDF directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, or PDF directory."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)
    
if __name__=="__main__":
    __all__=["DocumentProcessor"]