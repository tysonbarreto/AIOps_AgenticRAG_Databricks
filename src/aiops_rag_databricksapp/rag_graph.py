from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from aiops_rag_databricksapp.rag_state import RAGState
from aiops_rag_databricksapp.rag_react_node import ReActRAGNodes
from aiops_rag_databricksapp.rag_node import RAGNodes

from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_openai import ChatOpenAI



from dataclasses import dataclass

@dataclass
class RAGGraphBuilder:
    """Builds and manages the LangGraph workflow"""
    retriever: VectorStoreRetriever
    llm: ChatOpenAI
    
    def __post_init__(self):
        self.nodes:RAGNodes = RAGNodes(retriever=self.retriever,llm=self.llm)
        self.graph:CompiledStateGraph=None
    
    def build(self)->CompiledStateGraph:
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        
        builder = StateGraph(RAGState)
        
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        builder.set_entry_point("retriever")
        
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)
        
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question:str)->dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
            
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
    
if __name__=="__main__":
    __all__=["RAGGraphBuilder"]
        
        