from aiops_rag_databricksapp.rag_state import RAGState

from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_openai import ChatOpenAI


from dataclasses import dataclass

@dataclass
class RAGNodes:
    """Contains node functions for RAG workflow"""
    retriever:VectorStoreRetriever
    llm:ChatOpenAI
    
    def retrieve_docs(self, state:RAGState) -> RAGState:
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self,state:RAGState)->RAGState:
        """
        Generate answer from retrieved documents node
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        prompt = f"""
                Answer the question only based on the context.

                Context:
                {context}

                Question: {state.question}
        """
        
        response = self.llm.invoke(input=prompt)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )

if __name__=="__main__":
    __all__=["RAGNodes"]