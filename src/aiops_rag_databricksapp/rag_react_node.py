from typing import List, Optional
from aiops_rag_databricksapp.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.runnable import Runnable
from langchain_openai import OpenAI

from dataclasses import dataclass

@dataclass
class ReActRAGNodes:
    """Contains node functions for RAG workflow"""
    retriever: VectorStoreRetriever
    llm: OpenAI
    
    def __post_init__(self):
        self.__agent:Runnable = None

    def retrieve_docs(self, state:RAGState)->RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def __build_tools(self)->List[Tool]:
        """Build retriever + wikipedia tools"""
        
        def retriever_tool_fn(query:str)->str:
            docs:List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki.run,
        )
        return [retriever_tool, wikipedia_tool]

    def __build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self.__agent = create_react_agent(self.llm, tools=tools,prompt=system_prompt)

    def generate_answer(self, state:RAGState)->RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self.__agent is None:
            self.__build_agent()
        
        result = self.__agent.invoke({"messages":[HumanMessage(content=state.question)]})
        messages = result.get("messages",[])
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )

if __name__=="__main__":
    __all__=["ReActRAGNodes"]