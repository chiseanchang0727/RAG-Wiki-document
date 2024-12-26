from configs.rag_config import RAGConfig
from src.retriever.base import CombineRetriever, DocRetriever
from langchain_chroma import Chroma


def fusion_retriever(query,  chunked_data, config: RAGConfig, vectorstore, embedding_model):


    langchain_retriever = CombineRetriever(vectorstore=vectorstore, chunked_data=chunked_data, config=config)
    result = langchain_retriever.get_relevant_docs(query)

    return result