import pandas as pd
from langchain.schema import Document as LangChainDocument
from configs.rag_config import RAGConfig
from langchain_huggingface import HuggingFaceEmbeddings
from src.retriever.fusion_retrieval import fusion_retriever
from langchain_chroma import Chroma


def doc_retrieval(config:RAGConfig , vectorstore_path, qa_data: pd.DataFrame, chunked_data:LangChainDocument):

        embedding_model = HuggingFaceEmbeddings(model_name=config.vectorize_config.embedding_model_name)

        vectorstore = Chroma(
            persist_directory=vectorstore_path, 
            collection_name=config.vectorize_config.collection_name, 
            embedding_function=embedding_model
        )

        df = []
        for _, row in qa_data.iterrows():
            query = row['Question']
            row['retrieved_docs'] = fusion_retriever(
                  query,  chunked_data, config=config, vectorstore=vectorstore, embedding_model=embedding_model
            )

            df.append(row)
            
        df = pd.DataFrame(df)

        return df