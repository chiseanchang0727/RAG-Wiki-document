import pandas as pd
from langchain.schema import Document as LangChainDocument
from configs.rag_config import RAGConfig
from langchain_huggingface import HuggingFaceEmbeddings
from src.retriever.fusion_retrieval import fusion_retriever
from langchain_chroma import Chroma


def doc_retrieval(config:RAGConfig , llm, vectorstore_path, qa_data: pd.DataFrame, chunked_data:LangChainDocument):

        embedding_model = HuggingFaceEmbeddings(model_name=config.vectorize_config.embedding_model_name)

        vectorstore = Chroma(
            persist_directory=vectorstore_path, 
            collection_name=config.vectorize_config.collection_name, 
            embedding_function=embedding_model
        )

        df = []
        for idx, row in qa_data.iterrows():
            target = row['ArticleTitle']
            question = row['Question']
            print(f"Q: {question}")

            file_name, llm_answer = fusion_retriever(
                  target, question,  chunked_data, llm, config=config, vectorstore=vectorstore
            )

            row['retrieved_docs'] = file_name 
            row['llm_answer'] = llm_answer

            print(f'Filename: {file_name}')
            print(f'A: {llm_answer}')
            print(f"{idx} is done.")

            df.append(row)
        df = pd.DataFrame(df)

        return df