from configs.rag_config import RAGConfig
from langchain_huggingface import HuggingFaceEmbeddings
from src.retriever.base import CombineRetriever, DocRetriever
from src.data_io.read_data import read_txt_data
from src.chunk.chunk import chunk_data
from langchain_chroma import Chroma


def fusion_retriever(df_qa, config: RAGConfig, vectorstore_path, chunked_data):
    df = read_txt_data()
    df_list = df.to_dict(orient="records")

    embedding_model = HuggingFaceEmbeddings(model_name=config.vectorize_config.embedding_model_name)

    vectorstore = Chroma(
        persist_directory=vectorstore_path, 
        collection_name=config.vectorize_config.collection_name, 
        embedding_function=embedding_model
    )
    
    for _, row in df_qa.iterrows():
        query = row['Question']
        langchain_retriever = CombineRetriever(vectorstore=vectorstore, doc=df_list, config=config, chunked_data=chunked_data)
        result = langchain_retriever.get_relevant_docs(query)

    return result