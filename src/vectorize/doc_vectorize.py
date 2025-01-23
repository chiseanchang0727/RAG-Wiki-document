from datetime import datetime
from src.vectorize.base import DocumentVectorizer
from configs.rag_config import RAGConfig
from langchain.schema import Document as LangChainDocument

def vectorize(data: LangChainDocument, config: RAGConfig, chroma_db_path):
    vectorizer = DocumentVectorizer(config=config, chroma_db_path=chroma_db_path)

    if config.vectorize_config.renew_collection and vectorizer.check_collection(config.vectorize_config.collection_name):
        try:
            vectorizer.delete_collection(config.vectorize_config.collection_name)
        except Exception as e:
                raise RuntimeError(f"Failed to delete collection '{config.vectorize_config.collection_name}': {e}")

    print('Start vectorizing.')
    start_time = datetime.now()
    vectorizer.vectorization_and_store(documents=data)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f'Vectorizing ended in {elapsed_time:.2f} seconds.')