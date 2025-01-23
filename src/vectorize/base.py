import os
from configs.rag_config import RAGConfig
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentVectorizer:
    def __init__(self, config: RAGConfig, chroma_db_path):
        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path)
            print(f'Created folder: {chroma_db_path}')

        self.vector_db_path = chroma_db_path
        self.vector_store = None

        self.collection_name = config.vectorize_config.collection_name
        self.embedding_model_name = config.vectorize_config.embedding_model_name

    def check_collection(self, collection_name):
        try:
            # Initialize Chroma Vector Store
            self.vector_store = Chroma(persist_directory=self.vector_db_path, collection_name=collection_name)
            if self.vector_store._collection.count() != 0:
                return True
            else:
                return False
        except Exception as e:
            print(f"Failed to retrieve collection '{self.collection_name}'. Error: {e}")
            return None

    def vectorization_and_store(self, documents):
        # Use HuggingFaceEmbeddings for embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.vector_store = Chroma.from_documents(  
            documents=documents,
            persist_directory=self.vector_db_path, 
            collection_name=self.collection_name, 
            embedding=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{self.collection_name}' has been created and stored.")

    def delete_collection(self):
        try:
            # Remove the entire collection from Chroma storage
            self.vector_store = Chroma(persist_directory=self.vector_db_path, collection_name=self.collection_name)
            self.vector_store.delete_collection()
            print(f"Collection '{self.collection_name}' has been deleted.")
        except Exception as e:
            print(f"Failed to delete collection '{self.collection_name}'. Error: {e}")
