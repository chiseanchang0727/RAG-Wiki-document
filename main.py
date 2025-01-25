import os
import argparse
import chromadb.cli
from dotenv import load_dotenv
from utils import load_config_from_yaml
from configs.rag_config import RAGConfig
from src.data_io.read_data import read_txt_data, read_qa_data
from src.chunk.chunk import chunk_data
from src.vectorize.doc_vectorize import vectorize
from src.retriever.doc_retrieval import doc_retrieval
from chromadb import PersistentClient
import chromadb
from chromadb.config import Settings

def get_argument():
    args = argparse.ArgumentParser()


    args.add_argument('--chunk', required=False, action='store_true', help='chunk the data.')
    args.add_argument('--vectorize', required=False, action='store_true', help='vectorize the data')
    args.add_argument('--retrieve', required=False, action='store_true', help='retrieve the results.')
    args.add_argument('--deletecollection', required=False, help='delete the entered collection name.')

    return args.parse_args()

def main():
    load_dotenv()

    chroma_db_path = os.getenv('CHROMA_DB_PATH')
    args = get_argument()

    config_path = './test_configs/test.yaml'
    config = load_config_from_yaml(config_path, RAGConfig)

    if args.chunk:
        df = read_txt_data()
        chunked_data = chunk_data(df, config=config, page_content_column='file_content')
    
    if args.vectorize:
        df = read_txt_data()
        chunked_data = chunk_data(df, config=config, page_content_column='file_content')
        vectorize(data=chunked_data, config=config, chroma_db_path=chroma_db_path)

    if args.retrieve:
        df = read_txt_data()
        chunked_data = chunk_data(df, config=config, page_content_column='file_content')
        df_qa = read_qa_data()

        result = doc_retrieval(config=config , vectorstore_path=chroma_db_path, qa_data=df_qa, chunked_data=chunked_data)
        result
        print(1)

    if args.deletecollection:

        collection_name = args.deletecollection
        try:
            chroma_client = PersistentClient(path=os.getenv('CHROMA_DB_PATH'), )
            if collection_name != 'all':
                chroma_client.delete_collection(collection_name)
                print(f"Collection '{collection_name}' has been deleted.")
            else: 
                collections = chroma_client.list_collections()
                for collection in collections:
                    chroma_client.delete_collection(collection.name)
                    print(f"Deleted collection: {collection.name}")
            
        except Exception as e:
            print(f"Failed to delete collection '{collection_name}'. Error: {e}")

if __name__ == "__main__":
    main()
