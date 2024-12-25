import os
import argparse
from dotenv import load_dotenv
from utils import load_config_from_yaml
from configs.rag_config import RAGConfig
from src.data_io.read_data import read_txt_data
from src.chunk.chunk import chunk_data
from src.vectorize.doc_vectorize import vectorize
from src.retriever.fusion_retrieval import fusion_retriever

def get_argument():
    args = argparse.ArgumentParser()


    args.add_argument('--chunk', required=False, action='store_true', help='chunk the data.')
    args.add_argument('--vectorize', required=False, action='store_true', help='vectorize the data')
    args.add_argument('--retriever', required=False, action='store_true', help='vectorize the data')

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

    if args.retriever:
        # df = read_txt_data()
        # concat qa_pairs
        query= 'Was Abraham Lincoln the sixteenth President of the United States?'
        fusion_retriever(query=query, config=config, vectorstore_path=chroma_db_path)

if __name__ == "__main__":
    main()
