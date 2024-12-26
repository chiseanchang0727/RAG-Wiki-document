import os
import argparse
from dotenv import load_dotenv
from utils import load_config_from_yaml
from configs.rag_config import RAGConfig
from src.data_io.read_data import read_txt_data, read_qa_data
from src.chunk.chunk import chunk_data
from src.vectorize.doc_vectorize import vectorize
from src.retriever.doc_retrieval import doc_retrieval
from llm.config.llm_config import LLMConfig
from llm.bots.qa_bot import QABot

def get_argument():
    args = argparse.ArgumentParser()


    args.add_argument('--chunk', required=False, action='store_true', help='chunk the data.')
    args.add_argument('--vectorize', required=False, action='store_true', help='vectorize the data')
    args.add_argument('--retrieve', nargs='?', const=10, type=int, help='vectorize the data')
    args.add_argument('--all', action='store_true', help='If used with --retrieve, retrieves all rows.')
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

    if args.retrieve is not None:
        df = read_txt_data()
        chunked_data = chunk_data(df, config=config, page_content_column='file_content')
        df_qa = read_qa_data()

        if args.all:
            print("Retrieving ALL rows from df_qa...")
            df_qa_subset = df_qa
        else:
            # Otherwise, retrieve only the first N rows:
            n = args.retrieve  # If user typed `--retrieve` alone, this is 30
            print(f"Retrieving first {n} rows from df_qa...")
            df_qa_subset = df_qa.head(n)


        llm = QABot(config=LLMConfig())
        df_result = doc_retrieval(config=config , llm=llm, vectorstore_path=chroma_db_path, qa_data=df_qa_subset, chunked_data=chunked_data)
        df_result.to_csv('./data/results/retrieval_results.csv', index=False)
        
        print('done')
if __name__ == "__main__":
    main()
