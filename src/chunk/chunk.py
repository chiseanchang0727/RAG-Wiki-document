import pandas as pd
from src.chunk.base import DocumentChunk
from configs.rag_config import RAGConfig



def chunk_data(data:pd.DataFrame, config:RAGConfig, page_content_column):

    chunker = DocumentChunk(
        data, 
        chunk_method=config.chunk_config.chunk_method, 
        chunk_size=config.chunk_config.chunk_size, 
        chunk_overlap=config.chunk_config.chunk_overlap, 
        page_content_column=page_content_column
    )


    chunked_data = chunker.chunk()

    print(f'{len(data)} data has been chunked into {len(chunked_data)} pieces.')

    return chunked_data