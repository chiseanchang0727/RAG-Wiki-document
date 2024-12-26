from configs.enum import ChunkMethod
from langchain_community.document_loaders import DataFrameLoader
from langchain.schema import Document as LangChainDocument
from llama_index.core import Document as LlamaIndexDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
import uuid

class DocumentChunk:
    def __init__(self, df, chunk_method, chunk_size, chunk_overlap, page_content_column):
        self.chunk_method = chunk_method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        chunk_method_category = ChunkMethod(self.chunk_method).get_category()
        self.loaded_data = self.__data_encapsulate(df, chunk_method_category, page_content_column)
    
    def __data_encapsulate(self, df, chunk_method_category, page_content_column):

        if chunk_method_category == 'langchain':
        
            data_loader = DataFrameLoader(df, page_content_column=page_content_column)
            return data_loader.load()
        
        elif chunk_method_category == 'llamaindex':
            return [
                LlamaIndexDocument(
                    text=row[page_content_column], 
                    metadata={key: value for key, value in row.items() if key != page_content_column}
                ) 
                for _, row in df.iterrows()
            ]
    @staticmethod
    def __create_uuid(chunked_data):
        namespace = uuid.NAMESPACE_DNS
        uuid_set = [str(uuid.uuid5(namespace, str(idx))) for idx in range(len(chunked_data))]
        for idx, item in enumerate(chunked_data):
            item.metadata.update({"uuid": uuid_set[idx]}) 

        return chunked_data
        

    def recursive_splitter(self):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunked_data = text_splitter.transform_documents(self.loaded_data)

        chunked_data = self.__create_uuid(chunked_data)

        return chunked_data
    
    def sentence_splitter(self):
        splitter = SentenceSplitter(
        chunk_size = self.chunk_size,
        chunk_overlap = self.chunk_overlap,
        )
        chunked_data = splitter.get_nodes_from_documents(self.loaded_data)

        chunked_data = self.__create_uuid(chunked_data)
        
        # Convert nodes into LangChain Document objects for storage
        chunked_data_in_doc = [LangChainDocument(page_content=node.text, metadata=node.metadata) for node in chunked_data]

        return chunked_data_in_doc


    def chunk(self):

        if self.chunk_method == ChunkMethod.RECURSIVE_SPLITTER.__str__():
            return self.recursive_splitter()
        elif self.chunk_method == ChunkMethod.SENTENCE_SPLITTER.__str__():
            return self.sentence_splitter()
