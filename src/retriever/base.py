import numpy as np
import pandas as pd
from pydantic import Field
from configs.rag_config import RAGConfig
from rank_bm25 import BM25Okapi
from langchain_core.retrievers import BaseRetriever


class CombineRetriever:
    def __init__(self, vectorstore, chunked_data, config: RAGConfig):
        self.vectorstore = vectorstore
        self.kw_top_k = config.retriever_config.kw_top_k
        self.vector_k = config.retriever_config.vector_top_k
        self.top_n = config.retriever_config.top_n
        self.chunked_data = chunked_data


    def lexical_retrieval(self, query):

        tokenized_docs = [doc.page_content.split() for doc in self.chunked_data]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = query.split()

        scores = bm25.get_scores(tokenized_query)

        # return the doc index and sort in descending order
        top_k_indices = np.argsort(scores)[::-1][:self.kw_top_k]
        top_k_docs = [self.chunked_data[i] for i in top_k_indices]

        return [
            {
                'file_name': item.metadata['file_name'], 
                'uuid': item.metadata['uuid'],
                'file_content': item.page_content
            } 
            for item in top_k_docs
        ]
    
    def semantic_retrieval(self, query):

        vector_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.vector_k)

        return [
            {
                'file_name': item[0].metadata['file_name'], 
                'uuid': item[0].metadata['uuid'],
                'file_content': item[0].page_content
            } 
            for item in vector_results
        ]
    
    def rrf(self, result_list: list, k=60):
        rrf_scores = {}
        for result in result_list:
            for rank, doc in enumerate(result):
                doc_uu_id = doc['uuid']
                # get the score of doc_id, return 0 if the doc_id is not existed    
                rrf_scores[doc_uu_id] = rrf_scores.get(doc_uu_id, 0) + 1/(k + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        return [uuid for uuid, _ in sorted_results[:self.top_n]]
    
    def get_relevant_docs(self, query):
        lexical_result = self.lexical_retrieval(query)

        semantic_result = self.semantic_retrieval(query)

        
        combined_result_list = [lexical_result, semantic_result]
        rrf_result_uuid_list = self.rrf(combined_result_list)

        combined_result = lexical_result + semantic_result
        final_result =  [
            {
                'file_content': item['file_content'],
                'file_name': item['file_name']
            }
            for item in combined_result
            if item['uuid'] in rrf_result_uuid_list
        ]
    
        if final_result:
            return final_result[0]['file_name'], final_result[0]['file_content']
        else:
            return None, None


class DocRetriever(BaseRetriever):
    combined_retriever: CombineRetriever = Field(default_factory=CombineRetriever)

    def get_relevant_documents(self, query):
        result = self.combined_retriever.get_relevant_docs(query)
        return result