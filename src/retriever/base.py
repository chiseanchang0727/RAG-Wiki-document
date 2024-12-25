import numpy as np
import pandas as pd
from pydantic import Field
from configs.rag_config import RAGConfig
from rank_bm25 import BM25Okapi
from langchain_core.retrievers import BaseRetriever


class CombineRetriever:
    def __init__(self, vectorstore, doc:pd.DataFrame, config: RAGConfig):
        self.vectorstore = vectorstore
        self.kw_top_k = config.retriever_config.kw_top_k
        self.vector_k = config.retriever_config.vector_top_k
        self.top_n = config.retriever_config.top_n

        if isinstance(doc, pd.DataFrame):
            self.doc_list = doc.to_dict(orient='records')
        elif isinstance(doc, list):
            self.doc_list = doc
        else:
            raise ValueError("The `doc` parameter must be a pandas DataFrame or a list.")

    def lexical_retrieval(self, query):

        tokenized_docs = [doc['file_content'].split() for doc in self.doc_list]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = query.split()

        scores = bm25.get_scores(tokenized_query)

        # return the doc index and sort in descending order
        top_k_indices = np.argsort(scores)[::-1][:self.kw_top_k]
        top_k_docs = [self.doc_list[i] for i in top_k_indices]

        return top_k_docs
    
    def semantic_retrieval(self, query):

        vector_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=25)

        return [{'file_name': item[0].metadata['file_name'], 'file_content': item[0].page_content} for item in vector_results]
    
    def rrf(self, result_list: list, k=60):
        rrf_scores = {}
        for result in result_list:
            for rank, doc in enumerate(result):
                doc_id = doc['file_name']
                # get the score of doc_id, return 0 if the doc_id is not existed    
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/(k + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        return sorted_results
    
    def get_relevant_docs(self, query):
        lexical_result = self.lexical_retrieval(query)

        semantic_result = self.semantic_retrieval(query)

        combined_result_list = [lexical_result, semantic_result]

        rrf_result = self.rrf(combined_result_list)

        return rrf_result

            

class DocRetriever(BaseRetriever):
    combined_retriever: CombineRetriever = Field(default_factory=CombineRetriever)

    def get_relevant_documents(self, query):
        result = self.combined_retriever.get_relevant_docs(query)
        return result