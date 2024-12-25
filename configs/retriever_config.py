from pydantic import BaseModel, Field


class RetrieverConfig(BaseModel):
    kw_top_k: int = Field(default=None, description="Number of top results to return from BM25 (keyword-based search).")
    vector_top_k: int = Field(default=None, description="Number of top results to return from vector-based similarity search.")
    rrf_k: int = Field(default=None, description="Parameter for Reciprocal Rank Fusion (RRF), controlling the weight of rank position in combined results.")
    top_n: int = Field(default=None, description="Final number of top results to return after combining retrieval methods.")
