from pydantic import BaseModel, Field
from configs.chunk_config import ChunkConfig
from configs.vectorize_config import VectorizeConfig
from configs.retriever_config import RetrieverConfig

class RAGConfig(BaseModel):
    chunk_config: ChunkConfig = Field(default_factor=ChunkConfig)
    vectorize_config: VectorizeConfig = Field(default_factory=VectorizeConfig)
    retriever_config: RetrieverConfig = Field(default_factory=RetrieverConfig)
