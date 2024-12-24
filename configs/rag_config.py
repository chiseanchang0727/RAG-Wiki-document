from pydantic import BaseModel, Field
from configs.chunk_config import ChunkConfig
from configs.vectorize_config import VectorizeConfig


class RAGConfig(BaseModel):
    chunk_config: ChunkConfig = Field(default_factor=ChunkConfig)
    vectorize_config: VectorizeConfig = Field(default_factory=VectorizeConfig)
    top_n: int = Field(default=3)