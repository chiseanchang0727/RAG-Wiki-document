from pydantic import BaseModel, Field

class ChunkConfig(BaseModel):
    chunk_method: str = Field(default=None)
    chunk_size: int = Field(default=None)
    chunk_overlap: int = Field(default=None)