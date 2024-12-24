from pydantic import BaseModel, Field


class VectorizeConfig(BaseModel):
    collection_name: str = Field(default=None)
    renew_collection: bool = Field(default=False, description="True for deleting the content in the collection_name.")
    embedding_model_name: str = Field(default=None)