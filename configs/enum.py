from enum import Enum


class ChunkMethod(Enum):
    SENTENCE_SPLITTER = 'sentence_splitter'
    SENTENCE_WINDOW = 'sentence_window'
    SEMANTIC_SPLITTER = 'semantic_splitter'
    RECURSIVE_SPLITTER = 'recursive_splitter'

    def __str__(self):
        return self.value
    
    def get_category(self):
        """Return the category based on the library the chunk method belongs to."""
        if self in {ChunkMethod.SENTENCE_SPLITTER, ChunkMethod.SENTENCE_WINDOW}:
            return "llamaindex"
        elif self in {ChunkMethod.SEMANTIC_SPLITTER, ChunkMethod.RECURSIVE_SPLITTER}:
            return "langchain"
        else:
            raise ValueError('Invalide chunk method.')


class EmbeddingModel(Enum):
    Distiluse_Base_MultiLingual_V1 = 'sentence-transformers/distiluse-base-multilingual-cased-v1'

    def __str__(self):
        return self.value
