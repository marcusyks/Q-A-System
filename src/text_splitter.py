from typing import List
from langchain_core.documents import Document

try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    CharacterTextSplitter = None

class TextSplitterWrapper:
    """
    Wrap CharacterTextSplitter for Document splitting.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100, separator: str = "\n\n"):
        if CharacterTextSplitter is None:
            raise RuntimeError("langchain-text-splitters is not installed. Please install with 'pip install langchain-text-splitters'")
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Adds character start index to metadata
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
