from typing import List
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

class TextSplitterWrapper:
    """
    Wrap RecursiveCharacterTextSplitter for Document splitting.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("langchain-text-splitters is not installed. Please install with 'pip install langchain-text-splitters'")
        # Using RecursiveCharacterTextSplitter is generally recommended for text.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Adds character start index to metadata
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
