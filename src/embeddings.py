from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from dotenv import load_dotenv
import os

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

load_dotenv()


class EmbeddingsIndexer:
    """
    Create embeddings for Documents and optionally upsert to Pinecone.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initializes the EmbeddingsIndexer.

        Args:
            model_name (str): The name of the sentence-transformer model to use from Hugging Face.
            model_kwargs (dict, optional): Keyword arguments for the HuggingFaceEmbeddings model. 
                                         For example, {'device': 'cuda'}. Defaults to {'device': 'cpu'}.
        """
        if HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "Required packages are not installed. "
                "Please install with 'pip install langchain-community sentence-transformers'"
            )
        
        # Provide a sensible default for model_kwargs if not specified
        effective_model_kwargs = model_kwargs if model_kwargs is not None else {'device': 'cpu'}

        self.embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=effective_model_kwargs)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously creates an embedding for a single query text.
        """
        return await self.embedder.aembed_query(text)

    async def aembed_documents(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {"id": str, "embedding": List[float], "metadata": dict}
        """
        docs_with_content = [doc for doc in docs if doc.page_content]
        if not docs_with_content:
            return []

        texts = [d.page_content for d in docs_with_content]
        vectors = await self.embedder.aembed_documents(texts)

        # Create a unique ID for each chunk by combining file hash and chunk's start index
        # This prevents overwriting chunks from the same document in Pinecone.
        return [
            {
                "id": f"{d.metadata.get('hash')}-{d.metadata.get('start_index', 0)}-{d.metadata.get('page', 0)}-{d.metadata.get('row', 0)}",
                # Page and row are for PDFs and CSVs respectively
                "embedding": v,
                # Ensure the text content is stored in the metadata for retrieval
                "metadata": {
                    **(d.metadata or {}),
                    "page_content": d.page_content
                },
            }
            for d, v in zip(docs_with_content, vectors)
        ]
