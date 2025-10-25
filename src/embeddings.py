import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None
    
try:
    import pinecone
except ImportError:
    pinecone = None

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

    async def aembed_documents(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {"id": str, "embedding": List[float], "metadata": dict}
        """
        docs_with_content = [doc for doc in docs if doc.page_content]
        if not docs_with_content:
            return []

        texts = [d.page_content for d in docs_with_content]
        vectors = await self.embedder.aembed_documents(texts)

        return [{"id": str(uuid.uuid4()), "embedding": v, "metadata": d.metadata or {}} for d, v in zip(docs_with_content, vectors)]

    def upsert_to_pinecone(self, pinecone_api_key: str, pinecone_environment: str, index_name: str, items: List[Dict[str, Any]], namespace: Optional[str] = None):
        """
        Upsert items to Pinecone index. Each item must have 'id', 'embedding', 'metadata'.
        Creates index if not exists (vector_dim derived from first item).
        """
        if pinecone is None:
            raise RuntimeError("pinecone is not installed/importable.")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        if index_name not in pinecone.list_indexes():
            dim = len(items[0]["embedding"])
            pinecone.create_index(name=index_name, dimension=dim)
        index = pinecone.Index(index_name)
        # prepare tuples (id, vector, metadata)
        to_upsert = [(it["id"], it["embedding"], it.get("metadata", {})) for it in items]
        index.upsert(vectors=to_upsert, namespace=namespace)