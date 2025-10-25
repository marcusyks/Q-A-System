class EmbeddingsIndexer:
    """
    Create embeddings for Documents and optionally upsert to Pinecone.
    Uses OpenAIEmbeddings by default (expects OPENAI_API_KEY in env or passed).
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # LangChain's OpenAIEmbeddings picks up env var if not provided here.
        if openai_api_key:
            self.embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embedder = OpenAIEmbeddings()

    def embed_documents(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {"id": str, "embedding": List[float], "metadata": dict}
        """
        texts = [d.page_content for d in docs]
        vectors = self.embedder.embed_documents(texts)
        out = []
        for d, v in zip(docs, vectors):
            out.append({"id": str(uuid.uuid4()), "embedding": v, "metadata": d.metadata or {}})
        return out

    def upsert_to_pinecone(self, pinecone_api_key: str, pinecone_environment: str, index_name: str, items: List[Dict[str, Any]], namespace: Optional[str] = None):
        """
        Upsert items to Pinecone index. Each item must have 'id', 'embedding', 'metadata'.
        Creates index if not exists (vector_dim derived from first item).
        """
        if pinecone is None:
            raise RuntimeError("pinecone-client is not installed/importable.")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        if index_name not in pinecone.list_indexes():
            dim = len(items[0]["embedding"])
            pinecone.create_index(name=index_name, dimension=dim)
        index = pinecone.Index(index_name)
        # prepare tuples (id, vector, metadata)
        to_upsert = [(it["id"], it["embedding"], it.get("metadata", {})) for it in items]
        index.upsert(vectors=to_upsert, namespace=namespace)