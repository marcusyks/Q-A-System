class TextSplitterWrapper:
    """
    Wrap CharacterTextSplitter for Document splitting.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, separator: str = "\n\n"):
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        for d in docs:
            chunks = self.splitter.split_text(d.page_content)
            for i, chunk in enumerate(chunks):
                meta = dict(d.metadata or {})
                meta["_chunk"] = i
                meta["_chunk_chars"] = len(chunk)
                out.append(Document(page_content=chunk, metadata=meta))
        return out