"""
Document ingestion
"""

# Document parsing libs
import logging
import os
from typing import List
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pandas as pd
except ImportError:
    pd = None

class DocumentLoader:
    """
    Load files or directories into langchain.schema.Document objects.
    Supported file types: .txt, .md, .pdf, .docx, .csv
    """

    SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".csv"}

    # Main loading function
    def load(self, path: str, recursive: bool = False) -> List[Document]:
        base_dir = os.path.dirname(path) if os.path.isfile(path) else path
        if os.path.isdir(path):
            docs = []
            for root, _, files in os.walk(path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in self.SUPPORTED:
                        full_path = os.path.join(root, f)
                        docs.extend(self._load_file(full_path, base_dir))
                if not recursive:
                    break
            return docs
        else:
            return self._load_file(path, base_dir)

    # Helper function to differentiate loading based on file type
    def _load_file(self, path: str, base_dir: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        source_id = os.path.relpath(path, base_dir).replace("\\", "/") # Use forward slashes for consistency
        if ext in {".txt", ".md"}:
            return [self._load_text(path, source_id)]
        if ext == ".pdf":
            return self._load_pdf(path, source_id)
        if ext == ".docx":
            return [self._load_docx(path, source_id)]
        if ext == ".csv":
            return self._load_csv(path, source_id)
        return []
    

    """
        Loaders for different file types
    """

    # loading for .txt. file
    def _load_text(self, path: str, source_id: str) -> Document:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        meta = {"source": source_id}
        return Document(page_content=content, metadata=meta)
    
    # loading for .pdf file
    def _load_pdf(self, path: str, source_id: str) -> List[Document]:
        meta_base = {"source": source_id}
        pages = []
        if pdfplumber:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        meta = dict(meta_base)
                        meta["page"] = i + 1
                        pages.append(Document(page_content=txt, metadata=meta))
        else:
            logging.warning(
                "pdfplumber is not installed. Cannot parse PDF file: %s. "
                "Install with 'pip install pdfplumber'.",
                path
            )
        return pages
    
    # loading for .docx file
    def _load_docx(self, path: str, source_id: str) -> Document:
        text = ""
        if docx:
            d = docx.Document(path)
            paragraphs = [p.text for p in d.paragraphs if p.text and p.text.strip()]
            text = "\n".join(paragraphs)
        else:
            logging.warning(
                "python-docx is not installed. Cannot parse DOCX file: %s. "
                "Install with 'pip install python-docx'.", path
            )
        meta = {"source": source_id}
        return Document(page_content=text, metadata=meta)

    # loading for .csv file
    def _load_csv(self, path: str, source_id: str) -> List[Document]:
        docs = []
        if pd:
            try:
                df = pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
                # Create one document per row (join columns)
                for i, row in df.iterrows():
                    content = " \n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                    meta = {"source": source_id, "row": int(i)}
                    docs.append(Document(page_content=content, metadata=meta))
            except Exception as e:
                logging.error("Failed to parse CSV with pandas: %s. Error: %s", path, e)
                # Fallback to raw text reading on pandas error
                return [self._load_text(path, source_id)]
        else:
            logging.warning(
                "pandas is not installed. Reading CSV as raw text file: %s. "
                "Install with 'pip install pandas'.",
                path
            )
            # Fallback to raw text if pandas is not available
            return [self._load_text(path, source_id)]
        return docs