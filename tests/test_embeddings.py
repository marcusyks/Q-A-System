"""
Unit tests for EmbeddingsIndexer
"""

import os
import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from langchain_core.documents import Document
from src.embeddings import EmbeddingsIndexer
from src.text_splitter import TextSplitterWrapper, CharacterTextSplitter

@unittest.skipIf(CharacterTextSplitter is None, "langchain-text-splitter is not available")
class TestEmbeddingsIndexer(unittest.TestCase):

    def setUp(self):
        """Set up documents for testing."""
        # Simulate the full pipeline: a document is loaded and then split.
        splitter = TextSplitterWrapper(
            chunk_size=26, chunk_overlap=0, separator=". "
        )
        # The resulting chunks are what get passed to the EmbeddingsIndexer.
        original_doc = Document(
            page_content="This is the first sentence. This is the second sentence. This is the third sentence.",
            metadata={"source": "doc1.txt", "hash": "abc123"}
        )
        self.docs = splitter.split_documents([original_doc]) + [Document(page_content="", metadata={"source": "empty.txt", "hash": "def456"})]

    def test_init(self):
        """Test that the embedder is initialized correctly."""
        indexer = EmbeddingsIndexer()
        self.assertIsNotNone(indexer)

    @patch('src.embeddings.HuggingFaceEmbeddings')
    def test_aembed_documents(self, mock_huggingface_embeddings):
        """Test the asynchronous embedding of documents."""
        # Configure the mock embedder instance
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.aembed_documents = AsyncMock(return_value=[[1.0], [2.0], [3.0]])
        mock_huggingface_embeddings.return_value = mock_embedder_instance

        indexer = EmbeddingsIndexer()
        result = asyncio.run(indexer.aembed_documents(self.docs))

        # Assertions
        self.assertEqual(len(result), 3)
        # Check first item
        self.assertEqual(result[0]["id"], "abc123-0")
        self.assertEqual(result[0]["values"], [1.0])
        self.assertEqual(result[0]["metadata"]["source"], "doc1.txt")
        self.assertIn("start_index", result[0]["metadata"])

        # Check second item
        self.assertEqual(result[1]["id"], "abc123-28")
        self.assertEqual(result[1]["values"], [2.0])
        self.assertEqual(result[1]["metadata"]["source"], "doc1.txt")
        self.assertIn("start_index", result[1]["metadata"])

        # Check third item
        self.assertEqual(result[2]["id"], "abc123-57")
        self.assertEqual(result[2]["values"], [3.0])

if __name__ == "__main__":
    unittest.main()
