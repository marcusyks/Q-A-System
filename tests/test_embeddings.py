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
            chunk_size=35, chunk_overlap=0, separator=". "
        )
        # The resulting chunks are what get passed to the EmbeddingsIndexer.
        original_doc = Document(
            page_content="This is the first sentence. This is the second sentence. This is the third sentence.",
            metadata={"source": "doc1.txt"}
        )
        self.docs = splitter.split_documents([original_doc]) + [Document(page_content="", metadata={"source": "empty.txt"})]

    def test_init(self):
        """Test that the embedder is initialized correctly."""
        mock_embedder = MagicMock()
        indexer = EmbeddingsIndexer(embedder=mock_embedder)
        self.assertIsNotNone(indexer.embedder)
        self.assertEqual(indexer.embedder, mock_embedder)

    def test_aembed_documents(self):
        """Test the asynchronous embedding of documents."""
        # Configure the mock embedder
        mock_embedder = MagicMock()
        mock_embedder.aembed_documents = AsyncMock(return_value=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indexer = EmbeddingsIndexer(embedder=mock_embedder)
        
        # Run the async method
        result = asyncio.run(indexer.aembed_documents(self.docs))

        # Assertions
        self.assertEqual(len(result), 3)
        # Check first item
        self.assertIn("id", result[0])
        self.assertIsInstance(result[0]["id"], str)
        self.assertEqual(result[0]["embedding"], [1.0, 2.0])
        # Check that original metadata is preserved and start_index is added
        self.assertEqual(result[0]["metadata"]["source"], "doc1.txt")
        self.assertIn("start_index", result[0]["metadata"])
        # Check second item
        self.assertEqual(result[1]["embedding"], [3.0, 4.0])
        self.assertEqual(result[1]["metadata"]["source"], "doc1.txt")
        # Check that the underlying async method was called correctly
        mock_embedder.aembed_documents.assert_awaited_once_with(
            ["This is the first sentence.", "This is the second sentence.", "This is the third sentence."]
        )

    @patch('src.embeddings.pinecone')
    def test_upsert_to_pinecone(self, mock_pinecone):
        """Test the upsert functionality to Pinecone."""
        mock_embedder = MagicMock()
        # Prepare mock data and objects
        mock_index = MagicMock()
        mock_pinecone.Index.return_value = mock_index
        mock_pinecone.list_indexes.return_value = ["existing-index"]

        items_to_upsert = [
            {"id": "id1", "embedding": [1.0], "metadata": {"source": "s1"}},
            {"id": "id2", "embedding": [2.0], "metadata": {"source": "s2"}},
        ]

        indexer = EmbeddingsIndexer(embedder=mock_embedder)
        indexer.upsert_to_pinecone(
            pinecone_api_key="key",
            pinecone_environment="env",
            index_name="existing-index",
            items=items_to_upsert,
            namespace="test-ns"
        )

        # Assertions
        mock_pinecone.init.assert_called_once_with(api_key="key", environment="env")
        mock_pinecone.create_index.assert_not_called() # Index already exists
        mock_pinecone.Index.assert_called_once_with("existing-index")

        expected_upsert_data = [
            ("id1", [1.0], {"source": "s1"}),
            ("id2", [2.0], {"source": "s2"}),
        ]
        mock_index.upsert.assert_called_once_with(vectors=expected_upsert_data, namespace="test-ns")


if __name__ == "__main__":
    unittest.main()
