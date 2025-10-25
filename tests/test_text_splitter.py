"""
Unit tests for TextSplitterWrapper
"""

import unittest
from langchain_core.documents import Document
from src.text_splitter import TextSplitterWrapper, CharacterTextSplitter

@unittest.skipIf(CharacterTextSplitter is None, "langchain_text_splitter is not installed")
class TestTextSplitterWrapper(unittest.TestCase):

    def setUp(self):
        """Set up basic documents for testing."""
        self.doc1 = Document(
            page_content="This is the first sentence. This is the second sentence.",
            metadata={"source": "doc1.txt"}
        )
        self.doc2 = Document(
            page_content="A short document.",
            metadata={"source": "doc2.txt", "author": "tester"}
        )
        self.long_text = "a" * 150 + "\n\n" + "b" * 150
        self.doc3 = Document(
            page_content=self.long_text,
            metadata={"source": "doc3.txt"}
        )

    def test_basic_splitting(self):
        """Test splitting a single document into chunks."""
        splitter = TextSplitterWrapper(chunk_size=30, chunk_overlap=0, separator=" ")
        docs = [self.doc1]
        split_docs = splitter.split_documents(docs)

        self.assertTrue(len(split_docs) > 1)
        self.assertEqual(split_docs[0].page_content, "This is the first sentence.")
        self.assertEqual(split_docs[1].page_content, "This is the second sentence.")

        # Check metadata
        self.assertEqual(split_docs[0].metadata["source"], "doc1.txt")
        self.assertIn("start_index", split_docs[0].metadata)

    def test_splitting_with_overlap(self):
        """Test that chunk overlap is handled correctly."""
        text = "abcdefghijklmnopqrstuvwxyz"
        doc = Document(page_content=text)
        splitter = TextSplitterWrapper(chunk_size=10, chunk_overlap=5, separator="")
        split_docs = splitter.split_documents([doc])

        self.assertEqual(len(split_docs), 5)
        self.assertEqual(split_docs[0].page_content, "abcdefghij")
        self.assertEqual(split_docs[1].page_content, "fghijklmno") # Overlap of 'fghij'
        self.assertEqual(split_docs[2].page_content, "klmnopqrst") 
        self.assertEqual(split_docs[3].page_content, "pqrstuvwxy")
        self.assertEqual(split_docs[4].page_content, "uvwxyz")

    def test_split_multiple_documents(self):
        """Test splitting a list containing multiple documents."""
        splitter = TextSplitterWrapper(chunk_size=100, chunk_overlap=0)
        docs = [self.doc1, self.doc3]
        split_docs = splitter.split_documents(docs)

        # doc1 will be one chunk, doc3 will be two chunks
        self.assertEqual(len(split_docs), 3)

        # Check first doc's chunk
        self.assertEqual(split_docs[0].page_content, self.doc1.page_content)
        self.assertEqual(split_docs[0].metadata["source"], "doc1.txt")

        # Check second doc's chunks
        self.assertEqual(split_docs[1].page_content, "a" * 150)
        self.assertEqual(split_docs[1].metadata["source"], "doc3.txt")
        self.assertEqual(split_docs[1].metadata["start_index"], 0)
        self.assertEqual(split_docs[2].page_content, "b" * 150)
        self.assertEqual(split_docs[2].metadata["source"], "doc3.txt")

    def test_text_shorter_than_chunk_size(self):
        """Test behavior when document text is shorter than chunk_size."""
        splitter = TextSplitterWrapper(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents([self.doc2])

        self.assertEqual(len(split_docs), 1)
        self.assertEqual(split_docs[0].page_content, self.doc2.page_content)
        self.assertEqual(split_docs[0].metadata["source"], "doc2.txt")
        self.assertEqual(split_docs[0].metadata["author"], "tester")
        self.assertEqual(split_docs[0].metadata["start_index"], 0)

    def test_empty_document(self):
        """Test splitting a document with empty page_content."""
        empty_doc = Document(page_content="", metadata={"source": "empty.txt"})
        splitter = TextSplitterWrapper()
        split_docs = splitter.split_documents([empty_doc])

        self.assertEqual(len(split_docs), 0)

if __name__ == "__main__":
    unittest.main()
