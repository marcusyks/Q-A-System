"""
    Unit tests for Document Loader
"""


import os
import shutil
import unittest
from src.document_loader import DocumentLoader, docx, pd, pdfplumber

class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory with test files."""
        self.test_dir = "temp_test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        # Create a subdirectory for recursive loading test
        os.makedirs(os.path.join(self.test_dir, "subdir"), exist_ok=True)

        # .txt file
        self.txt_path = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_path, "w") as f:
            f.write("This is a text file.")

        # .md file (in subdir)
        self.md_path = os.path.join(self.test_dir, "subdir", "test.md")
        with open(self.md_path, "w") as f:
            f.write("# Markdown File")

        # .csv file
        self.csv_path = os.path.join(self.test_dir, "test.csv")
        with open(self.csv_path, "w") as f:
            f.write("id,name\n1,Alice\n2,Bob")

        # .docx file (only if python-docx is installed)
        self.docx_path = os.path.join(self.test_dir, "test.docx")
        if docx:
            doc = docx.Document()
            doc.add_paragraph("This is a docx file.")
            doc.save(self.docx_path)

        # .pdf file (only if pdfplumber is installed)
        self.pdf_path = os.path.join(self.test_dir, "test.pdf")
        if pdfplumber:
            # Create a minimal, valid PDF file by hand for testing
            pdf_content = b"""%PDF-1.7
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/MediaBox[0 0 612 792]/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 55>>stream
BT /F1 12 Tf 50 700 Td (This is a PDF file.) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000010 00000 n
0000000059 00000 n
0000000112 00000 n
0000000214 00000 n
0000000268 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
353
%%EOF"""
            with open(self.pdf_path, "wb") as f:
                f.write(pdf_content)

        # Unsupported file
        self.unsupported_path = os.path.join(self.test_dir, "test.xyz")
        with open(self.unsupported_path, "w") as f:
            f.write("unsupported")

        self.loader = DocumentLoader()

    def tearDown(self):
        """Remove the temporary directory and its contents."""
        shutil.rmtree(self.test_dir)


    """
        Different test cases
    """

    def test_load_txt_file(self):
        docs = self.loader.load(self.txt_path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "This is a text file.")
        self.assertEqual(docs[0].metadata["source"], "test.txt")

    @unittest.skipIf(not pd, "pandas is not installed")
    def test_load_csv_file(self):

        # Checks if CSV loaded
        docs = self.loader.load(self.csv_path)
        self.assertEqual(len(docs), 2)

        # Check contents 
        self.assertEqual(docs[0].page_content, "id: 1 \nname: Alice")
        self.assertEqual(docs[0].metadata["source"], "test.csv")
        self.assertEqual(docs[0].metadata["row"], 0)
        self.assertEqual(docs[1].page_content, "id: 2 \nname: Bob")
        self.assertEqual(docs[1].metadata["row"], 1)

    @unittest.skipIf(not docx, "python-docx is not installed")
    def test_load_docx_file(self):
        docs = self.loader.load(self.docx_path)

        # Checks if DOCX loaded
        self.assertEqual(len(docs), 1)

        # Checks contents
        self.assertEqual(docs[0].page_content, "This is a docx file.")
        self.assertEqual(docs[0].metadata["source"], "test.docx")

    @unittest.skipIf(not pdfplumber, "pdfplumber is not installed")
    def test_load_pdf_file(self):
        docs = self.loader.load(self.pdf_path)

        # Checks if PDF loaded
        self.assertEqual(len(docs), 1)
        
        # Checks contents
        self.assertEqual(docs[0].page_content, "This is a PDF file.")
        self.assertEqual(docs[0].metadata["source"], "test.pdf")
        self.assertEqual(docs[0].metadata["page"], 1)

    def test_load_directory_non_recursive(self):
        docs = self.loader.load(self.test_dir, recursive=False)

        # Should find .txt, .csv, .docx, .pdf but not the .md in the subdir
        # CSV creates 1 docs, PDF creates 1, TXT creates 1, DOCX creates 1.
        expected_docs = 2 # .txt and invalid
        if pd: expected_docs += 1 # .csv 
        if docx: expected_docs += 1 # .docx
        if pdfplumber: expected_docs += 1 # .pdf
        self.assertEqual(len(docs), expected_docs)

    def test_load_directory_recursive(self):
        docs = self.loader.load(self.test_dir, recursive=True)
        # Should find .txt, .csv, .docx, .pdf AND the .md in the subdir
        expected_docs = 3 # .txt and .md and invalid
        if pd: expected_docs += 1 # .csv
        if docx: expected_docs += 1 # .docx
        if pdfplumber: expected_docs += 1 # .pdf
        self.assertEqual(len(docs), expected_docs)

    def test_unsupported_file_type(self):
        docs = self.loader.load(self.unsupported_path)
        self.assertEqual(len(docs), 0)

if __name__ == "__main__":
    unittest.main()