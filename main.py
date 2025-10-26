import argparse
import asyncio
from langchain_core.output_parsers import StrOutputParser
import logging
import os

from dotenv import load_dotenv

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

try:
    import torch
    from transformers import pipeline
    from langchain_huggingface.llms import HuggingFacePipeline
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    HuggingFacePipeline = None

from src.database import PineconeManager
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingsIndexer
from src.text_splitter import TextSplitterWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_indexing(path: str, recursive: bool):
    """
    Main execution function to load, split, embed, and upsert documents.
    """
    if Pinecone is None:
        raise ImportError("The 'pinecone-client' library is not installed. Please install it with 'pip install pinecone-client'.")

    # --- 1. Initialize clients ---
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable not set.")
    
    pc = Pinecone(api_key=pinecone_api_key)
    db_manager = PineconeManager(pinecone_client=pc, index_name=pinecone_index_name)
    loader = DocumentLoader()
    splitter = TextSplitterWrapper()
    indexer = EmbeddingsIndexer()

    # --- 2. Load documents ---
    logging.info(f"Loading documents from path: {path}")
    docs, file_count = loader.load(path, recursive=recursive)
    logging.info(f"Found {file_count} file(s) to process.")
    if not docs:
        logging.warning("No document objects were loaded. Exiting.")
        return
    logging.info(f"Loaded content into {len(docs)} document objects.")

    # --- 3. Split documents ---
    logging.info("Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    if not chunks:
        logging.warning("No chunks were created from the documents. Exiting.")
        return
    logging.info(f"Created {len(chunks)} chunks.")

    # --- 4. Generate embeddings ---
    logging.info("Generating embeddings for chunks...")
    items_to_upsert = await indexer.aembed_documents(chunks)
    logging.info(f"Generated embeddings for {len(items_to_upsert)} items.")

    # --- 5. Upsert to Pinecone ---
    logging.info("Upserting embeddings to Pinecone...")
    db_manager.upsert(items=items_to_upsert)
    logging.info("Upsert complete. Process finished successfully.")

async def run_query_mode():
    """
    Starts an interactive query loop to chat with the indexed documents.
    """
    if HuggingFacePipeline is None:
        raise ImportError("Required packages for query mode are not installed. Please run 'pip install langchain-community accelerate'.")

    # --- 1. Initialize clients and models ---
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if not all([pinecone_api_key, pinecone_index_name]):
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in your environment.")

    pc = Pinecone(api_key=pinecone_api_key)
    db_manager = PineconeManager(pinecone_client=pc, index_name=pinecone_index_name)
    indexer = EmbeddingsIndexer()

    logging.info("Initializing local LLM... (This may take a moment)")
    # Use a local model for question answering. TinyLlama is small and has a 2048 token context window.
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    task = "text-generation"
    
    hf_pipeline = pipeline(
        task,
        model=model_id,
        device_map="auto", # Automatically use GPU if available
        model_kwargs={"torch_dtype": torch.bfloat16} # Use bfloat16 for memory efficiency
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    logging.info("LLM initialized successfully.")

    # --- 2. Define the retriever of relevant content ---
    async def retriever(query: str):
        query_embedding = await indexer.aembed_query(query)
        matches = db_manager.query(query_vector=query_embedding, top_k=5) # can be increased based on LLM size

        # Sort matches to provide a more coherent context.
        # It sorts by source file, then by page number, then row, then start_index.
        sorted_matches = sorted(
            matches,
            key=lambda m: (
                m['metadata'].get('source', ''),
                m['metadata'].get('page', 0),
                m['metadata'].get('row', 0),
                m['metadata'].get('start_index', 0)
            )
        )

        # Format context for the LLM
        context = "\n---\n".join([match['metadata']['page_content'] for match in sorted_matches])
        return context

    # --- 3. Define the RAG chain using LCEL ---
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Inputs -> Prompt -> LLM -> Output
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 4. Start interactive loop ---
    logging.info("Entering query mode. Type 'exit' to quit.")
    while True:
        try:
            question = input("Ask a question: ")
            if question.lower() == 'exit':
                break
            if not question.strip():
                continue
            
            print("\nThinking...")
            answer = await rag_chain.ainvoke(question)
            print("\nAnswer:", answer)
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            break
    logging.info("Exiting query mode.")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Document Indexing and Querying CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-parser for the "index" command
    index_parser = subparsers.add_parser("index", help="Load, embed, and index documents.")
    index_parser.add_argument("path", type=str, help="The path to the file or directory to process.")
    index_parser.add_argument("--recursive", action="store_true", help="Process directories recursively.")

    # Sub-parser for the "query" command
    query_parser = subparsers.add_parser("query", help="Start an interactive session to query the documents.")

    args = parser.parse_args()

    if args.command == "index":
        asyncio.run(run_indexing(args.path, args.recursive))
    elif args.command == "query":
        asyncio.run(run_query_mode())
