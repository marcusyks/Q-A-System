import logging
from typing import Any, Dict, List
from collections import defaultdict

try:
    from pinecone import ServerlessSpec
except ImportError:
    ServerlessSpec = None

class PineconeManager:
    """
    Manages interactions with a Pinecone index, including upserting,
    and deleting documents based on their hash.
    """
    def __init__(self, pinecone_client: Any, index_name: str):
        """
        Initializes the PineconeManager.

        Args:
            pinecone_client (Any): An initialized Pinecone client instance.
            index_name (str): The name of the Pinecone index to manage.
        """
        if pinecone_client is None:
            raise ValueError("A valid Pinecone client must be provided.")
        if not index_name:
            raise ValueError("A Pinecone index name must be provided.")

        self.pc = pinecone_client
        self.index_name = index_name
        self.index_was_created = False
        self.index = self._get_or_create_index()

    def _get_or_create_index(self, dimension: int = None):
        """
        Retrieves the Pinecone index object. If it doesn't exist, it cannot be
        created without a dimension, so it returns None. The first upsert will
        trigger the creation with the correct dimension.
        """
        if self.pc.has_index(self.index_name):
            logging.info(f"Connecting to existing Pinecone index: {self.index_name}")
            index_description = self.pc.describe_index(name=self.index_name)
            return self.pc.Index(host=index_description.host)
        
        if dimension:
            logging.info(f"Creating new Pinecone index: {self.index_name} with dimension {dimension}")
            self.index_was_created = True # Flag that we just created the index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                deletion_protection="disabled"
            )
            index_description = self.pc.describe_index(name=self.index_name)
            return self.pc.Index(host=index_description.host)
        
        logging.info(f"Index '{self.index_name}' does not exist. It will be created upon first upsert.")
        return None

    def upsert(self, items: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upserts items to the Pinecone index, ensuring atomicity per file hash.
        For each unique file hash in the items, it first deletes all existing
        vectors with that hash and then inserts the new ones.
        """
        if not items:
            logging.warning("No items provided to upsert.")
            return

        # Ensure the index object is initialized before any operations.
        # If the index doesn't exist, this will create it using the dimension
        # from the first item.
        if self.index is None:
            dim = len(items[0].get("embedding", []))
            if dim == 0:
                raise ValueError("Cannot determine embedding dimension from the first item.")
            self.index = self._get_or_create_index(dimension=dim)

        # Group items by their file hash
        items_by_hash = defaultdict(list)
        for item in items:
            if 'hash' in item.get('metadata', {}):
                items_by_hash[item['metadata']['hash']].append(item)

        for file_hash, items_to_upsert in items_by_hash.items():
            # 1. Delete existing vectors, but only if the index wasn't just created.
            if not self.index_was_created:
                self.delete_by_hash(file_hash)

            # 2. Upsert the new vectors for this file hash in batches
            for i in range(0, len(items_to_upsert), batch_size):
                batch = items_to_upsert[i:i + batch_size]
                to_upsert = [{"id": it["id"], "values": it["embedding"], "metadata": it.get("metadata", {})} for it in batch]
                logging.info(f"Upserting batch of {len(to_upsert)} vectors for hash '{file_hash}'.")
                self.index.upsert(vectors=to_upsert, namespace="__default__")

    def delete_by_hash(self, file_hash: str):
        """Deletes all vectors associated with a specific file hash from the index."""
        logging.info(f"Deleting vectors with hash: {file_hash}")
        self.index.delete(
            filter={
                "hash": {"$eq": file_hash}
            }, 
            namespace="__default__"
        )

    def query(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index for the most similar vectors.

        Args:
            query_vector (List[float]): The vector representation of the query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of matching documents with their metadata.
        """
        if self.index is None:
            logging.error("Index has not been initialized. Cannot perform a query.")
            return []
        
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True, namespace="__default__")
        return results.get("matches", [])
