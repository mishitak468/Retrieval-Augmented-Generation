"""
Vector Database Implementation using FAISS
Builds and searches a vector index for semantic document retrieval
"""

import json
import numpy as np
import faiss
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer


class VectorDatabase:
    def __init__(self, embedding_dim=768):
        """
        Initialize vector database

        Args:
            embedding_dim: Dimension of embedding vectors (768 for BGE)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.id_to_doc = {}  # Map document ID to full document data

    def load_documents(self, filepath):
        """Load preprocessed documents with embeddings"""
        print(f"Loading documents from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        # Create ID mapping for quick lookup
        self.id_to_doc = {doc['id']: doc for doc in self.documents}
        print(f"Loaded {len(self.documents)} documents")

    def build_index(self, index_type='flat'):
        """
        Build FAISS index from loaded documents

        Args:
            index_type: 'flat' for exact search, 'ivf' for approximate search
        """
        if not self.documents:
            raise ValueError(
                "No documents loaded. Call load_documents() first.")

        print(f"Building {index_type} index...")

        # Extract embeddings and convert to numpy array
        embeddings = np.array([doc['embedding']
                              for doc in self.documents], dtype='float32')

        if index_type == 'flat':
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == 'ivf':
            # Approximate search using IVF (Inverted File Index)
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add embeddings to index
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors")

    def search(self, query_embedding, k=5, remove_duplicates=True):
        """
        Search for k most similar documents

        Args:
            query_embedding: Query vector (768-dim numpy array or list)
            k: Number of results to return
            remove_duplicates: If True, filter out duplicate document IDs

        Returns:
            List of tuples: (document_id, distance, document_text)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Ensure query is 2D numpy array of shape (1, 768)
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype='float32')

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Perform search
        distances, indices = self.index.search(
            query_embedding, k if not remove_duplicates else k * 3)

        # Extract results
        results = []
        seen_ids = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            doc = self.documents[idx]
            doc_id = doc['id']

            # Skip duplicates if requested
            if remove_duplicates:
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

            results.append((doc_id, float(dist), doc['text']))

            if len(results) >= k:
                break

        return results

    def search_by_text(self, query_text, k=5, remove_duplicates=True):
        """
        Search using text query (encodes text to embedding first)

        Args:
            query_text: Natural language query
            k: Number of results to return
            remove_duplicates: If True, filter out duplicate document IDs

        Returns:
            List of tuples: (document_id, distance, document_text)
        """
        # Encode query text
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        query_embedding = model.encode([query_text])[0]

        return self.search(query_embedding, k, remove_duplicates)


def test_vector_db():
    """Test the vector database with a self-similarity check"""
    print("\n=== Testing Vector Database ===")

    # Initialize and load
    db = VectorDatabase()
    db.load_documents('preprocessed_documents.json')
    db.build_index(index_type='flat')

    # Test 1: Self-similarity check
    print("\nTest 1: Self-similarity check")
    test_doc_idx = 42 if len(db.documents) > 42 else 0
    test_doc = db.documents[test_doc_idx]
    test_embedding = np.array(test_doc['embedding'], dtype='float32')

    results = db.search(test_embedding, k=5, remove_duplicates=False)

    print(f"\nQuery: Document ID {test_doc['id']}")
    print(f"Text preview: {test_doc['text'][:100]}...")
    print("\nTop 5 results:")
    for rank, (doc_id, dist, text) in enumerate(results, 1):
        print(f"{rank}. ID={doc_id}, Distance={dist:.4f}")
        print(f"   Text: {text[:80]}...")

    # Verify top result
    if results[0][0] == test_doc['id'] and results[0][1] < 0.01:
        print("\n✓ Test PASSED: Document found itself with distance ≈ 0")
    else:
        print("\n✗ Test FAILED: Self-similarity check failed")

    # Test 2: Text query search
    print("\n\nTest 2: Semantic search with text query")
    query = "What causes animals to lose fur?"
    print(f"Query: '{query}'")
    results = db.search_by_text(query, k=3, remove_duplicates=True)

    print("\nTop 3 results:")
    for rank, (doc_id, dist, text) in enumerate(results, 1):
        print(f"{rank}. ID={doc_id}, Distance={dist:.2f}")
        print(f"   Text: {text[:150]}...")
        print()


def main():
    """Example usage"""
    # Create database
    db = VectorDatabase()

    # Load preprocessed documents
    db.load_documents('preprocessed_documents.json')

    # Build index (use 'flat' for exact search, 'ivf' for faster approximate search)
    db.build_index(index_type='flat')

    # Example search with embedding
    print("\nExample: Search with document embedding")
    query_embedding = db.documents[0]['embedding']
    results = db.search(query_embedding, k=5, remove_duplicates=True)

    for rank, (doc_id, dist, text) in enumerate(results, 1):
        print(f"{rank}. Document ID: {doc_id}, Distance: {dist:.4f}")
        print(f"   Preview: {text[:100]}...\n")

    # Run tests
    test_vector_db()


if __name__ == "__main__":
    main()
