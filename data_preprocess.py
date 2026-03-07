"""
Data Preprocessing Script for RAG Vector Database
Processes MS MARCO documents into embeddings using BGE model
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_documents(filepath):
    """Load documents from JSON file"""
    print(f"Loading documents from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents")
    return documents


def encode_documents(documents, model_name='BAAI/bge-base-en-v1.5', batch_size=32):
    """
    Encode documents using BGE model

    Args:
        documents: List of document dictionaries with 'id' and 'text' fields
        model_name: HuggingFace model identifier
        batch_size: Number of documents to process at once

    Returns:
        List of documents with added 'embedding' field
    """
    print(f"Loading BGE model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract texts for batch encoding
    texts = [doc['text'] for doc in documents]

    print(f"Encoding {len(texts)} documents...")
    # Encode in batches with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False  # Keep raw embeddings
    )

    # Add embeddings to documents
    processed_docs = []
    for doc, embedding in zip(documents, embeddings):
        processed_doc = {
            'id': doc['id'],
            'text': doc['text'],
            'embedding': embedding.tolist()  # Convert numpy array to list for JSON
        }
        processed_docs.append(processed_doc)

    return processed_docs


def save_preprocessed_data(documents, output_path):
    """Save preprocessed documents with embeddings to JSON"""
    print(f"Saving preprocessed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    print(f"Successfully saved {len(documents)} documents")


def main():
    # Configuration
    INPUT_FILE = 'documents.json'
    OUTPUT_FILE = 'preprocessed_documents.json'
    MODEL_NAME = 'BAAI/bge-base-en-v1.5'
    BATCH_SIZE = 32

    # Step 1: Load documents
    documents = load_documents(INPUT_FILE)

    # Step 2: Encode documents
    processed_documents = encode_documents(
        documents,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE
    )

    # Verify embedding dimensions
    if processed_documents:
        embedding_dim = len(processed_documents[0]['embedding'])
        print(f"Embedding dimension: {embedding_dim}")
        assert embedding_dim == 768, f"Expected 768 dimensions, got {embedding_dim}"

    # Step 3: Save preprocessed data
    save_preprocessed_data(processed_documents, OUTPUT_FILE)

    print("\nPreprocessing complete!")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
