from sentence_transformers import SentenceTransformer
import numpy as np


class QueryEncoder:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        print(f"Loading encoder model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 768
        print(f"Encoder ready. Output dimension: {self.embedding_dim}")

    def encode(self, query_text):
        # Encode the query
        embedding = self.model.encode(query_text, normalize_embeddings=False)

        # Ensure correct shape and type
        embedding = np.array(embedding, dtype='float32')

        # Verify dimensions
        assert embedding.shape == (self.embedding_dim,), \
            f"Expected shape ({self.embedding_dim},), got {embedding.shape}"

        return embedding

    def encode_batch(self, query_texts):
        """
        Convert multiple queries into embeddings

        Args:
            query_texts: List of query strings

        Returns:
            numpy array of shape (n_queries, 768)
        """
        embeddings = self.model.encode(
            query_texts,
            normalize_embeddings=False,
            show_progress_bar=False
        )
        return np.array(embeddings, dtype='float32')


def test_encoder():
    """Test the encoder with sample queries"""
    print("\n" + "="*60)
    print("Testing Query Encoder")
    print("="*60)

    encoder = QueryEncoder()

    # Test single query
    test_query = "What causes animals to lose fur?"
    print(f"\nTest Query: '{test_query}'")

    embedding = encoder.encode(test_query)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {embedding.dtype}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    # Test batch encoding
    batch_queries = [
        "How do I fix my computer?",
        "What is the weather today?",
        "Cornell course policies"
    ]

    print(f"\nTesting batch encoding with {len(batch_queries)} queries...")
    batch_embeddings = encoder.encode_batch(batch_queries)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")

    print("\n✓ Encoder test passed!")


if __name__ == "__main__":
    test_encoder()
