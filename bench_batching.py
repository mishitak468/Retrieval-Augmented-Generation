# bench_batching.py

import time
import csv
import argparse
import json
from pathlib import Path
import numpy as np

# Import your modules
from encode import QueryEncoder
from vector_db import VectorDatabase


def now():
    """Returns the current high-resolution performance counter time."""
    return time.perf_counter()


def load_queries(path, max_q=None):
    """
    Load queries from JSON file and handles repetition to meet max_q.
    Filters out empty or invalid queries.
    """
    print(f"Loading queries from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Extract and clean valid query texts
    raw_texts = []
    for item in data:
        q_text = None
        if isinstance(item, dict):
            # Attempt to extract 'query' or 'text'
            q_text = item.get('query') or item.get('text')
        elif isinstance(item, str):
            q_text = item

        # Ensure it is a non-empty string
        if q_text and str(q_text).strip():
            raw_texts.append(str(q_text).strip())

    if not raw_texts:
        raise ValueError(f"No valid queries found in {path}.")

    # 2. Handle max_queries and repetition
    total_queries = []
    if max_q and len(raw_texts) < max_q:
        # Repeat the queries to reach max_q
        full_cycles = max_q // len(raw_texts)
        remaining = max_q % len(raw_texts)

        for _ in range(full_cycles):
            total_queries.extend(raw_texts)
        total_queries.extend(raw_texts[:remaining])

    elif max_q:
        # Truncate to max_q
        total_queries = raw_texts[:max_q]
    else:
        total_queries = raw_texts

    print(
        f"Loaded {len(raw_texts)} unique queries, using {len(total_queries)} for benchmark.")

    return total_queries


def run(args):
    """Initializes RAG components and runs the batching benchmark."""
    # Init components
    encoder = QueryEncoder()
    db = VectorDatabase()
    db.load_documents(args.preprocessed)
    db.build_index(index_type=args.index_type)

    # Load all queries required for the max_queries total
    all_queries = load_queries(args.queries, max_q=args.max_queries)
    num_total_queries = len(all_queries)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / "batching_latencies.csv"

    print(f"\nStarting batch benchmark across sizes: {args.batch_sizes}")

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_size", "t_encode_total", "t_encode_per_query",
                         "t_search_total", "t_search_per_query", "num_vectors_in_batch"])

        for batch_size in args.batch_sizes:
            batch_size = int(batch_size)

            # Use the first 'batch_size' queries from the padded list for this test
            queries_batch = all_queries[:batch_size]

            # --- Measure Encoding Latency ---
            t0 = now()
            # Note: SentenceTransformer.encode handles batching internally,
            # so we measure the total time to encode the whole batch list.
            embeddings_batch = encoder.encode_batch(queries_batch)
            t_encode_total = now() - t0

            # --- Measure Search Latency ---
            t1 = now()

            # The FAISS index searches all vectors in the batch simultaneously.
            # We time the single call to search all batch embeddings.
            distances, indices = db.index.search(embeddings_batch, k=args.k)
            t_search_total = now() - t1

            # --- Calculate Metrics ---

            # Average time per query in this batch
            t_encode_per_query = t_encode_total / batch_size
            t_search_per_query = t_search_total / batch_size

            print(
                f"  Batch Size {batch_size}: Encode Time={t_encode_total:.4f}s, Search Time={t_search_total:.4f}s")

            # --- Write Results ---
            writer.writerow([
                batch_size,
                f"{t_encode_total:.6f}",
                f"{t_encode_per_query:.6f}",
                f"{t_search_total:.6f}",
                f"{t_search_per_query:.6f}",
                batch_size
            ])

    print("\nBatching benchmark complete!")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RAG Query Encoding and Vector Search batch performance.")

    # Required Configuration Files
    parser.add_argument("--preprocessed", default="preprocessed_documents.json",
                        help="Path to the preprocessed documents JSON file.")
    parser.add_argument("--queries", default="queries.json",
                        help="Path to the JSON file containing the queries for testing.")

    # RAG Component Configuration
    parser.add_argument("--index_type", default="flat", choices=["flat", "ivf"],
                        help="Type of FAISS index to use for search.")

    # Benchmark Scope and Parameters
    parser.add_argument("--k", type=int, default=5,
                        help="Number of documents (k) to retrieve per query.")
    parser.add_argument("--max_queries", type=int, default=128,
                        help="Total number of queries (unique or repeated) to prepare for the largest batch test.")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save the CSV results file.")

    # Batching Specific Parameters
    parser.add_argument("--batch_sizes", type=int, nargs='+', required=True,
                        help="List of batch sizes to test (e.g., 1 4 8 16 32 64 128).")

    args = parser.parse_args()

    # Ensure max_queries is large enough for the largest requested batch size
    if args.batch_sizes and args.max_queries < max(args.batch_sizes):
        print(
            f"Warning: --max_queries ({args.max_queries}) is smaller than the largest batch size ({max(args.batch_sizes)}). Setting max_queries to largest batch size.")
        args.max_queries = max(args.batch_sizes)

    run(args)
