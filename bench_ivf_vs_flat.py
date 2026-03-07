# bench_ivf_vs_flat.py
"""
Benchmarks FAISS IndexFlatL2 (exact search) against IndexIVFFlat 
(approximate search) based on search latency and recall@k.
"""

import time
import argparse
import numpy as np
import csv
from pathlib import Path
from encode import QueryEncoder
from vector_db import VectorDatabase
import faiss
import json
import math  # Added for robust query loading logic


def now():
    """Returns the current high-resolution performance counter time."""
    return time.perf_counter()


def load_queries(path, max_q=None):
    """
    Load queries from JSON file, clean/filter invalid entries, 
    and handle repetition to meet max_q.
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

        # Filter: must exist and be a non-empty string
        if q_text and str(q_text).strip():
            raw_texts.append(str(q_text).strip())

    if not raw_texts:
        raise ValueError(f"No valid queries found in {path}.")

    # 2. Handle max_queries and repetition
    total_queries = []
    if max_q and len(raw_texts) < max_q:
        # Repeat the queries to reach max_q

        # Calculate how many times to repeat the unique list
        total_repetitions = math.ceil(max_q / len(raw_texts))

        for _ in range(total_repetitions):
            total_queries.extend(raw_texts)

        # Truncate to the exact max_q size
        total_queries = total_queries[:max_q]

    elif max_q:
        # Truncate raw_texts to max_q
        total_queries = raw_texts[:max_q]
    else:
        total_queries = raw_texts

    print(
        f"Loaded {len(raw_texts)} unique queries, using {len(total_queries)} for benchmark.")

    return total_queries


def build_ivf(db, nlist=100):
    """Builds and trains a FAISS IndexIVFFlat index."""
    embeddings = np.array([doc['embedding']
                          for doc in db.documents], dtype='float32')
    quantizer = faiss.IndexFlatL2(db.embedding_dim)
    ivf = faiss.IndexIVFFlat(quantizer, db.embedding_dim, nlist)
    print("Training IVFFlat ...")
    ivf.train(embeddings)
    ivf.add(embeddings)
    return ivf


def run(args):
    """Orchestrates the benchmark run."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load DB and Build Indexes
    db = VectorDatabase()
    db.load_documents(args.preprocessed)

    # Build flat index (FAISS) - IndexFlatL2 provides the ground truth
    embeddings = np.array([doc['embedding']
                          for doc in db.documents], dtype='float32')
    flat = faiss.IndexFlatL2(db.embedding_dim)
    flat.add(embeddings)
    print("Flat index built")

    # Build ivf index
    ivf = build_ivf(db, nlist=args.nlist)
    print("IVF built")

    # 2. Load Queries and Encode
    encoder = QueryEncoder()

    # Use the new, robust loader function
    queries = load_queries(args.queries, max_q=args.max_queries)

    # Encode all queries in a single batch
    qembs = encoder.encode_batch(queries).astype('float32')

    # 3. Run Benchmark and Save Results
    csv_path = Path(args.output_dir)/"ivf_vs_flat.csv"
    print(f"\nStarting search comparison with nprobe={args.nprobe}...")

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query_idx", "flat_time_ms", "ivf_time_ms", "ivf_nprobe",
                        "flat_top1_idx", "ivf_top1_idx", "topK", "recall_at_k"])

        for i, emb in enumerate(qembs):
            emb = emb.reshape(1, -1)

            # --- FLAT (Exact Search) ---
            s = now()
            Df, If = flat.search(emb, args.k)
            tflat = (now()-s)*1000.0  # Time in milliseconds

            # --- IVF (Approximate Search) ---
            ivf.nprobe = args.nprobe
            s = now()
            Div, Iiv = ivf.search(emb, args.k)
            tivf = (now()-s)*1000.0  # Time in milliseconds

            # --- Compute Recall@k ---
            flat_ids = list(If[0])
            ivf_ids = list(Iiv[0])

            # Recall@k: Fraction of top-K exact results (flat_ids) found by the IVF search (ivf_ids)
            overlap = len(set(flat_ids).intersection(set(ivf_ids)))
            recall = overlap / float(args.k)

            # --- Write Row ---
            writer.writerow([i, f"{tflat:.3f}", f"{tivf:.3f}", args.nprobe,
                            flat_ids[0], ivf_ids[0], args.k, f"{recall:.3f}"])

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(qembs)} queries.")

    print("\nDone. Results saved to:", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares FAISS FLAT vs IVF Index performance.")
    parser.add_argument(
        "--preprocessed", default="preprocessed_documents.json",
        help="Path to the preprocessed documents JSON file.")
    parser.add_argument("--queries", default="queries.json",
                        help="Path to the JSON file containing the queries for testing.")
    parser.add_argument("--max_queries", type=int, default=200,
                        help="Total number of queries to use for the benchmark (repeats if necessary).")
    parser.add_argument("--nlist", type=int, default=100,
                        help="Number of clusters (inverted lists) for the IVF index.")
    parser.add_argument("--nprobe", type=int, default=1,
                        help="Number of clusters to search in the IVF index during query time (lower = faster, lower recall).")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbors to retrieve (topK).")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save the CSV results file.")
    args = parser.parse_args()

    # You must run this command *before* running the script to prevent SegFaults:
    # export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

    run(args)
