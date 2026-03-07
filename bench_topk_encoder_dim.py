# bench_topk_encoder_dim.py
import argparse
import time
import csv
import json
from pathlib import Path
from encode import QueryEncoder
from vector_db import VectorDatabase
from llm_generation import LLMGenerator
import math  # <-- ADDED: Needed for math.ceil in load_queries


def now(): return time.perf_counter()


def load_queries(path, max_q=None):
    """
    Load queries from JSON file, clean/filter invalid entries, 
    and handle repetition to meet max_q.
    """
    print(f"Loading queries from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_texts = []
    for item in data:
        q_text = None
        if isinstance(item, dict):
            q_text = item.get('query') or item.get('text')
        elif isinstance(item, str):
            q_text = item

        # Filter: must exist and be a non-empty string
        if q_text and str(q_text).strip():
            raw_texts.append(str(q_text).strip())

    if not raw_texts:
        raise ValueError(f"No valid queries found in {path}.")

    total_queries = []
    if max_q and len(raw_texts) < max_q:
        total_repetitions = math.ceil(max_q / len(raw_texts))
        for _ in range(total_repetitions):
            total_queries.extend(raw_texts)
        total_queries = total_queries[:max_q]

    elif max_q:
        total_queries = raw_texts[:max_q]
    else:
        total_queries = raw_texts

    print(
        f"Loaded {len(raw_texts)} unique queries, using {len(total_queries)} for benchmark.")

    return total_queries


def run(args):
    """Runs the full RAG pipeline benchmark for top-K analysis."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Initialize Components
    encoder = QueryEncoder(model_name=args.encoder_model)
    db = VectorDatabase()
    db.load_documents(args.preprocessed)
    db.build_index(index_type=args.index_type)
    llm = LLMGenerator(args.model_path, model_type=args.model_type)

    # 2. Load and Clean Queries (CORRECTED LINE)
    # Replaced manual, error-prone loading with the new function
    queries = load_queries(args.queries, max_q=args.max_queries)

    csv_path = Path(args.output_dir) / \
        f"topk_encoder_{args.encoder_model.replace('/','_')}_topk{args.top_k}.csv"

    print(f"Starting benchmark for {len(queries)} queries...")

    # 3. Benchmark Loop
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["qidx", "top_k", "t_total_ms", "t_encode_ms",
                        "t_search_ms", "t_prompt_ms", "t_llm_ms", "num_docs"])
        for i, q in enumerate(queries):
            t0 = now()

            # Encode Query
            emb = encoder.encode(q)
            t1 = now()

            # Retrieve Documents
            results = db.search(emb, k=args.top_k, remove_duplicates=True)
            t2 = now()

            # Create Augmented Prompt
            prompt = llm.create_augmented_prompt(q, results)
            t3 = now()

            # Generate Answer
            ans = llm.generate(prompt, max_tokens=args.max_tokens)
            t4 = now()

            # Write Results
            writer.writerow([i, args.top_k, (t4-t0)*1000.0, (t1-t0)*1000.0,
                            (t2-t1)*1000.0, (t3-t2)*1000.0, (t4-t3)*1000.0, len(results)])

            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(queries)} done")

    print("Saved", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed", default="preprocessed_documents.json")
    parser.add_argument("--queries", default="queries.json")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="tinyllama")
    parser.add_argument("--encoder_model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--index_type", default="flat")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_queries", type=int, default=100)
    # Corrected default path
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    # Ensure environment variables are set for stability
    # export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

    run(args)
