# bench_instrumentation.py

import time
import csv
import argparse
import json
from pathlib import Path
import numpy as np


# Import your modules (assumes your repo root)
from llm_generation import LLMGenerator
from vector_db import VectorDatabase
from encode import QueryEncoder


def now():
    return time.perf_counter()


def load_queries(path, max_q=None):
    with open(path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    # queries.json can be list of { "query": ... } - adapt as needed
    texts = []
    for item in queries:
        if isinstance(item, dict):
            # common MS MARCO format: {"id":..., "query": "..."}
            texts.append(item.get('query') or item.get('text') or '')
        else:
            texts.append(str(item))
        if max_q and len(texts) >= max_q:
            break
    return texts


def run(args):
    # Init components
    encoder = QueryEncoder()
    db = VectorDatabase()
    db.load_documents(args.preprocessed)
    db.build_index(index_type=args.index_type)
    llm = LLMGenerator(
        args.model_path, model_type=args.model_type, n_threads=args.n_threads)

    queries = load_queries(args.queries, max_q=args.max_queries)
    print(f"Loaded {len(queries)} queries")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / "component_latencies.csv"

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query_idx", "query_text", "t_encode", "t_search",
                        "t_retrieve_prompt", "t_llm", "t_total", "num_docs_retrieved"])
        for i, q in enumerate(queries):
            t0 = now()
            t1 = now()
            emb = encoder.encode(q)
            t2 = now()
            results = db.search(emb, k=args.top_k, remove_duplicates=True)
            t3 = now()
            # retrieval + prompt
            # build prompt as in your LLMGenerator.create_augmented_prompt
            # We'll include time to fetch texts and build the string
            retrieved_texts = [r[2] for r in results]
            prompt = llm.create_augmented_prompt(q, results)
            t4 = now()
            # LLM generation
            ans = llm.generate(prompt, max_tokens=args.max_tokens)
            t5 = now()

            t_encode = t2 - t1
            t_search = t3 - t2
            t_retrieve_prompt = t4 - t3
            t_llm = t5 - t4
            t_total = t5 - t0

            writer.writerow([i, q.replace("\n", " "), f"{t_encode:.6f}", f"{t_search:.6f}",
                            f"{t_retrieve_prompt:.6f}", f"{t_llm:.6f}", f"{t_total:.6f}", len(results)])

            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{len(queries)} queries")

    print("Done. Results saved to", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed", default="preprocessed_documents.json")
    parser.add_argument("--queries", default="queries.json")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="tinyllama")
    parser.add_argument("--index_type", default="flat",
                        choices=["flat", "ivf"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_queries", type=int, default=200)
    parser.add_argument("--output_dir", default="bench/results")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--n_threads", type=int, default=4)
    args = parser.parse_args()
    run(args)
