# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import seaborn as sns


def plot_component_breakdown(csv_path, outdir="bench/plots"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    # compute numeric columns
    for c in ['t_encode', 't_search', 't_retrieve_prompt', 't_llm', 't_total']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
    # boxplot per-component
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df[['t_encode', 't_search', 't_retrieve_prompt', 't_llm']])
    plt.ylabel("Time (s)")
    plt.title("Latency distribution by component (boxplot)")
    plt.savefig(Path(outdir)/"component_boxplot.png", dpi=200)
    plt.close()

    # stacked area: compute median per component, normalized
    medians = df[['t_encode', 't_search',
                  't_retrieve_prompt', 't_llm']].median()
    medians_norm = medians / medians.sum()
    medians_norm.plot(kind='bar', figsize=(6, 4), stacked=True)
    plt.ylabel("Fraction of total (median)")
    plt.title("Median fraction of time by component")
    plt.savefig(Path(outdir)/"component_fraction_median.png", dpi=200)
    plt.close()
    print("Saved plots to", outdir)


def plot_batching(csv_path, outdir="bench/plots"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['batch_size'] = df['batch_size'].astype(int)
    plt.figure(figsize=(8, 6))
    plt.plot(df['batch_size'], df['t_search_per_query'], marker='o')
    plt.xscale('log', base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Avg search time per query (ms)")
    plt.title("FAISS search: avg per-query time vs batch size")
    plt.grid(True)
    plt.savefig(Path(outdir)/"batching_avg_query_ms.png", dpi=200)
    plt.close()
    print("Saved batching plot")


def plot_ivf_vs_flat(csv_path, outdir="bench/plots"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['flat_time_ms'] = pd.to_numeric(df['flat_time_ms'])
    df['ivf_time_ms'] = pd.to_numeric(df['ivf_time_ms'])
    plt.figure(figsize=(8, 6))
    plt.scatter(df.index, df['flat_time_ms'], s=10, label='Flat')
    plt.scatter(df.index, df['ivf_time_ms'], s=10, label='IVF')
    plt.xlabel("Query index")
    plt.ylabel("Search time (ms)")
    plt.legend()
    plt.title("Per-query search time: Flat vs IVFFlat")
    plt.savefig(Path(outdir)/"ivf_vs_flat_times.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component_csv",
                        default="bench/results/component_latencies.csv")
    parser.add_argument(
        "--batching_csv", default="bench/results/batching_results.csv")
    parser.add_argument("--ivf_csv", default="bench/results/ivf_vs_flat.csv")
    args = parser.parse_args()
    if Path(args.component_csv).exists():
        plot_component_breakdown(args.component_csv)
    if Path(args.batching_csv).exists():
        plot_batching(args.batching_csv)
    if Path(args.ivf_csv).exists():
        plot_ivf_vs_flat(args.ivf_csv)
