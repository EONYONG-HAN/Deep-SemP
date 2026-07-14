import pandas as pd
import numpy as np
from collections import Counter

# ==========================================
# Configuration
# ==========================================

ASSEMBLIES = {
    "Single Baseline": {
        "fasta": "/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/baseline/05_final_coding.fasta",
        "blast": "/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results/single_baseline_blast.tsv",
        "total_contigs": 8285,
    },
    "Single Student": {
        "fasta": "/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/student_stratB/05_final_coding.fasta",
        "blast": "/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results/single_student_blast.tsv",
        "total_contigs": 7601,
    },
    "Merged Baseline": {
        "fasta": "/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/merged_baseline/05_final_coding.fasta",
        "blast": "/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results/merged_baseline_blast.tsv",
        "total_contigs": 9219,
    },
    "Merged Student": {
        "fasta": "/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/merged_student_stratB/05_final_coding.fasta",
        "blast": "/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results/merged_student_blast.tsv",
        "total_contigs": 9144,
    },
}


COLUMNS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
           "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen"]

# ==========================================
# FASTA General Statistics
# ==========================================
def parse_fasta(fasta_path):
    """Parse FASTA and return list of (name, seq) tuples."""
    sequences = []
    name, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    sequences.append((name, "".join(seq)))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if name is not None:
        sequences.append((name, "".join(seq)))
    return sequences

def calc_n50(lengths):
    lengths_sorted = sorted(lengths, reverse=True)
    total = sum(lengths_sorted)
    cumsum = 0
    for l in lengths_sorted:
        cumsum += l
        if cumsum >= total / 2:
            return l
    return 0

def gc_content(seq):
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return (gc / len(seq) * 100) if len(seq) > 0 else 0

def fasta_stats(fasta_path, name):
    print(f"  Parsing {name}...")
    seqs = parse_fasta(fasta_path)
    lengths = [len(s) for _, s in seqs]
    gc_values = [gc_content(s) for _, s in seqs]

    return {
        "Contig Count":         len(lengths),
        "Total Assembly (bp)":  sum(lengths),
        "Avg Contig Size (bp)": round(np.mean(lengths), 1),
        "Median Contig (bp)":   int(np.median(lengths)),
        "Min Contig (bp)":      min(lengths),
        "Max Contig (bp)":      max(lengths),
        "N50 (bp)":             calc_n50(lengths),
        "Avg GC Content (%)":   round(np.mean(gc_values), 2),
    }

# ==========================================
# BLAST Statistics
# ==========================================
def blast_stats(tsv_path, total_contigs, name):
    print(f"  Parsing BLAST for {name}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t', names=COLUMNS)
    except FileNotFoundError:
        print(f"  [ERROR] {tsv_path} not found.")
        return None

    df['query_coverage']   = (df['length'] / df['qlen'] * 100).clip(upper=100)
    df['subject_coverage'] = (df['length'] / df['slen'] * 100).clip(upper=100)

    unique_hits  = df['qseqid'].nunique()
    unique_genes = df['sseqid'].nunique()

    return {
        "Contigs with Hits":                  unique_hits,
        "Annotation Rate (%)":                round(unique_hits / total_contigs * 100, 2),
        "Unique Genes Found":                 unique_genes,
        "Avg Identity (%)":                   round(df['pident'].mean(), 2),
        "Avg Query Coverage (%)":             round(df['query_coverage'].mean(), 2),
        "Avg Reference Coverage (%)":         round(df['subject_coverage'].mean(), 2),
        "Full-Length Transcripts (>90% Cov)": int((df['subject_coverage'] >= 90).sum()),
    }

# ==========================================
# Main
# ==========================================
def main():
    fasta_results = {}
    blast_results = {}

    print("\n=== Computing FASTA Statistics ===")
    for name, cfg in ASSEMBLIES.items():
        fasta_results[name] = fasta_stats(cfg["fasta"], name)

    print("\n=== Computing BLAST Statistics ===")
    for name, cfg in ASSEMBLIES.items():
        result = blast_stats(cfg["blast"], cfg["total_contigs"], name)
        if result:
            blast_results[name] = result

    # ── Print FASTA stats ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("              GENERAL ASSEMBLY STATISTICS")
    print("=" * 65)
    fasta_df = pd.DataFrame(fasta_results).T
    print(fasta_df.to_string())

    # ── Print BLAST stats ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("                 BLAST EVALUATION RESULTS")
    print("=" * 65)
    blast_df = pd.DataFrame(blast_results).T
    print(blast_df.to_string())

    # ── Save combined TSV ──────────────────────────────────────
    combined = pd.concat([fasta_df, blast_df], axis=1)
    combined.to_csv("full_assembly_comparison.tsv", sep='\t')
    print("\nSaved: full_assembly_comparison.tsv")

if __name__ == "__main__":
    main()