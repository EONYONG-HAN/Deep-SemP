import pandas as pd
import os

COLS = ["qseqid","sseqid","pident","length","mismatch","gapopen",
        "qstart","qend","sstart","send","evalue","bitscore","qlen","slen"]

BASE = "/data3/projects/2025_Assembly/eyh"
OUT  = "./gene_lists"
os.makedirs(OUT, exist_ok=True)

species = {
    "elegans": {
        "baseline": f"{BASE}/c_elegans/blast_results/baseline_blast.tsv",
        "student":  f"{BASE}/c_elegans/blast_results/student_blast.tsv",
    },
    "briggsae": {
        "baseline": f"{BASE}/c_briggsae/blast_results/merged_baseline_blast.tsv",
        "student":  f"{BASE}/c_briggsae/blast_results/merged_student_blast.tsv",
    },
    "remanei": {
        "baseline": f"{BASE}/c_remanei/blast_results/remanei_baseline_blast.tsv",
        "student":  f"{BASE}/c_remanei/blast_results/remanei_student_blast.tsv",
    },
    "latens": {
        "baseline": f"{BASE}/c_latens/blast_results/latens_baseline_blast.tsv",
        "student":  f"{BASE}/c_latens/blast_results/latens_student_blast.tsv",
    },
}

print(f"{'Species':<16} {'Baseline':>10} {'Student':>10} "
      f"{'Shared':>10} {'Rescued':>10} {'Missed':>10}")
print("-" * 65)

for sp, paths in species.items():
    base = pd.read_csv(paths["baseline"], sep='\t', names=COLS)
    stud = pd.read_csv(paths["student"],  sep='\t', names=COLS)

    base_genes = set(base['sseqid'].unique())
    stud_genes = set(stud['sseqid'].unique())

    missed  = base_genes - stud_genes
    rescued = stud_genes - base_genes
    shared  = base_genes & stud_genes

    print(f"{sp:<16} {len(base_genes):>10} {len(stud_genes):>10} "
          f"{len(shared):>10} {len(rescued):>10} {len(missed):>10}")

    with open(f"{OUT}/{sp}_rescued.txt", "w") as f:
        for g in sorted(rescued):
            f.write(g.rsplit('.', 1)[0] + "\n")

    with open(f"{OUT}/{sp}_missed.txt", "w") as f:
        for g in sorted(missed):
            f.write(g.rsplit('.', 1)[0] + "\n")

print(f"\nAll gene lists saved to: {OUT}/")
print("\nFiles generated:")
for sp in species.keys():
    print(f"  {sp}_rescued.txt  |  {sp}_missed.txt")