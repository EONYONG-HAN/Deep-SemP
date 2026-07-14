import pandas as pd

COLS = ["qseqid","sseqid","pident","length","mismatch","gapopen",
        "qstart","qend","sstart","send","evalue","bitscore","qlen","slen"]

# ── C. remanei exclusively rescued ──────────────────────────────
base_rem = pd.read_csv(
    "/data3/projects/2025_Assembly/eyh/c_remanei/blast_results/remanei_baseline_blast.tsv",
    sep='\t', names=COLS)
stud_rem = pd.read_csv(
    "/data3/projects/2025_Assembly/eyh/c_remanei/blast_results/remanei_student_blast.tsv",
    sep='\t', names=COLS)

base_genes_rem = set(base_rem['sseqid'].unique())
stud_genes_rem = set(stud_rem['sseqid'].unique())
rescued_rem    = stud_genes_rem - base_genes_rem
missed_rem     = base_genes_rem - stud_genes_rem

print(f"Remanei — rescued by student : {len(rescued_rem)}")
print(f"Remanei — missed by student  : {len(missed_rem)}")

# with open("remanei_rescued_genes.txt","w") as f:
#     for g in sorted(rescued_rem): f.write(g+"\n")

# ── C. briggsae missed genes ─────────────────────────────────────
base_bri = pd.read_csv(
    "/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results/merged_baseline_blast.tsv",
    sep='\t', names=COLS)
stud_bri = pd.read_csv(
    "/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results/merged_student_blast.tsv",
    sep='\t', names=COLS)

base_genes_bri = set(base_bri['sseqid'].unique())
stud_genes_bri = set(stud_bri['sseqid'].unique())
rescued_bri    = stud_genes_bri - base_genes_bri
missed_bri     = base_genes_bri - stud_genes_bri

print(f"Briggsae — rescued by student : {len(rescued_bri)}")
print(f"Briggsae — missed by student  : {len(missed_bri)}")

with open("briggsae_missed_genes.txt","w") as f:
    for g in sorted(missed_bri): f.write(g+"\n")
with open("briggsae_rescued_genes.txt","w") as f:
    for g in sorted(rescued_bri): f.write(g+"\n")