import numpy as np

species = {
    "C. elegans": "/data3/projects/2025_Assembly/eyh/c_elegans/partitioned_reads/student_nh_illumina5/bucket_distribution_merged.tsv",
    "C. briggsae": "/data3/projects/2025_Assembly/eyh/c_briggsae/partitioned_reads/student_nh_illumina5_merged/bucket_distribution_merged.tsv",
    "C. remanei":  "/data3/projects/2025_Assembly/eyh/c_remanei/partitioned_reads/student_nh_illumina5/bucket_distribution_merged.tsv",
    "C. latens":   "/data3/projects/2025_Assembly/eyh/c_latens/partitioned_reads/student_nh_illumina5/bucket_distribution_merged.tsv",
    "H. sapiens":  "/data3/projects/2025_Assembly/eyh/h_sapiens/partitioned_reads/student_nh_illumina5/bucket_distribution_merged.tsv",
}

print(f"{'Species':<20} {'Entropy':>10} {'Norm. Entropy':>14} {'Top Bucket%':>12} {'Ratio max/min':>14}")
print("-" * 75)

for name, path in species.items():
    try:
        counts = {}
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                counts[int(parts[0])] = int(parts[1])

        total = sum(counts.values())
        props = np.array([v/total for v in counts.values()])
        props = props[props > 0]

        entropy     = -np.sum(props * np.log(props))
        max_entropy = np.log(len(counts))
        norm_entropy = entropy / max_entropy
        top_pct     = max(props) * 100
        ratio       = max(counts.values()) / max(1, min(counts.values()))

        print(f"{name:<20} {entropy:>10.4f} {norm_entropy:>14.4f} {top_pct:>11.1f}% {ratio:>14.1f}x")
    except FileNotFoundError:
        print(f"{name:<20} {'FILE NOT FOUND':>10}")