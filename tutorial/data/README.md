# Toy data

`toy_R1.fastq.gz` and `toy_R2.fastq.gz` are a small (~few MB) paired-end sample of
*C. elegans* RNA-seq (subsampled from SRR36278209) used by the quickstart.

Generate them with:

```bash
R1=/path/SRR36278209_1.fastq.gz R2=/path/SRR36278209_2.fastq.gz \
  N=20000 bash tutorial/make_toy_data.sh
```

then commit the two `.gz` files (they are whitelisted in `.gitignore`).
