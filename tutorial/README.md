# Deep-SemP Quickstart (toy example)

Run Deep-SemP end-to-end on a tiny dataset (~few MB) in a few minutes: download the
trained student model, partition (route) reads into semantic buckets, and assemble one
bucket with Trinity. This shows the mechanics; for full runs see [`docs/tutorial.md`](../../docs/tutorial.md).

Runs on GPU if available, otherwise CPU (slower, but fine at this size).

---

## 0. Environment

```bash
conda activate deep-semp        # from environment.yml (includes Trinity, seqtk)
```

## 1. Download the trained student model

The 8-layer student (`best_student_model.pt`) is hosted on the COBI lab page rather than
GitHub (it exceeds the free-tier size limit) and is git-ignored in this repo.

```bash
bash tutorial/download_model.sh          # -> models/best_student_model.pt
# or manually:
#   https://cobi.knu.ac.kr/tools/deepsemp/best_student_model.pt
```

## 2. Get the toy reads

A small paired-end sample ships in `tutorial/data/` as `toy_R1.fastq.gz` and
`toy_R2.fastq.gz`. To regenerate (or make your own from a full FASTQ):

```bash
R1=/path/SRR36278209_1.fastq.gz R2=/path/SRR36278209_2.fastq.gz \
  N=20000 bash tutorial/make_toy_data.sh
```

## 3. Partition the reads (routing)

```bash
python pipeline/step3_partition_assemble/partition_reads.py \
  --model_path models/best_student_model.pt \
  --r1 tutorial/data/toy_R1.fastq.gz \
  --r2 tutorial/data/toy_R2.fastq.gz \
  --output_dir tutorial/out/buckets \
  --num_labels 50 --num_layers 8 --d_model 384 --masked_pooling \
  --batch_size 256 --max_len 100
```

This writes per-bucket files `bucket_00_R1.fastq` … `bucket_49_R1.fastq` (and `_R2`) into
`out/buckets/`. **Keep the architecture flags exactly as above** (`--num_layers 8
--d_model 384 --masked_pooling --num_labels 50`) — they must match the shipped model.
Add `--fp16` on a GPU for speed.

## 4. Assemble one bucket with Trinity

Assemble the largest non-empty bucket as a demo:

```bash
BUCKET=$(ls -S tutorial/out/buckets/*_R1.fastq | head -1 | sed 's/_R1\.fastq$//')
echo "Assembling ${BUCKET}"
Trinity --seqType fq \
  --left  "${BUCKET}_R1.fastq" \
  --right "${BUCKET}_R2.fastq" \
  --CPU 4 --max_memory 8G \
  --output tutorial/out/trinity_demo --full_cleanup
```

## 5. Check the result

```bash
grep -c '>' tutorial/out/trinity_demo.Trinity.fasta
```

You should get a handful of assembled contigs from that bucket — confirming the model
routes reads and the per-bucket assembly works. To assemble all buckets with the
complexity-adaptive scheduler (as in the paper), use
`pipeline/step3_partition_assemble/run_trinity_adaptive.sh` and then the Step 4
post-processing pipeline.

---

### Notes
- Toy assemblies are illustrative, not benchmark-quality — the point is to verify the
  install and the route→assemble flow.
- CPU routing of 20k pairs takes a couple of minutes; a GPU does it in seconds.
- Bucket files are plain `.fastq`; empty buckets are normal for a small sample.
