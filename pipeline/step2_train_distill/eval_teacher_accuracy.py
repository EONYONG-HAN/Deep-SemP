import pandas as pd
import torch
import gzip
import subprocess
import random
import os
import sys
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda"
ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")

# FASTQ_FILE = ROOT / "real_data/SRR36278209_1.fastq.gz"
FASTQ_FILE = "/data3/projects/2025_Assembly/eyh/c_elegans_rep2/raw_data/SRR36114682_1.fastq.gz"
REF_CDNA = ROOT / "reference/c_elegans_cdna.fa.gz"
BUCKET_MAP = ROOT / "buckets/semantic_map.tsv"
CHECKPOINT_PATH = ROOT / "models/checkpoints/track_a_direct_full/best_model.pt"
MODEL_BASE_NAME = "zhihan1996/DNABERT-2-117M"

# --- Path to your training CSV for imbalance reference ---
TRAIN_CSV = ROOT / "training_data/simulation_data_full.csv"

SAMPLE_SIZE = 10000
TOP_K = 5
NUM_LABELS = 50


def load_bucket_map():
    print(f"1. Loading Map from {BUCKET_MAP}...")
    tx_to_buckets = {}
    with open(BUCKET_MAP, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            b_id = int(parts[0])
            for tx in parts[3].split(','):
                clean_tx = tx.strip()
                if clean_tx not in tx_to_buckets:
                    tx_to_buckets[clean_tx] = set()
                tx_to_buckets[clean_tx].add(b_id)
    print(f"   Mapped {len(tx_to_buckets)} transcripts.")
    return tx_to_buckets


def get_real_reads():
    print(f"2. Sampling {SAMPLE_SIZE} real reads...")
    reads = []
    try:
        with gzip.open(FASTQ_FILE, 'rt') as f:
            while len(reads) < SAMPLE_SIZE:
                try:
                    head = f.readline(); seq = f.readline(); f.readline(); f.readline()
                    if not head:
                        break
                    if len(seq.strip()) >= 50:
                        if random.random() < 0.1:
                            reads.append(seq.strip())
                except ValueError:
                    break
    except Exception as e:
        print(f"Error reading FASTQ: {e}")
    print(f"   Sampled {len(reads)} reads.")
    return reads


def get_ground_truth(reads):
    print("3. Running BLAST to find True Identity...")

    if not os.path.exists("ref_cdna_db.nhr"):
        print("   Building BLAST DB (unzipping first)...")
        subprocess.run(f"gunzip -c {REF_CDNA} > temp_ref.fa", shell=True)
        subprocess.run(f"makeblastdb -in temp_ref.fa -dbtype nucl -out ref_cdna_db", shell=True)
        if os.path.exists("temp_ref.fa"):
            os.remove("temp_ref.fa")

    with open("temp_query.fa", "w") as f:
        for i, seq in enumerate(reads):
            f.write(f">read_{i}\n{seq}\n")

    cmd = "blastn -query temp_query.fa -db ref_cdna_db -outfmt '6 qseqid sseqid' -max_target_seqs 1 -evalue 1e-5"
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
    except subprocess.CalledProcessError:
        print("   BLAST failed.")
        return {}

    truth = {}
    for line in output.splitlines():
        qid, sid = line.split()[:2]
        idx = int(qid.split('_')[1])
        truth[idx] = sid

    print(f"   Identified true genes for {len(truth)} reads.")
    return truth


def load_train_label_counts():
    """Load training label distribution for imbalance reference."""
    if not os.path.exists(TRAIN_CSV):
        print(f"   [WARN] Training CSV not found at {TRAIN_CSV}, skipping imbalance reference.")
        return {}
    df = pd.read_csv(TRAIN_CSV, usecols=["label"])
    return df["label"].value_counts().to_dict()


def evaluate_model(reads, truth, tx_map):
    print("4. Testing Model Accuracy...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    config.num_labels = NUM_LABELS
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = False
    config._attn_implementation = "eager"

    model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys (up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (up to 20): {unexpected[:20]}")

    model.to(device)
    model.eval()

    # --- Per-bucket tracking ---
    bucket_correct_top1  = defaultdict(int)
    bucket_correct_topk  = defaultdict(int)
    bucket_total         = defaultdict(int)
    bucket_pred_dist     = defaultdict(int)   # what the model actually predicts

    correct_top1 = 0
    correct_topk = 0
    total = 0

    print("   Running Inference...")
    for i, seq in enumerate(reads):
        if i not in truth:
            continue
        true_tx = truth[i]

        valid_buckets = None
        if true_tx in tx_map:
            valid_buckets = tx_map[true_tx]
        else:
            base = true_tx.rsplit('.', 1)[0]
            if base in tx_map:
                valid_buckets = tx_map[base]

        if not valid_buckets:
            continue

        total += 1

        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=100,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            _, indices = torch.topk(probs, TOP_K)
            preds = indices[0].cpu().numpy()

        top1_correct = preds[0] in valid_buckets
        topk_correct = any(p in valid_buckets for p in preds)

        if top1_correct:
            correct_top1 += 1
        if topk_correct:
            correct_topk += 1

        # Track per true bucket
        for true_bucket in valid_buckets:
            bucket_total[true_bucket] += 1
            if top1_correct:
                bucket_correct_top1[true_bucket] += 1
            if topk_correct:
                bucket_correct_topk[true_bucket] += 1

        # Track prediction distribution (what bucket model favored)
        bucket_pred_dist[int(preds[0])] += 1

    # --- Overall Results ---
    print("\n" + "=" * 50)
    if total > 0:
        print(f"REAL DATA ACCURACY (N={total})")
        print(f"Top-1 Accuracy: {correct_top1/total:.2%}")
        print(f"Top-{TOP_K} Accuracy: {correct_topk/total:.2%}")
    else:
        print("   No reads matched between BLAST and Bucket Map.")
    print("=" * 50)

    # --- Per-Bucket Breakdown ---
    train_counts = load_train_label_counts()

    print("\n--- Per-Bucket Accuracy (sorted by Top-1 Acc) ---")
    print(f"{'Bucket':>8} | {'N_real':>7} | {'Top1%':>7} | {'Top5%':>7} | {'Train_N':>10} | {'Flag'}")
    print("-" * 65)

    rows = []
    for bucket in range(NUM_LABELS):
        n = bucket_total[bucket]
        if n == 0:
            continue
        top1_acc = bucket_correct_top1[bucket] / n * 100
        topk_acc = bucket_correct_topk[bucket] / n * 100
        train_n  = train_counts.get(bucket, 0)
        rows.append((bucket, n, top1_acc, topk_acc, train_n))

    # Sort by Top-1 accuracy ascending (worst buckets first)
    rows.sort(key=lambda x: x[2])

    for bucket, n, top1_acc, topk_acc, train_n in rows:
        # Flag buckets that are both low accuracy AND low training data
        flag = ""
        if top1_acc < 50.0 and train_n < 50000:
            flag = "<-- LOW DATA + LOW ACC"
        elif top1_acc < 50.0:
            flag = "<-- LOW ACC"
        elif train_n < 50000:
            flag = "<-- LOW TRAIN DATA"
        print(f"{bucket:>8} | {n:>7} | {top1_acc:>6.1f}% | {topk_acc:>6.1f}% | {train_n:>10,} | {flag}")

    # --- Prediction Bias Check ---
    print("\n--- Top-10 Most Predicted Buckets (Prediction Bias) ---")
    print(f"{'Bucket':>8} | {'Pred_Count':>10} | {'Train_N':>10}")
    print("-" * 40)
    for bucket, count in sorted(bucket_pred_dist.items(), key=lambda x: -x[1])[:10]:
        train_n = train_counts.get(bucket, 0)
        print(f"{bucket:>8} | {count:>10} | {train_n:>10,}")

    # --- Save full results to TSV ---
    out_path = "per_bucket_accuracy.tsv"
    with open(out_path, "w") as f:
        f.write("bucket\tn_real\ttop1_acc\ttop5_acc\ttrain_n\n")
        for bucket, n, top1_acc, topk_acc, train_n in sorted(rows, key=lambda x: x[0]):
            f.write(f"{bucket}\t{n}\t{top1_acc:.2f}\t{topk_acc:.2f}\t{train_n}\n")
    print(f"\nFull per-bucket results saved to: {out_path}")


if __name__ == "__main__":
    tx_map = load_bucket_map()
    reads = get_real_reads()
    truth = get_ground_truth(reads)
    evaluate_model(reads, truth, tx_map)