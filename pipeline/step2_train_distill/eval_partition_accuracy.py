import os
import gzip
import subprocess
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")

# FASTQ_FILE = ROOT / "real_data/SRR36278209_1.fastq.gz"  # rep1
FASTQ_FILE = "/data3/projects/2025_Assembly/eyh/c_elegans_rep2/raw_data/SRR36114682_1.fastq.gz"  # rep2

REF_CDNA    = ROOT / "reference/c_elegans_cdna.fa.gz"
BUCKET_MAP  = ROOT / "buckets/semantic_map.tsv"
TRAIN_CSV   = ROOT / "training_data/simulation_data_full.csv"

CHECKPOINT_PATH = Path(
    "/data3/projects/2025_Assembly/eyh/models/checkpoints/track_a_direct_full/distilled_models/weekend_grid/run_dm384_l6_a03_t2_mask/best_student_model.pt"
)
MODEL_BASE_NAME = "zhihan1996/DNABERT-2-117M"

SAMPLE_SIZE = 10000
TOP_K       = 5
NUM_LABELS  = 50
MAX_LEN     = 100

# --- Must match training run ---
D_MODEL         = 384
NHEAD           = 8
NUM_LAYERS      = 6
DIM_FEEDFORWARD = 1024
DROPOUT         = 0.1
MAX_POS_LEN     = 512
MASKED_POOLING  = True


# ==========================================
# Student Model
# ==========================================
class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_pos_len=512,
                 num_buckets=50, masked_pooling=False):
        super().__init__()
        self.masked_pooling = masked_pooling
        self.embedding     = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_pos_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_buckets)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if self.masked_pooling and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)

        return self.fc(x)


# ==========================================
# Data helpers
# ==========================================
def load_bucket_map():
    print(f"1. Loading Map from {BUCKET_MAP}...")
    tx_to_buckets = {}
    with open(BUCKET_MAP, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            b_id = int(parts[0])
            for tx in parts[3].split(","):
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
        with gzip.open(FASTQ_FILE, "rt") as f:
            while len(reads) < SAMPLE_SIZE:
                try:
                    head = f.readline(); seq = f.readline(); f.readline(); f.readline()
                    if not head:
                        break
                    if len(seq.strip()) >= 50 and random.random() < 0.1:
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
        subprocess.run(f"gunzip -c {REF_CDNA} > temp_ref.fa", shell=True, check=True)
        subprocess.run("makeblastdb -in temp_ref.fa -dbtype nucl -out ref_cdna_db", shell=True, check=True)
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
        truth[int(qid.split("_")[1])] = sid
    print(f"   Identified true genes for {len(truth)} reads.")
    return truth


def load_train_label_counts():
    if not os.path.exists(TRAIN_CSV):
        print(f"   [WARN] Training CSV not found at {TRAIN_CSV}, skipping imbalance reference.")
        return {}
    df = pd.read_csv(TRAIN_CSV, usecols=["label"])
    return df["label"].value_counts().to_dict()


# ==========================================
# Evaluation
# ==========================================
def evaluate_student_model(reads, truth, tx_map):
    print("4. Testing STUDENT Model Accuracy...")
    print(f"   Using device: {DEVICE}")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    model = StudentTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, max_pos_len=MAX_POS_LEN,
        num_buckets=NUM_LABELS, masked_pooling=MASKED_POOLING,
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"[WARN] Missing keys    (up to 20): {missing[:20]}")
    if unexpected: print(f"[WARN] Unexpected keys (up to 20): {unexpected[:20]}")

    model.to(DEVICE)
    model.eval()

    # --- Per-bucket tracking ---
    bucket_correct_top1 = defaultdict(int)
    bucket_correct_topk = defaultdict(int)
    bucket_total        = defaultdict(int)
    bucket_pred_dist    = defaultdict(int)

    # --- Per-bucket confidence tracking (avg max softmax) ---
    bucket_conf_sum     = defaultdict(float)

    correct_top1 = correct_topk = total = 0

    print("   Running Inference...")
    for i, seq in enumerate(reads):
        if i not in truth:
            continue

        true_tx = truth[i]
        valid_buckets = tx_map.get(true_tx) or tx_map.get(true_tx.rsplit(".", 1)[0])
        if not valid_buckets:
            continue

        total += 1

        inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                           max_length=MAX_LEN, padding="max_length")
        input_ids      = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=-1)
            top_probs, indices = torch.topk(probs, TOP_K)
            preds      = indices[0].cpu().numpy()
            top1_conf  = top_probs[0][0].item()

        top1_ok = preds[0] in valid_buckets
        topk_ok = any(p in valid_buckets for p in preds)

        if top1_ok: correct_top1 += 1
        if topk_ok: correct_topk += 1

        for true_bucket in valid_buckets:
            bucket_total[true_bucket]        += 1
            bucket_conf_sum[true_bucket]     += top1_conf
            if top1_ok: bucket_correct_top1[true_bucket] += 1
            if topk_ok: bucket_correct_topk[true_bucket] += 1

        bucket_pred_dist[int(preds[0])] += 1

    # ==========================================
    # Overall Results
    # ==========================================
    print("\n" + "=" * 60)
    if total > 0:
        print(f"REAL DATA STUDENT ACCURACY (N={total})")
        print(f"Top-1 Accuracy : {correct_top1/total:.2%}")
        print(f"Top-{TOP_K} Accuracy : {correct_topk/total:.2%}")
    else:
        print("   No reads matched between BLAST and Bucket Map.")
    print("=" * 60)

    # ==========================================
    # Per-Bucket Breakdown
    # ==========================================
    train_counts = load_train_label_counts()

    print("\n--- Per-Bucket Accuracy (sorted by Top-1 Acc, worst first) ---")
    print(f"{'Bucket':>8} | {'N_real':>7} | {'Top1%':>7} | {'Top5%':>7} | {'AvgConf':>8} | {'Train_N':>10} | Flag")
    print("-" * 78)

    rows = []
    for bucket in range(NUM_LABELS):
        n = bucket_total[bucket]
        if n == 0:
            continue
        top1_acc = bucket_correct_top1[bucket] / n * 100
        topk_acc = bucket_correct_topk[bucket] / n * 100
        avg_conf = bucket_conf_sum[bucket] / n * 100
        train_n  = train_counts.get(bucket, 0)
        rows.append((bucket, n, top1_acc, topk_acc, avg_conf, train_n))

    rows.sort(key=lambda x: x[2])  # sort by Top-1 asc

    for bucket, n, top1_acc, topk_acc, avg_conf, train_n in rows:
        flag = ""
        if   top1_acc < 50 and train_n < 50_000: flag = "<-- LOW DATA + LOW ACC"
        elif top1_acc < 50:                       flag = "<-- LOW ACC"
        elif train_n  < 50_000:                   flag = "<-- LOW TRAIN DATA"
        # Flag buckets where model is confidently wrong
        if top1_acc < 70 and avg_conf > 70:
            flag += " !! CONFIDENT+WRONG"
        print(f"{bucket:>8} | {n:>7} | {top1_acc:>6.1f}% | {topk_acc:>6.1f}% | {avg_conf:>7.1f}% | {train_n:>10,} | {flag}")

    # ==========================================
    # Teacher vs Student gap (requires teacher
    # per_bucket_accuracy.tsv from prior run)
    # ==========================================
    teacher_tsv = Path("per_bucket_accuracy.tsv")
    if teacher_tsv.exists():
        print("\n--- Teacher vs Student Gap (per bucket) ---")
        t_df = pd.read_csv(teacher_tsv, sep="\t")
        t_df = t_df.set_index("bucket")

        print(f"{'Bucket':>8} | {'Teacher%':>9} | {'Student%':>9} | {'Gap':>7} | {'Train_N':>10}")
        print("-" * 58)

        gap_rows = []
        for bucket, n, top1_acc, topk_acc, avg_conf, train_n in rows:
            if bucket in t_df.index:
                teacher_acc = t_df.loc[bucket, "top1_acc"]
                gap = teacher_acc - top1_acc
                gap_rows.append((bucket, teacher_acc, top1_acc, gap, train_n))

        gap_rows.sort(key=lambda x: -x[3])  # sort by gap descending

        for bucket, t_acc, s_acc, gap, train_n in gap_rows:
            marker = " ***" if gap > 20 else ""
            print(f"{bucket:>8} | {t_acc:>8.1f}% | {s_acc:>8.1f}% | {gap:>+6.1f}% | {train_n:>10,}{marker}")
    else:
        print(f"\n[INFO] No teacher TSV found at '{teacher_tsv}'. Run eval_real_per_bucket.py first to enable gap analysis.")

    # ==========================================
    # Prediction Bias
    # ==========================================
    print("\n--- Top-10 Most Predicted Buckets (Student Bias) ---")
    print(f"{'Bucket':>8} | {'Pred_Count':>10} | {'Train_N':>10}")
    print("-" * 38)
    for bucket, count in sorted(bucket_pred_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"{bucket:>8} | {count:>10} | {train_counts.get(bucket, 0):>10,}")

    # ==========================================
    # Save TSV
    # ==========================================
    out_path = "per_bucket_accuracy_student.tsv"
    with open(out_path, "w") as f:
        f.write("bucket\tn_real\ttop1_acc\ttop5_acc\tavg_conf\ttrain_n\n")
        for bucket, n, top1_acc, topk_acc, avg_conf, train_n in sorted(rows, key=lambda x: x[0]):
            f.write(f"{bucket}\t{n}\t{top1_acc:.2f}\t{topk_acc:.2f}\t{avg_conf:.2f}\t{train_n}\n")
    print(f"\nFull per-bucket student results saved to: {out_path}")


if __name__ == "__main__":
    tx_map = load_bucket_map()
    reads  = get_real_reads()
    truth  = get_ground_truth(reads)
    evaluate_student_model(reads, truth, tx_map)