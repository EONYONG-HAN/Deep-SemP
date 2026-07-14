import os
import gzip
import subprocess
import random
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")

# FASTQ_FILE = ROOT / "real_data/SRR36278209_1.fastq.gz"  # rep1
FASTQ_FILE = "/data3/projects/2025_Assembly/eyh/c_elegans_rep2/raw_data/SRR36114682_1.fastq.gz"  # rep2

REF_CDNA = ROOT / "reference/c_elegans_cdna.fa.gz"
BUCKET_MAP = ROOT / "buckets/semantic_map.tsv"

# --- Student checkpoint ---
CHECKPOINT_PATH = Path(
    "/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/track_a_direct_full/distilled_models/higher_error/d384_l6_a0.3_t2/best_student_model.pt"
)
# CHECKPOINT_PATH = Path("/data3/projects/2025_Assembly/eyh/models/checkpoints/track_a_direct_full/distilled_models/run_dm384_l6_a03_t3_mask/best_student_model_epoch1_snapshot.pt")
MODEL_BASE_NAME = "zhihan1996/DNABERT-2-117M"

SAMPLE_SIZE = 10000
TOP_K = 5
NUM_LABELS = 50
MAX_LEN = 100

# --- Must match training run ---
D_MODEL = 384
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
MAX_POS_LEN = 512
MASKED_POOLING = True


class StudentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_pos_len=512,
        num_buckets=50,
        masked_pooling=False,
    ):
        super().__init__()

        self.masked_pooling = masked_pooling
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_pos_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_buckets)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if self.masked_pooling and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
            x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)

        logits = self.fc(x)
        return logits


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
                    head = f.readline()
                    seq = f.readline()
                    f.readline()
                    f.readline()
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
        idx = int(qid.split("_")[1])
        truth[idx] = sid

    print(f"   Identified true genes for {len(truth)} reads.")
    return truth


def evaluate_student_model(reads, truth, tx_map):
    print("4. Testing STUDENT Model Accuracy...")
    print(f"   Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    model = StudentTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_pos_len=MAX_POS_LEN,
        num_buckets=NUM_LABELS,
        masked_pooling=MASKED_POOLING,
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys (up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (up to 20): {unexpected[:20]}")

    model.to(DEVICE)
    model.eval()

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
            base = true_tx.rsplit(".", 1)[0]
            if base in tx_map:
                valid_buckets = tx_map[base]

        if not valid_buckets:
            continue

        total += 1

        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            _, indices = torch.topk(probs, TOP_K)
            preds = indices[0].cpu().numpy()

        if preds[0] in valid_buckets:
            correct_top1 += 1
        if any(p in valid_buckets for p in preds):
            correct_topk += 1

    print("-" * 30)
    if total > 0:
        print(f"REAL DATA STUDENT ACCURACY (N={total})")
        print(f"Top-1 Accuracy: {correct_top1 / total:.2%}")
        print(f"Top-{TOP_K} Accuracy: {correct_topk / total:.2%}")
    else:
        print("   No reads matched between BLAST and Bucket Map.")
    print("-" * 30)


if __name__ == "__main__":
    tx_map = load_bucket_map()
    reads = get_real_reads()
    truth = get_ground_truth(reads)
    evaluate_student_model(reads, truth, tx_map)