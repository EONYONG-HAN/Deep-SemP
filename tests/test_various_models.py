import os
import gzip
import subprocess
import random
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT         = Path("/data3/projects/2025_Assembly/eyh/c_elegans")
FASTQ_FILE   = "/data3/projects/2025_Assembly/eyh/c_elegans_rep2/raw_data/SRR36114682_1.fastq.gz"
REF_CDNA     = ROOT / "reference/c_elegans_cdna.fa.gz"
MODEL_BASE_NAME = "zhihan1996/DNABERT-2-117M"

SAMPLE_SIZE = 10000
TOP_K       = 5
MAX_LEN     = 100


# ============================================================
# Model configs
# ============================================================

@dataclass
class StudentConfig:
    name:             str
    checkpoint_path:  str
    bucket_map:       str                        # path to semantic_map.tsv
    num_labels:       int   = 50
    d_model:          int   = 384
    nhead:            int   = 8
    num_layers:       int   = 8
    dim_feedforward:  int   = 1024
    dropout:          float = 0.1
    max_pos_len:      int   = 512
    masked_pooling:   bool  = True
    model_type:       str   = "student"          # always "student"


@dataclass
class TeacherConfig:
    name:             str
    checkpoint_path:  str
    bucket_map:       str                        # path to semantic_map.tsv
    num_labels:       int   = 50
    model_type:       str   = "teacher"          # always "teacher"


# ============================================================
# Register models here
# ============================================================
BASE_CKPT = "/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints"
BUCKET_DIR = str(ROOT / "buckets")

MODEL_CONFIGS = [

    # ── Illumina5 students (K=50) ────────────────────────────
    StudentConfig(
        name="student_nh_illumina5_k50",
        checkpoint_path=f"{BASE_CKPT}/track_a_direct_full/distilled_models/"
                        "illumina5_grid/d384_l8_a03_t2_no_hidden/best_student_model.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map.tsv",
        num_labels=50, d_model=384, nhead=8, num_layers=8,
    ),
    StudentConfig(
        name="student_h_illumina5_k50",
        checkpoint_path=f"{BASE_CKPT}/track_a_direct_full/distilled_models/"
                        "illumina5_grid/d384_l8_a03_t2_hidden_b01/best_student_model.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map.tsv",
        num_labels=50, d_model=384, nhead=8, num_layers=8,
    ),

    # ── Clean data student baseline (K=50) ───────────────────
    StudentConfig(
        name="student_clean_l8_k50",
        checkpoint_path=f"{BASE_CKPT}/track_a_direct_full/distilled_models/"
                        "deeper_grid_t2/run_dm384_l8_a03_t2_mask/best_student_model.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map.tsv",
        num_labels=50, d_model=384, nhead=8, num_layers=8,
    ),

    # ── Teachers (K=50) ──────────────────────────────────────
    TeacherConfig(
        name="teacher_clean_k50",
        checkpoint_path=f"{BASE_CKPT}/track_a_direct_full/best_model.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map.tsv",
        num_labels=50,
    ),
    TeacherConfig(
        name="teacher_illumina5_k50_ep5",
        checkpoint_path=f"{BASE_CKPT}/teacher_k50_illumina5/model_epoch_5.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map.tsv",
        num_labels=50,
    ),

    # ── Teachers (K=70) ──────────────────────────────────────
    TeacherConfig(
        name="teacher_illumina5_k70_ep5",
        checkpoint_path=f"{BASE_CKPT}/teacher_k70_illumina5/model_epoch_5.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map_k70.tsv",
        num_labels=70,
    ),

    # ── Teachers (K=90) ──────────────────────────────────────
    TeacherConfig(
        name="teacher_illumina5_k90_ep5",
        checkpoint_path=f"{BASE_CKPT}/teacher_k90_illumina5/model_epoch_5.pt",
        bucket_map=f"{BUCKET_DIR}/semantic_map_k90.tsv",
        num_labels=90,
    ),

    # Add more models here ...
]


# ============================================================
# Student model definition
# ============================================================
class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=8,
                 dim_feedforward=1024, dropout=0.1, max_pos_len=512,
                 num_buckets=50, masked_pooling=True):
        super().__init__()
        self.masked_pooling = masked_pooling
        self.embedding      = nn.Embedding(vocab_size, d_model)
        self.pos_embedding  = nn.Embedding(max_pos_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_buckets)

    def forward(self, input_ids, attention_mask=None):
        B, L    = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        x   = self.transformer(x, src_key_padding_mask=kpm)
        if self.masked_pooling and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x    = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)
        return self.fc(x)


# ============================================================
# Data helpers — run once, shared across all models
# ============================================================
def load_bucket_map(map_path: str) -> dict:
    tx_to_buckets = {}
    with open(map_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            b_id = int(parts[0])
            for tx in parts[3].split(","):
                tx_to_buckets.setdefault(tx.strip(), set()).add(b_id)
    return tx_to_buckets


def get_real_reads() -> list:
    print(f"[1] Sampling {SAMPLE_SIZE} real reads from {FASTQ_FILE}...")
    reads = []
    with gzip.open(FASTQ_FILE, "rt") as f:
        while len(reads) < SAMPLE_SIZE:
            head = f.readline()
            seq  = f.readline().strip()
            f.readline(); f.readline()
            if not head:
                break
            if len(seq) >= 50 and random.random() < 0.1:
                reads.append(seq)
    print(f"   Sampled {len(reads)} reads.")
    return reads


def get_ground_truth(reads: list) -> dict:
    print("[2] Running BLAST...")
    if not os.path.exists("ref_cdna_db.nhr"):
        print("   Building BLAST DB...")
        subprocess.run(f"gunzip -c {REF_CDNA} > temp_ref.fa", shell=True, check=True)
        subprocess.run("makeblastdb -in temp_ref.fa -dbtype nucl -out ref_cdna_db",
                       shell=True, check=True)
        os.remove("temp_ref.fa")

    with open("temp_query.fa", "w") as f:
        for i, seq in enumerate(reads):
            f.write(f">read_{i}\n{seq}\n")

    cmd = ("blastn -query temp_query.fa -db ref_cdna_db "
           "-outfmt '6 qseqid sseqid' -max_target_seqs 1 -evalue 1e-5")
    output = subprocess.check_output(cmd, shell=True).decode()

    truth = {}
    for line in output.splitlines():
        qid, sid = line.split()[:2]
        truth[int(qid.split("_")[1])] = sid
    print(f"   Identified true genes for {len(truth)} reads.")
    return truth


def tokenize_reads(reads, truth, tokenizer) -> list:
    """Tokenize all reads once. Returns list of (idx, input_ids, attn_mask, true_tx)."""
    print("[3] Tokenizing reads (once for all models)...")
    samples = []
    for i, seq in enumerate(reads):
        if i not in truth:
            continue
        inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                           max_length=MAX_LEN, padding="max_length")
        samples.append((i, inputs["input_ids"], inputs["attention_mask"], truth[i]))
    print(f"   {len(samples)} reads tokenized.")
    return samples


def resolve_buckets(true_tx: str, tx_map: dict):
    return tx_map.get(true_tx) or tx_map.get(true_tx.rsplit(".", 1)[0])


# ============================================================
# Model loaders
# ============================================================
def load_student(cfg: StudentConfig, vocab_size: int) -> nn.Module:
    model = StudentTransformer(
        vocab_size=vocab_size, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout, max_pos_len=cfg.max_pos_len,
        num_buckets=cfg.num_labels, masked_pooling=cfg.masked_pooling,
    )
    ckpt       = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"   [WARN] Missing   : {missing[:5]}")
    if unexpected: print(f"   [WARN] Unexpected: {unexpected[:5]}")
    return model.to(DEVICE).eval()


def load_teacher(cfg: TeacherConfig) -> nn.Module:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    config    = AutoConfig.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    config.num_labels = cfg.num_labels
    config.use_cache  = False
    config.__dict__["pad_token_id"] = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    if not hasattr(config, "alibi_starting_size"):
        config.__dict__["alibi_starting_size"] = 512

    with torch.device("cpu"):
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=True
        )

    ckpt       = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"   [WARN] Missing   : {missing[:5]}")
    if unexpected: print(f"   [WARN] Unexpected: {unexpected[:5]}")
    return model.to(DEVICE).eval()


# ============================================================
# Evaluation — works for both student and teacher
# ============================================================
def evaluate_model(cfg, samples: list, vocab_size: int) -> dict:
    print(f"\n{'='*55}")
    print(f"  Evaluating : {cfg.name}  [{cfg.model_type.upper()}]")
    print(f"  Checkpoint : {cfg.checkpoint_path}")
    print(f"  Bucket map : {cfg.bucket_map}  (K={cfg.num_labels})")
    print(f"{'='*55}")

    # Load bucket map for this model's K
    tx_map = load_bucket_map(cfg.bucket_map)

    # Load model
    if cfg.model_type == "student":
        model = load_student(cfg, vocab_size)
    else:
        model = load_teacher(cfg)

    correct_top1 = correct_topk = total = 0

    with torch.no_grad():
        for _, input_ids, attention_mask, true_tx in samples:
            valid_buckets = resolve_buckets(true_tx, tx_map)
            if not valid_buckets:
                continue

            ids  = input_ids.to(DEVICE)
            mask = attention_mask.to(DEVICE)

            if cfg.model_type == "student":
                logits = model(ids, mask)
            else:
                logits = model(input_ids=ids, attention_mask=mask).logits

            _, indices = torch.topk(torch.softmax(logits, dim=-1), TOP_K)
            preds = indices[0].cpu().numpy()

            total += 1
            if preds[0] in valid_buckets:
                correct_top1 += 1
            if any(p in valid_buckets for p in preds):
                correct_topk += 1

    result = {
        "name":      cfg.name,
        "type":      cfg.model_type,
        "K":         cfg.num_labels,
        "n":         total,
        "top1":      correct_top1 / total if total else 0.0,
        f"top{TOP_K}": correct_topk / total if total else 0.0,
    }

    print(f"  N={total} | Top-1: {result['top1']:.2%} | Top-{TOP_K}: {result[f'top{TOP_K}']:.2%}")

    del model
    torch.cuda.empty_cache()
    return result


# ============================================================
# Summary table
# ============================================================
def print_summary(results: list):
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Type':<8} {'K':>4} {'N':>6} {'Top-1':>8} {f'Top-{TOP_K}':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['type']:<8} {r['K']:>4} {r['n']:>6} "
              f"{r['top1']:>8.2%} {r[f'top{TOP_K}']:>8.2%}")
    print(f"{'='*70}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    reads   = get_real_reads()
    truth   = get_ground_truth(reads)
    samples = tokenize_reads(reads, truth, tokenizer)

    results = []
    for cfg in MODEL_CONFIGS:
        # Skip missing checkpoints gracefully
        if not os.path.exists(cfg.checkpoint_path):
            print(f"\n[SKIP] {cfg.name} — checkpoint not found: {cfg.checkpoint_path}")
            continue
        results.append(evaluate_model(cfg, samples, vocab_size))

    print_summary(results)