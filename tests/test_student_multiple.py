import os
import gzip
import subprocess
import random
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")
FASTQ_FILE = "/data3/projects/2025_Assembly/eyh/c_elegans_rep2/raw_data/SRR36114682_1.fastq.gz"
REF_CDNA = ROOT / "reference/c_elegans_cdna.fa.gz"
BUCKET_MAP = ROOT / "buckets/semantic_map.tsv"
MODEL_BASE_NAME = "zhihan1996/DNABERT-2-117M"

SAMPLE_SIZE = 10000
TOP_K = 5
NUM_LABELS = 50
MAX_LEN = 100


# --- Per-model config ---
@dataclass
class ModelConfig:
    name: str
    checkpoint_path: str
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_pos_len: int = 512
    masked_pooling: bool = True
    num_labels: int = NUM_LABELS


# ============================================================
#  Register your models here
# ============================================================
MODEL_CONFIGS = [
    ModelConfig(
        name="d384_l8_a03_t2_nh",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/illumina5_grid/"
                        "d384_l8_a03_t2_no_hidden/best_student_model.pt",
        d_model=384, nhead=8, num_layers=8,
    ),
    ModelConfig(
        name="d384_l8_a03_t2_h",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/illumina5_grid/"
                        "d384_l8_a03_t2_hidden_b01/best_student_model.pt",
        d_model=384, nhead=8, num_layers=8,
    ),
    ModelConfig(
        name="run_dm384_l8_a03_t2_mask",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "run_dm384_l8_a03_t2_mask/best_student_model.pt",
        d_model=384, nhead=8, num_layers=8,
    ),
    ModelConfig(
        name="run_dm384_l8_a05_t2_mask",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "run_dm384_l8_a05_t2_mask/best_student_model.pt",
        d_model=384, nhead=8, num_layers=8,
    ),
    ModelConfig(
        name="run_dm384_l10_a03_t2_mask",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "run_dm384_l10_a03_t2_mask/best_student_model.pt",
        d_model=384, nhead=8, num_layers=10,
    ),
    ModelConfig(
        name="run_dm384_l10_a05_t2_mask",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "run_dm384_l10_a05_t2_mask/best_student_model.pt",
        d_model=384, nhead=8, num_layers=10,
    ),
    ModelConfig(
        name="run_dm384_l12_a03_t2_mask",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "run_dm384_l12_a03_t2_mask/best_student_model.pt",
        d_model=384, nhead=8, num_layers=12,
    ),
    ModelConfig(
        name="run_dm384_l12_a05_t2_mask",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "run_dm384_l12_a05_t2_mask/best_student_model.pt",
        d_model=384, nhead=8, num_layers=12,
    ),
    ModelConfig(
        name="d384_l6_a0.3_t2",
        checkpoint_path="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/"
                        "track_a_direct_full/distilled_models/deeper_grid_t2/"
                        "d384_l6_a0.3_t2/best_student_model.pt",
        d_model=384, nhead=8, num_layers=6,
    ),


    # Add more models here ...
]


class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_pos_len=512,
                 num_buckets=50, masked_pooling=False):
        super().__init__()
        self.masked_pooling = masked_pooling
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_pos_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_buckets)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        if self.masked_pooling and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)
        return self.fc(x)


# ── Data loading (unchanged, runs once) ─────────────────────────────────────

def load_bucket_map():
    print(f"[1] Loading bucket map from {BUCKET_MAP}...")
    tx_to_buckets = {}
    with open(BUCKET_MAP) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            b_id = int(parts[0])
            for tx in parts[3].split(","):
                tx = tx.strip()
                tx_to_buckets.setdefault(tx, set()).add(b_id)
    print(f"   Mapped {len(tx_to_buckets)} transcripts.")
    return tx_to_buckets


def get_real_reads():
    print(f"[2] Sampling {SAMPLE_SIZE} real reads from {FASTQ_FILE}...")
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


def get_ground_truth(reads):
    print("[3] Running BLAST...")
    if not os.path.exists("ref_cdna_db.nhr"):
        print("   Building BLAST DB...")
        subprocess.run(f"gunzip -c {REF_CDNA} > temp_ref.fa", shell=True, check=True)
        subprocess.run("makeblastdb -in temp_ref.fa -dbtype nucl -out ref_cdna_db", shell=True, check=True)
        os.remove("temp_ref.fa")

    with open("temp_query.fa", "w") as f:
        for i, seq in enumerate(reads):
            f.write(f">read_{i}\n{seq}\n")

    cmd = "blastn -query temp_query.fa -db ref_cdna_db -outfmt '6 qseqid sseqid' -max_target_seqs 1 -evalue 1e-5"
    output = subprocess.check_output(cmd, shell=True).decode()

    truth = {}
    for line in output.splitlines():
        qid, sid = line.split()[:2]
        truth[int(qid.split("_")[1])] = sid
    print(f"   Identified true genes for {len(truth)} reads.")
    return truth


# ── Tokenize once, reuse across models ───────────────────────────────────────

def tokenize_reads(reads, truth, tx_map, tokenizer):
    """Returns list of (index, input_ids, attention_mask, valid_buckets)."""
    print("[4] Tokenizing reads (once for all models)...")
    samples = []
    for i, seq in enumerate(reads):
        if i not in truth:
            continue
        true_tx = truth[i]
        valid_buckets = tx_map.get(true_tx) or tx_map.get(true_tx.rsplit(".", 1)[0])
        if not valid_buckets:
            continue

        inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                           max_length=MAX_LEN, padding="max_length")
        samples.append((i, inputs["input_ids"], inputs["attention_mask"], valid_buckets))

    print(f"   {len(samples)} reads ready for evaluation.")
    return samples


# ── Single-model evaluation ───────────────────────────────────────────────────

def load_model(cfg: ModelConfig, vocab_size: int) -> StudentTransformer:
    model = StudentTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        max_pos_len=cfg.max_pos_len, num_buckets=cfg.num_labels,
        masked_pooling=cfg.masked_pooling,
    )
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"   [WARN] Missing keys   : {missing[:20]}")
    if unexpected: print(f"   [WARN] Unexpected keys: {unexpected[:20]}")
    return model.to(DEVICE)


def evaluate_model(cfg: ModelConfig, samples, vocab_size: int) -> dict:
    print(f"\n{'='*55}")
    print(f"  Evaluating: {cfg.name}")
    print(f"  d_model={cfg.d_model}, nhead={cfg.nhead}, num_layers={cfg.num_layers}")
    print(f"{'='*55}")

    model = load_model(cfg, vocab_size)
    model.eval()

    correct_top1 = correct_topk = total = 0

    with torch.no_grad():
        for _, input_ids, attention_mask, valid_buckets in samples:
            logits = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))
            _, indices = torch.topk(torch.softmax(logits, dim=-1), TOP_K)
            preds = indices[0].cpu().numpy()

            total += 1
            if preds[0] in valid_buckets:   correct_top1 += 1
            if any(p in valid_buckets for p in preds): correct_topk += 1

    result = {
        "name":     cfg.name,
        "n":        total,
        "top1":     correct_top1 / total if total else 0.0,
        f"top{TOP_K}": correct_topk / total if total else 0.0,
    }

    print(f"  N={total}  |  Top-1: {result['top1']:.2%}  |  Top-{TOP_K}: {result[f'top{TOP_K}']:.2%}")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

    return result


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    header = f"{'Model':<30} {'N':>6} {'Top-1':>8} {f'Top-{TOP_K}':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<30} {r['n']:>6} {r['top1']:>8.2%} {r[f'top{TOP_K}']:>8.2%}")
    print(f"{'='*55}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tx_map   = load_bucket_map()
    reads    = get_real_reads()
    truth    = get_ground_truth(reads)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    # Tokenize ONCE — shared across all models
    samples = tokenize_reads(reads, truth, tx_map, tokenizer)

    results = []
    for cfg in MODEL_CONFIGS:
        results.append(evaluate_model(cfg, samples, vocab_size))

    print_summary(results)