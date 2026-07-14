import os
import gzip
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# =========================
# Constants
# =========================
MODEL_NAME = "zhihan1996/DNABERT-2-117M"


# =========================
# Student model
# =========================
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
        # Ensure dtype matches fc layer before final projection
        x = x.to(self.fc.weight.dtype)
        return self.fc(x)


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired-end read partitioning using distilled student model"
    )

    # I/O
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Path to best_student_model.pt")
    parser.add_argument("--r1",          type=str, required=True,
                        help="R1 FASTQ (.fastq or .fastq.gz)")
    parser.add_argument("--r2",          type=str, required=True,
                        help="R2 FASTQ (.fastq or .fastq.gz)")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Output directory for partitioned FASTQ bins")

    # Model architecture — must match training run
    parser.add_argument("--num_labels",      type=int,   default=50)
    parser.add_argument("--d_model",         type=int,   default=384)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=8)   # updated default l8
    parser.add_argument("--dim_feedforward", type=int,   default=1024)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--max_pos_len",     type=int,   default=512)
    parser.add_argument("--masked_pooling",  action="store_true")

    # Runtime
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--max_len",     type=int, default=100)
    parser.add_argument("--log_every",   type=int, default=100000,
                        help="Print progress every N pairs")
    parser.add_argument("--fp16",        action="store_true",
                        help="Use half precision (float16) for faster inference")

    return parser.parse_args()


# =========================
# Model loading
# =========================
def load_student_model(args, device):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Building student model (d={args.d_model}, l={args.num_layers}, "
          f"K={args.num_labels}, masked_pooling={args.masked_pooling})...")
    model = StudentTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, max_pos_len=args.max_pos_len,
        num_buckets=args.num_labels, masked_pooling=args.masked_pooling,
    )

    print(f"Loading weights from {args.model_path}...")
    ckpt       = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"[WARN] Missing keys    : {missing[:5]}")
    if unexpected: print(f"[WARN] Unexpected keys : {unexpected[:5]}")

    model.to(device).eval()
    if args.fp16 and device.type == "cuda":
        model.half()
        print("Using float16 (half precision) for inference")
    print(f"Model ready on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return model, tokenizer


# =========================
# Batch inference
# =========================
def process_batch(batch_r1, batch_r2, model, tokenizer,
                  device, file_handles, max_len, bucket_counts):
    seqs = [r["seq"] for r in batch_r1]
    inputs = tokenizer(
        seqs, return_tensors="pt",
        padding="max_length", truncation=True, max_length=max_len,
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds  = torch.argmax(logits, dim=-1).cpu().numpy()

    for r1, r2, bucket_id in zip(batch_r1, batch_r2, preds):
        bid = int(bucket_id)
        h1, h2 = file_handles[bid]
        h1.write(f"@{r1['id']}\n{r1['seq']}\n+\n{r1['qual']}\n")
        h2.write(f"@{r2['id']}\n{r2['seq']}\n+\n{r2['qual']}\n")
        bucket_counts[bid] += 1


# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Open output file handles ──────────────────────────────────
    print(f"Opening {args.num_labels} output FASTQ pairs in {args.output_dir}...")
    file_handles = {}
    for i in range(args.num_labels):
        f1 = open(os.path.join(args.output_dir, f"bucket_{i:02d}_R1.fastq"), "w")
        f2 = open(os.path.join(args.output_dir, f"bucket_{i:02d}_R2.fastq"), "w")
        file_handles[i] = (f1, f2)

    # ── Load model ────────────────────────────────────────────────
    model, tokenizer = load_student_model(args, device)

    # ── Routing ───────────────────────────────────────────────────
    print(f"\nRouting reads:")
    print(f"  R1 : {args.r1}")
    print(f"  R2 : {args.r2}")
    print(f"  batch_size : {args.batch_size}")
    print()

    bucket_counts = defaultdict(int)
    batch_r1, batch_r2 = [], []
    total_pairs  = 0

    t_io_total      = 0.0
    t_infer_total   = 0.0
    t_write_total   = 0.0
    wall_start      = time.time()

def open_fastq(path: str):
    """Open FASTQ file — supports .gz, .zst, and plain text."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    elif path.endswith(".zst"):
        # Stream zstd via subprocess — no Python zstd dependency needed
        proc = subprocess.Popen(
            ["zstd", "-d", "--stdout", path],
            stdout=subprocess.PIPE,
        )
        import io
        return io.TextIOWrapper(proc.stdout, encoding="utf-8")
    else:
        return open(path, "r")

    with open_fastq(args.r1) as f1, open_fastq(args.r2) as f2:
        while True:
            # ── Read one pair ──────────────────────────────────────
            t0 = time.time()
            h1_line = f1.readline()
            h2_line = f2.readline()
            if not h1_line or not h2_line:
                break

            r1 = {"id": h1_line.strip().lstrip("@"),
                  "seq":  f1.readline().strip(),
                  "qual": (f1.readline(), f1.readline().strip())[1]}
            r2 = {"id": h2_line.strip().lstrip("@"),
                  "seq":  f2.readline().strip(),
                  "qual": (f2.readline(), f2.readline().strip())[1]}
            t_io_total += time.time() - t0

            batch_r1.append(r1)
            batch_r2.append(r2)

            if len(batch_r1) >= args.batch_size:
                # ── Inference ───────────────────────────────────────
                t1 = time.time()
                seqs   = [r["seq"] for r in batch_r1]
                inputs = tokenizer(seqs, return_tensors="pt",
                                   padding="max_length", truncation=True,
                                   max_length=args.max_len)
                iids  = inputs["input_ids"].to(device)
                amask = inputs["attention_mask"].to(device)
                with torch.no_grad():
                    preds = torch.argmax(model(iids, amask), dim=-1).cpu().numpy()
                t_infer_total += time.time() - t1

                # ── Write ────────────────────────────────────────────
                t2 = time.time()
                for r1, r2, bid in zip(batch_r1, batch_r2, preds):
                    bid = int(bid)
                    h1, h2 = file_handles[bid]
                    h1.write(f"@{r1['id']}\n{r1['seq']}\n+\n{r1['qual']}\n")
                    h2.write(f"@{r2['id']}\n{r2['seq']}\n+\n{r2['qual']}\n")
                    bucket_counts[bid] += 1
                t_write_total += time.time() - t2

                total_pairs += len(batch_r1)
                batch_r1, batch_r2 = [], []

                if total_pairs % args.log_every == 0:
                    elapsed = time.time() - wall_start
                    rate    = total_pairs / elapsed
                    print(f"  {total_pairs:>10,} pairs | "
                          f"{rate:>8.0f} pairs/sec | "
                          f"elapsed: {elapsed/3600:.2f}h", flush=True)

        # ── Flush remaining batch ─────────────────────────────────
        if batch_r1:
            seqs   = [r["seq"] for r in batch_r1]
            inputs = tokenizer(seqs, return_tensors="pt",
                               padding="max_length", truncation=True,
                               max_length=args.max_len)
            iids  = inputs["input_ids"].to(device)
            amask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                preds = torch.argmax(model(iids, amask), dim=-1).cpu().numpy()
            for r1, r2, bid in zip(batch_r1, batch_r2, preds):
                bid = int(bid)
                h1, h2 = file_handles[bid]
                h1.write(f"@{r1['id']}\n{r1['seq']}\n+\n{r1['qual']}\n")
                h2.write(f"@{r2['id']}\n{r2['seq']}\n+\n{r2['qual']}\n")
                bucket_counts[bid] += 1
            total_pairs += len(batch_r1)

    # ── Close file handles ────────────────────────────────────────
    for h1, h2 in file_handles.values():
        h1.close()
        h2.close()

    # ── Timing summary ────────────────────────────────────────────
    wall_elapsed = time.time() - wall_start
    rate         = total_pairs / wall_elapsed if wall_elapsed > 0 else 0

    print(f"\n{'='*55}")
    print(f"  ROUTING COMPLETE")
    print(f"{'='*55}")
    print(f"  Total pairs routed : {total_pairs:,}")
    print(f"  Output directory   : {args.output_dir}")
    print(f"  Num buckets        : {args.num_labels}")
    print(f"")
    print(f"  Wall time          : {wall_elapsed/3600:.3f} hours  "
          f"({wall_elapsed:.1f} sec)")
    print(f"  Avg speed          : {rate:.1f} pairs/sec")
    print(f"")
    print(f"  Time breakdown:")
    print(f"    I/O (read)       : {t_io_total:.1f}s  "
          f"({100*t_io_total/wall_elapsed:.1f}%)")
    print(f"    Inference        : {t_infer_total:.1f}s  "
          f"({100*t_infer_total/wall_elapsed:.1f}%)")
    print(f"    Write            : {t_write_total:.1f}s  "
          f"({100*t_write_total/wall_elapsed:.1f}%)")

    # ── Per-bucket distribution ───────────────────────────────────
    print(f"\n  Per-bucket read distribution:")
    print(f"  {'Bucket':>8} | {'Pairs':>10} | {'%':>6}")
    print(f"  {'-'*30}")
    for bid in range(args.num_labels):
        count = bucket_counts.get(bid, 0)
        pct   = 100.0 * count / total_pairs if total_pairs > 0 else 0
        print(f"  {bid:>8} | {count:>10,} | {pct:>5.2f}%")

    # Save distribution to TSV
    dist_path = os.path.join(args.output_dir, "bucket_distribution.tsv")
    with open(dist_path, "w") as f:
        f.write("bucket\tpairs\tpercent\n")
        for bid in range(args.num_labels):
            count = bucket_counts.get(bid, 0)
            pct   = 100.0 * count / total_pairs if total_pairs > 0 else 0
            f.write(f"{bid}\t{count}\t{pct:.4f}\n")
    print(f"\n  Distribution saved to: {dist_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()