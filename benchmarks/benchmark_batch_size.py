"""
Quick benchmark to find optimal batch size for student model routing.
Tests each batch size on the first N pairs and reports throughput.
Run before committing to full dataset routing.
"""
import os
import gzip
import time
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_NAME  = "zhihan1996/DNABERT-2-117M"
WARMUP_PAIRS = 1000    # discard first N pairs (GPU warmup)
MEASURE_PAIRS = 20000  # measure over this many pairs per batch size


# ==========================================
# Student Model
# ==========================================
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


def parse_args():
    parser = argparse.ArgumentParser(description="Batch size benchmark for student routing")
    parser.add_argument("--model_path",      type=str, required=True)
    parser.add_argument("--r1",              type=str, required=True)
    parser.add_argument("--num_labels",      type=int, default=50)
    parser.add_argument("--d_model",         type=int, default=384)
    parser.add_argument("--nhead",           type=int, default=8)
    parser.add_argument("--num_layers",      type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_pos_len",     type=int, default=512)
    parser.add_argument("--masked_pooling",  action="store_true")
    parser.add_argument("--max_len",         type=int, default=100)
    parser.add_argument("--gpu",             type=str, default="2")
    parser.add_argument("--fp16",            action="store_true")
    parser.add_argument("--batch_sizes",     type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048, 4096, 8192, 9216, 10240,11264,12288,13312,14336,15360, 16384],
                        help="List of batch sizes to test")
    parser.add_argument("--warmup_pairs",    type=int, default=WARMUP_PAIRS)
    parser.add_argument("--measure_pairs",   type=int, default=MEASURE_PAIRS)
    return parser.parse_args()


def load_reads(fastq_path, n):
    """Load first n reads from FASTQ into memory for reuse across batch sizes."""
    reads = []
    opener = gzip.open if fastq_path.endswith(".gz") else open
    with opener(fastq_path, "rt") as f:
        while len(reads) < n:
            header = f.readline().strip()
            seq    = f.readline().strip()
            f.readline(); f.readline()
            if not header:
                break
            if len(seq) >= 50:
                reads.append(seq)
    return reads


def benchmark_one(model, tokenizer, reads, batch_size, max_len,
                  warmup_pairs, measure_pairs, device, fp16):
    """Run inference on pre-loaded reads at given batch_size. Returns pairs/sec."""

    total_needed = warmup_pairs + measure_pairs
    # Repeat reads if needed
    all_reads = (reads * ((total_needed // len(reads)) + 1))[:total_needed]

    # Warmup
    for i in range(0, warmup_pairs, batch_size):
        batch = all_reads[i: i + batch_size]
        if not batch:
            break
        enc = tokenizer(batch, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        if fp16:
            ids = ids
            with torch.no_grad():
                model(ids, mask)
        else:
            with torch.no_grad():
                model(ids, mask)

    torch.cuda.synchronize()

    # Timed measurement
    t0 = time.time()
    processed = 0
    for i in range(warmup_pairs, total_needed, batch_size):
        batch = all_reads[i: i + batch_size]
        if not batch:
            break
        enc = tokenizer(batch, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            model(ids, mask)
        processed += len(batch)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return processed / elapsed if elapsed > 0 else 0.0


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"fp16   : {args.fp16}")
    print(f"Warmup : {args.warmup_pairs:,} pairs | Measure: {args.measure_pairs:,} pairs")
    print()

    # Load tokenizer and model once
    print("Loading tokenizer...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    print(f"Loading model (l={args.num_layers}, d={args.d_model})...")
    model = StudentTransformer(
        vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        max_pos_len=args.max_pos_len, num_buckets=args.num_labels,
        masked_pooling=args.masked_pooling,
    )
    ckpt       = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    if args.fp16:
        model.half()
        print("Using float16 (half precision)")

    # Pre-load reads once
    n_needed = args.warmup_pairs + args.measure_pairs
    print(f"Pre-loading {n_needed:,} reads from {args.r1}...")
    reads = load_reads(args.r1, n_needed)
    print(f"Loaded {len(reads):,} reads.\n")

    # Benchmark each batch size
    results = []
    print(f"{'Batch size':>12} | {'pairs/sec':>12} | {'Est. time (75M pairs)':>22} | VRAM used")
    print("-" * 70)

    for bs in args.batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats()
            speed = benchmark_one(
                model, tokenizer, reads, bs,
                args.max_len, args.warmup_pairs, args.measure_pairs,
                device, args.fp16,
            )
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            est_hrs = 75_000_000 / speed / 3600 if speed > 0 else float("inf")
            results.append((bs, speed, est_hrs, vram_mb))
            print(f"{bs:>12,} | {speed:>12,.0f} | {est_hrs:>20.2f}h | {vram_mb:>6.0f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>12,} | {'OOM':>12} | {'':>22} |")
            torch.cuda.empty_cache()

    # Summary
    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n{'='*70}")
        print(f"  Optimal batch size : {best[0]:,}")
        print(f"  Best throughput    : {best[1]:,.0f} pairs/sec")
        print(f"  Est. time (75M)    : {best[2]:.2f} hours")
        print(f"  VRAM at peak       : {best[3]:.0f} MB")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()