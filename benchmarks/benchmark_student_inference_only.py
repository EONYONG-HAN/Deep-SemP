import os
import gzip
import argparse
import time
import collections
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

# =========================
# Default config
# =========================
DEFAULT_MODEL_PATH = "/data3/projects/2025_Assembly/eyh/models/checkpoints/track_a_direct_full/distilled_models/run_dm384_l6_a03_t2_mask/best_student_model.pt"
DEFAULT_R1 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_1.fastq.gz"
DEFAULT_R2 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_2.fastq.gz"

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
NUM_LABELS = 50


# =========================
# Student model
# =========================
class StudentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=384,
        nhead=8,
        num_layers=6,
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


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only benchmark for distilled Student model")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--r1", type=str, default=DEFAULT_R1)
    parser.add_argument("--r2", type=str, default=DEFAULT_R2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=100)

    # student architecture: must match checkpoint
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_pos_len", type=int, default=512)
    parser.add_argument("--masked_pooling", action="store_true")

    # runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=100000)
    parser.add_argument("--limit_pairs", type=int, default=0, help="0 means process all pairs")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision on CUDA")

    return parser.parse_args()


# =========================
# Load model
# =========================
def load_student_model(args):
    print("1. Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"2. Building student model on {device}...")
    model = StudentTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_pos_len=args.max_pos_len,
        num_buckets=NUM_LABELS,
        masked_pooling=args.masked_pooling,
    )

    print(f"3. Loading weights from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location="cpu")
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

    if device.type == "cuda":
        print("   GPU:", torch.cuda.get_device_name(0))

    return model, tokenizer, device


# =========================
# Inference batch
# =========================
def infer_batch(batch_seqs, model, tokenizer, device, max_len, use_amp=False):
    inputs = tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    input_ids = inputs["input_ids"].to(device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)

        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    return preds


# =========================
# Main benchmark
# =========================
def main():
    args = parse_args()
    model, tokenizer, device = load_student_model(args)

    print(f"Processing pairs: {Path(args.r1).name} & {Path(args.r2).name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Masked pooling: {args.masked_pooling}")
    print(f"AMP: {args.use_amp}")
    print(f"Limit pairs: {args.limit_pairs if args.limit_pairs > 0 else 'ALL'}")

    batch_seqs = []
    total_pairs = 0
    bucket_counter = collections.Counter()

    start_time = time.time()

    with gzip.open(args.r1, "rt") as f1, gzip.open(args.r2, "rt") as f2:
        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            if not line1 or not line2:
                break

            seq1 = f1.readline().strip()
            f1.readline()
            f1.readline()

            f2.readline()
            f2.readline()
            f2.readline()

            batch_seqs.append(seq1)

            if len(batch_seqs) >= args.batch_size:
                preds = infer_batch(
                    batch_seqs, model, tokenizer, device,
                    args.max_len, args.use_amp
                )
                bucket_counter.update(int(x) for x in preds)
                total_pairs += len(batch_seqs)
                batch_seqs = []

                if total_pairs % args.log_every == 0:
                    elapsed = time.time() - start_time
                    rate = total_pairs / elapsed if elapsed > 0 else 0.0
                    print(f"Processed {total_pairs:,} pairs | {rate:.2f} pairs/sec", flush=True)

                if args.limit_pairs > 0 and total_pairs >= args.limit_pairs:
                    break

        if batch_seqs and (args.limit_pairs == 0 or total_pairs < args.limit_pairs):
            preds = infer_batch(
                batch_seqs, model, tokenizer, device,
                args.max_len, args.use_amp
            )
            bucket_counter.update(int(x) for x in preds)
            total_pairs += len(batch_seqs)

    elapsed = time.time() - start_time
    rate = total_pairs / elapsed if elapsed > 0 else 0.0

    print("\n===== Inference-only Benchmark Done =====")
    print(f"Total pairs processed: {total_pairs:,}")
    print(f"Elapsed time: {elapsed / 3600:.2f} hours")
    print(f"Average speed: {rate:.2f} pairs/sec")
    print("Top 10 predicted buckets:")
    for bucket_id, count in bucket_counter.most_common(10):
        print(f"  bucket_{bucket_id}: {count:,}")


if __name__ == "__main__":
    main()