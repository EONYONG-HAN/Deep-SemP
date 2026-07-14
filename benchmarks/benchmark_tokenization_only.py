import os
import gzip
import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer

# =========================
# Default config
# =========================
DEFAULT_R1 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_1.fastq.gz"
DEFAULT_R2 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_2.fastq.gz"
MODEL_NAME = "zhihan1996/DNABERT-2-117M"

def parse_args():
    parser = argparse.ArgumentParser(description="FASTQ read + tokenization-only benchmark")
    parser.add_argument("--r1", type=str, default=DEFAULT_R1)
    parser.add_argument("--r2", type=str, default=DEFAULT_R2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=100000)
    parser.add_argument("--limit_pairs", type=int, default=0, help="0 means process all pairs")
    return parser.parse_args()

def main():
    args = parse_args()

    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Processing pairs: {Path(args.r1).name} & {Path(args.r2).name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_len}")
    print(f"Limit pairs: {args.limit_pairs if args.limit_pairs > 0 else 'ALL'}")

    batch_seqs = []
    total_pairs = 0
    total_tokens = 0
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

            # consume R2 in sync, but do not use it
            f2.readline()
            f2.readline()
            f2.readline()

            batch_seqs.append(seq1)

            if len(batch_seqs) >= args.batch_size:
                inputs = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_len,
                )
                total_tokens += inputs["input_ids"].numel()
                total_pairs += len(batch_seqs)
                batch_seqs = []

                if total_pairs % args.log_every == 0:
                    elapsed = time.time() - start_time
                    rate = total_pairs / elapsed if elapsed > 0 else 0.0
                    toks = total_tokens / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Processed {total_pairs:,} pairs | "
                        f"{rate:.2f} pairs/sec | "
                        f"{toks:.2f} tokenized elems/sec",
                        flush=True
                    )

                if args.limit_pairs > 0 and total_pairs >= args.limit_pairs:
                    break

        if batch_seqs and (args.limit_pairs == 0 or total_pairs < args.limit_pairs):
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_len,
            )
            total_tokens += inputs["input_ids"].numel()
            total_pairs += len(batch_seqs)

    elapsed = time.time() - start_time
    rate = total_pairs / elapsed if elapsed > 0 else 0.0
    toks = total_tokens / elapsed if elapsed > 0 else 0.0

    print("\n===== Tokenization-only Benchmark Done =====")
    print(f"Total pairs processed: {total_pairs:,}")
    print(f"Elapsed time: {elapsed / 3600:.2f} hours")
    print(f"Average speed: {rate:.2f} pairs/sec")
    print(f"Average tokenized elems/sec: {toks:.2f}")

if __name__ == "__main__":
    main()