import os
import gzip
import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

# =========================
# Default config
# =========================
DEFAULT_MODEL_PATH = "/data3/projects/2025_Assembly/eyh/models/checkpoints/track_a_direct_full/distilled_models/weekend_grid/run_dm384_l6_a05_t3_mask/best_student_model.pt"
DEFAULT_R1 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_1.fastq.gz"
DEFAULT_R2 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_2.fastq.gz"
DEFAULT_OUTPUT_DIR = "/data3/projects/2025_Assembly/eyh/c_elegans/partitioned_reads_student_case2"

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
    parser = argparse.ArgumentParser(description="Paired-End Partitioning using distilled Student model")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--r1", type=str, default=DEFAULT_R1)
    parser.add_argument("--r2", type=str, default=DEFAULT_R2)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=100)

    # student architecture: must match training run
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

    return parser.parse_args()


# =========================
# Model loading
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
# Inference
# =========================
def process_batch(batch_r1, batch_r2, model, tokenizer, device, file_handles, max_len):
    inputs = tokenizer(
        [r["seq"] for r in batch_r1],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    for r1, r2, bucket_id in zip(batch_r1, batch_r2, preds):
        h1, h2 = file_handles[int(bucket_id)]
        h1.write(f"@{r1['id']}\n{r1['seq']}\n+\n{r1['qual']}\n")
        h2.write(f"@{r2['id']}\n{r2['seq']}\n+\n{r2['qual']}\n")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening {NUM_LABELS} output pairs in {args.output_dir} ...")
    file_handles = {}
    for i in range(NUM_LABELS):
        f1 = open(os.path.join(args.output_dir, f"bucket_{i}_R1.fastq"), "w")
        f2 = open(os.path.join(args.output_dir, f"bucket_{i}_R2.fastq"), "w")
        file_handles[i] = (f1, f2)

    model, tokenizer, device = load_student_model(args)

    print(f"Processing pairs: {Path(args.r1).name} & {Path(args.r2).name} ...")
    print(f"Batch size: {args.batch_size}")
    print(f"Masked pooling: {args.masked_pooling}")

    batch_r1, batch_r2 = [], []
    total_pairs = 0
    start_time = time.time()

    with gzip.open(args.r1, "rt") as f1, gzip.open(args.r2, "rt") as f2:
        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            if not line1 or not line2:
                break

            r1 = {
                "id": line1.strip().replace("@", ""),
                "seq": f1.readline().strip()
            }
            f1.readline()
            r1["qual"] = f1.readline().strip()

            r2 = {
                "id": line2.strip().replace("@", ""),
                "seq": f2.readline().strip()
            }
            f2.readline()
            r2["qual"] = f2.readline().strip()

            batch_r1.append(r1)
            batch_r2.append(r2)

            if len(batch_r1) >= args.batch_size:
                process_batch(
                    batch_r1, batch_r2,
                    model, tokenizer, device,
                    file_handles, args.max_len
                )
                total_pairs += len(batch_r1)
                batch_r1, batch_r2 = [], []

                if total_pairs % args.log_every == 0:
                    elapsed = time.time() - start_time
                    rate = total_pairs / elapsed if elapsed > 0 else 0.0
                    print(f"Processed {total_pairs:,} pairs | {rate:.2f} pairs/sec", flush=True)

        if batch_r1:
            process_batch(
                batch_r1, batch_r2,
                model, tokenizer, device,
                file_handles, args.max_len
            )
            total_pairs += len(batch_r1)

    for h1, h2 in file_handles.values():
        h1.close()
        h2.close()

    elapsed = time.time() - start_time
    rate = total_pairs / elapsed if elapsed > 0 else 0.0
    print(f"\nDone! Partitioned {total_pairs:,} pairs into {args.output_dir}")
    print(f"Elapsed time: {elapsed / 3600:.2f} hours")
    print(f"Average speed: {rate:.2f} pairs/sec")


if __name__ == "__main__":
    main()