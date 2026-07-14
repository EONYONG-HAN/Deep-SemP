import os
import sys
import math
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


# ==========================================
# 1. Configuration & Arguments
# ==========================================
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
NUM_LABELS = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Deep-SemP")

    # data / paths
    parser.add_argument("--teacher_weights", type=str, required=True, help="Path to best Deep-SemP model (.pt)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to full training CSV")
    parser.add_argument("--output_dir", type=str, default="./distilled_models")
    parser.add_argument("--log_file", type=str, default=None, help="Default: output_dir/distillation.log")

    # split / loader
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--stratify_split", action="store_true", help="Use stratified train/val split by label")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=100)

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)

    # student architecture
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_pos_len", type=int, default=512)
    parser.add_argument("--masked_pooling", action="store_true", help="Use attention-mask-aware mean pooling")
    parser.add_argument("--student_type", type=str, default="transformer", choices=["transformer", "lite_mlp"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    return parser.parse_args()


# ==========================================
# 2. Logging
# ==========================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(log_path):
    log_f = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    return log_f


# ==========================================
# 3. Dataset & Tokenization
# ==========================================
class SequenceLabelDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label = int(self.labels[idx])

        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


# ==========================================
# 4. Teacher Loading
# ==========================================
def load_teacher(model_path, device):
    print(f"Loading Teacher Model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    config.num_labels = NUM_LABELS
    config.use_cache = False

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    config.__dict__["pad_token_id"] = pad_id

    if not hasattr(config, "alibi_starting_size"):
        config.__dict__["alibi_starting_size"] = 512

    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    print("config.pad_token_id:", config.__dict__.get("pad_token_id"))
    print("torch default device:", torch.get_default_device())

    with torch.device("cpu"):
        teacher = AutoModelForSequenceClassification.from_config(
            config,
            trust_remote_code=True
        )

    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print("First few missing keys:", missing[:10])
    if unexpected:
        print("First few unexpected keys:", unexpected[:10])

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


# ==========================================
# 5. Student Model
# ==========================================

class StudentLiteMLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        hidden_dim=256,
        max_pos_len=512,
        num_buckets=50,
        dropout=0.1,
        masked_pooling=True,
    ):
        super().__init__()
        self.masked_pooling = masked_pooling

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_pos_len, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_buckets),
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        if self.masked_pooling and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
            x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)

        return self.mlp(x)

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


# ==========================================
# 6. Metrics / Loss
# ==========================================
def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
    hard_loss = F.cross_entropy(student_logits, true_labels)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl_loss = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)
    total_loss = (alpha * hard_loss) + ((1 - alpha) * kl_loss * (temperature * temperature))
    return total_loss, hard_loss.detach(), kl_loss.detach()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_student(student, teacher, val_loader, device):
    student.eval()
    correct_student, correct_teacher, agreements, total = 0, 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

            student_logits = student(input_ids, attention_mask)

            student_preds = torch.argmax(student_logits, dim=1)
            teacher_preds = torch.argmax(teacher_logits, dim=1)

            total += labels.size(0)
            correct_student += (student_preds == labels).sum().item()
            correct_teacher += (teacher_preds == labels).sum().item()
            agreements += (student_preds == teacher_preds).sum().item()

    student_acc = 100.0 * correct_student / total
    teacher_acc = 100.0 * correct_teacher / total
    agreement_rate = 100.0 * agreements / total

    print(f"\n--- Validation Results ---")
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Accuracy: {student_acc:.2f}%")
    print(f"Agreement Rate:   {agreement_rate:.2f}%\n")

    student.train()
    return {
        "teacher_acc": teacher_acc,
        "student_acc": student_acc,
        "agreement_rate": agreement_rate,
    }


# ==========================================
# 7. Training
# ==========================================
def train_distillation(student, teacher, train_loader, val_loader, optimizer, scheduler, args, device):
    best_agreement = -1.0
    best_metrics = None

    for epoch in range(args.epochs):
        student.train()
        running_total_loss = 0.0
        running_hard_loss = 0.0
        running_kl_loss = 0.0

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_logits = student(input_ids, attention_mask)

            loss, hard_loss, kl_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                args.temperature,
                args.alpha,
            )
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_total_loss += loss.item()
            running_hard_loss += hard_loss.item()
            running_kl_loss += kl_loss.item()

            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] | "
                    f"Batch {batch_idx} | "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Hard CE: {hard_loss.item():.4f} | "
                    f"Soft KL: {kl_loss.item():.4f} | "
                    f"LR: {current_lr:.6e}"
                )

        avg_total_loss = running_total_loss / len(train_loader)
        avg_hard_loss = running_hard_loss / len(train_loader)
        avg_kl_loss = running_kl_loss / len(train_loader)

        print(f"=== End of Epoch {epoch+1} | Avg Total Loss: {avg_total_loss:.4f} | Avg Hard CE: {avg_hard_loss:.4f} | Avg Soft KL: {avg_kl_loss:.4f} ===")

        metrics = evaluate_student(student, teacher, val_loader, device)

        if metrics["agreement_rate"] > best_agreement:
            best_agreement = metrics["agreement_rate"]
            best_metrics = {
                "epoch": epoch + 1,
                "teacher_acc": metrics["teacher_acc"],
                "student_acc": metrics["student_acc"],
                "agreement_rate": metrics["agreement_rate"],
                "avg_total_loss": avg_total_loss,
                "avg_hard_loss": avg_hard_loss,
                "avg_kl_loss": avg_kl_loss,
            }

            save_path = os.path.join(args.output_dir, "best_student_model.pt")
            torch.save(student.state_dict(), save_path)
            print(f">>> New best Student model saved to {save_path}! <<<\n")

            metrics_path = os.path.join(args.output_dir, "best_metrics.txt")
            with open(metrics_path, "w") as f:
                for k, v in best_metrics.items():
                    f.write(f"{k}\t{v}\n")

    if best_metrics is not None:
        print("===== Best Validation Summary =====")
        for k, v in best_metrics.items():
            print(f"{k}: {v}")


# ==========================================
# 8. Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = args.log_file if args.log_file is not None else os.path.join(args.output_dir, "distillation.log")
    log_handle = setup_logging(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("torch default device:", torch.get_default_device())
    if torch.cuda.is_available():
        print("Visible GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    if args.stratify_split:
        train_df, val_df = train_test_split(
            df,
            test_size=args.val_split,
            random_state=42,
            stratify=df["label"],
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=args.val_split,
            random_state=42,
        )

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    train_dataset = SequenceLabelDataset(
        train_df.sequence.to_numpy(),
        train_df.label.to_numpy(),
        tokenizer,
        args.max_len
    )
    val_dataset = SequenceLabelDataset(
        val_df.sequence.to_numpy(),
        val_df.label.to_numpy(),
        tokenizer,
        args.max_len
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    teacher = load_teacher(args.teacher_weights, device)

    if args.student_type == "transformer":
        student = StudentTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_pos_len=args.max_pos_len,
            num_buckets=NUM_LABELS,
            masked_pooling=args.masked_pooling,
        )
    elif args.student_type == "lite_mlp":
        student = StudentLiteMLP(
            vocab_size=vocab_size,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            max_pos_len=args.max_pos_len,
            num_buckets=NUM_LABELS,
            dropout=args.dropout,
            masked_pooling=args.masked_pooling,
        )
    else:
        raise ValueError(f"Unknown student_type: {args.student_type}")
    student.to(device)

    print("===== Student Configuration =====")
    print(f"vocab_size: {vocab_size}")
    print(f"d_model: {args.d_model}")
    print(f"nhead: {args.nhead}")
    print(f"num_layers: {args.num_layers}")
    print(f"dim_feedforward: {args.dim_feedforward}")
    print(f"dropout: {args.dropout}")
    print(f"max_pos_len: {args.max_pos_len}")
    print(f"masked_pooling: {args.masked_pooling}")
    print(f"trainable_params: {count_parameters(student):,}")

    optimizer = AdamW(student.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print("===== Training Configuration =====")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"temperature: {args.temperature}")
    print(f"alpha: {args.alpha}")
    print(f"total_steps: {total_steps}")
    print(f"warmup_steps: {warmup_steps}")
    print(f"log_file: {log_path}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print("Starting Distillation Training...")
    train_distillation(student, teacher, train_loader, val_loader, optimizer, scheduler, args, device)

    log_handle.close()


if __name__ == "__main__":
    main()