import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import random
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation v3 - Illumina5 Dataset + Hidden State Distillation"
    )

    # data / paths
    parser.add_argument("--teacher_weights", type=str, required=True)
    parser.add_argument("--data_path",        type=str, required=True)
    parser.add_argument("--simval_path",      type=str, default=None)
    parser.add_argument("--output_dir",       type=str, default="./distilled_models")
    parser.add_argument("--log_file",         type=str, default=None)

    # split / loader
    parser.add_argument("--val_split",        type=float, default=0.1)
    parser.add_argument("--stratify_split",   action="store_true")
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--max_len",          type=int,   default=100)
    parser.add_argument("--weighted_sampler", action="store_true")

    # training
    parser.add_argument("--epochs",           type=int,   default=20)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--warmup_ratio",     type=float, default=0.05)
    parser.add_argument("--temperature",      type=float, default=2.0)
    parser.add_argument("--alpha",            type=float, default=0.3)

    # hidden state distillation
    parser.add_argument("--beta",             type=float, default=0.1)
    parser.add_argument("--no_hidden_distill",action="store_true")

    # resume
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to resume_checkpoint.pt or best_student_model.pt")
    parser.add_argument("--start_epoch",     type=int, default=0,
                        help="Epoch to resume from (0-based). Only used for "
                             "weights-only resume (best_student_model.pt). "
                             "Full resume_checkpoint.pt ignores this.")

    # student architecture
    parser.add_argument("--d_model",          type=int,   default=384)
    parser.add_argument("--nhead",            type=int,   default=8)
    parser.add_argument("--num_layers",       type=int,   default=8)
    parser.add_argument("--dim_feedforward",  type=int,   default=1024)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--max_pos_len",      type=int,   default=512)
    parser.add_argument("--masked_pooling",   action="store_true")
    parser.add_argument("--student_type",     type=str,   default="transformer",
                        choices=["transformer", "lite_mlp"])
    parser.add_argument("--hidden_dim",       type=int,   default=256)
    return parser.parse_args()


# ==========================================
# 2. Logging
# ==========================================
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

def setup_logging(log_path):
    log_f = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    return log_f


# ==========================================
# 3. Dataset
# ==========================================
class SequenceLabelDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences  = sequences
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = str(self.sequences[idx])
        label = int(self.labels[idx])
        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return (
            inputs["input_ids"].squeeze(0),
            inputs["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )


# ==========================================
# 4. Teacher Loading
# ==========================================
def load_teacher(model_path, device):
    print(f"Loading Teacher from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config    = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.num_labels = NUM_LABELS
    config.use_cache  = False
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    config.__dict__["pad_token_id"] = pad_id
    if not hasattr(config, "alibi_starting_size"):
        config.__dict__["alibi_starting_size"] = 512

    with torch.device("cpu"):
        teacher = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=True
        )

    ckpt       = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_hidden_size = config.hidden_size
    print(f"Teacher hidden size: {teacher_hidden_size}")
    return teacher, teacher_hidden_size


# ==========================================
# 5. Student Models
# ==========================================
class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=8,
                 dim_feedforward=1024, dropout=0.1, max_pos_len=512,
                 num_buckets=50, masked_pooling=True):
        super().__init__()
        self.masked_pooling = masked_pooling
        self.d_model        = d_model
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

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if self.masked_pooling and attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = x.mean(dim=1)

        return self.fc(pooled), pooled


class StudentLiteMLP(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden_dim=256, max_pos_len=512,
                 num_buckets=50, dropout=0.1, masked_pooling=True):
        super().__init__()
        self.masked_pooling = masked_pooling
        self.d_model        = d_model
        self.embedding      = nn.Embedding(vocab_size, d_model)
        self.pos_embedding  = nn.Embedding(max_pos_len, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_buckets),
        )

    def forward(self, input_ids, attention_mask=None):
        B, L    = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        if self.masked_pooling and attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = x.mean(dim=1)

        return self.mlp(pooled), pooled


# ==========================================
# 6. Projection Layer
# ==========================================
class HiddenProjector(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.GELU(),
            nn.Linear(student_dim, student_dim),
        )

    def forward(self, x):
        return self.proj(x)


# ==========================================
# 7. Loss
# ==========================================
def distillation_loss(student_logits, teacher_logits,
                      student_hidden, teacher_hidden_projected,
                      true_labels, temperature, alpha, beta,
                      use_hidden_distill=True):
    hard_loss    = F.cross_entropy(student_logits, true_labels)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits     / temperature, dim=1)
    kl_loss      = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)

    rep_loss = torch.tensor(0.0, device=student_logits.device)
    if use_hidden_distill and teacher_hidden_projected is not None:
        rep_loss = F.mse_loss(
            F.normalize(student_hidden,           dim=-1),
            F.normalize(teacher_hidden_projected, dim=-1),
        )

    total = (alpha * hard_loss) \
          + ((1.0 - alpha) * kl_loss * (temperature ** 2)) \
          + (beta * rep_loss)

    return total, hard_loss.detach(), kl_loss.detach(), rep_loss.detach()


# ==========================================
# 8. Evaluation
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(student, projector, teacher, loader, device, args,
             label="Validation", compute_loss=True):
    student.eval()
    if projector is not None:
        projector.eval()

    correct_student = correct_teacher = agreements = total = 0
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids      = input_ids.to(device,     non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels         = labels.to(device,         non_blocking=True)

            t_out          = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)
            teacher_logits = t_out.logits
            t_hidden_last  = t_out.hidden_states[-1]
            mask           = attention_mask.unsqueeze(-1).float()
            t_pooled       = (t_hidden_last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

            student_logits, s_pooled = student(input_ids, attention_mask)
            t_proj = projector(t_pooled) if projector is not None else None

            if compute_loss:
                loss, *_ = distillation_loss(
                    student_logits, teacher_logits, s_pooled, t_proj,
                    labels, args.temperature, args.alpha, args.beta,
                    use_hidden_distill=(not args.no_hidden_distill),
                )
                total_loss += loss.item()

            s_preds = student_logits.argmax(dim=1)
            t_preds = teacher_logits.argmax(dim=1)

            total           += labels.size(0)
            correct_student += (s_preds == labels).sum().item()
            correct_teacher += (t_preds == labels).sum().item()
            agreements      += (s_preds == t_preds).sum().item()

    s_acc  = 100.0 * correct_student / total
    t_acc  = 100.0 * correct_teacher / total
    agree  = 100.0 * agreements      / total
    avg_vl = total_loss / len(loader) if compute_loss else None

    print(f"\n--- {label} ---")
    print(f"Teacher Acc : {t_acc:.2f}%")
    print(f"Student Acc : {s_acc:.2f}%")
    print(f"Agreement   : {agree:.2f}%")
    if avg_vl is not None:
        print(f"Val Loss    : {avg_vl:.4f}")
    print()

    student.train()
    if projector is not None:
        projector.train()

    return {"teacher_acc": t_acc, "student_acc": s_acc,
            "agreement_rate": agree, "val_loss": avg_vl}


# ==========================================
# 9. Checkpoint save / load
# ==========================================
def save_checkpoint(student, projector, optimizer, scheduler,
                    epoch, best_agreement, best_metrics, output_dir):
    """Save full training state — overwrites each epoch."""
    state = {
        "epoch":          epoch,
        "best_agreement": best_agreement,
        "best_metrics":   best_metrics,
        "student":        student.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
    }
    if projector is not None:
        state["projector"] = projector.state_dict()
    path = os.path.join(output_dir, "resume_checkpoint.pt")
    torch.save(state, path)
    print(f"  [ckpt] Resume state saved → {path}")


def load_checkpoint(student, projector, optimizer, scheduler,
                    checkpoint_path, args, device):
    """
    Load checkpoint. Two modes:
      - resume_checkpoint.pt : full state (weights + optimizer + scheduler)
      - best_student_model.pt: weights only, uses --start_epoch and --lr
    Returns (start_epoch, best_agreement, best_metrics).
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "student" in ckpt:
        # ── Full resume ──────────────────────────────────────────────
        student.load_state_dict(ckpt["student"])
        if projector is not None and "projector" in ckpt:
            projector.load_state_dict(ckpt["projector"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"  Scheduler restored — last LR: {scheduler.get_last_lr()}")
        start_epoch    = ckpt.get("epoch", 0) + 1
        best_agreement = ckpt.get("best_agreement", -1.0)
        best_metrics   = ckpt.get("best_metrics", None)
        print(f"  Full resume from epoch {start_epoch} "
              f"| best agreement: {best_agreement:.2f}%")

    else:
        # ── Weights-only resume (best_student_model.pt) ──────────────
        student.load_state_dict(ckpt)
        if projector is not None:
            proj_path = checkpoint_path.replace(
                "best_student_model.pt", "best_projector.pt"
            )
            if os.path.exists(proj_path):
                projector.load_state_dict(
                    torch.load(proj_path, map_location=device)
                )
                print(f"  Projector weights loaded from {proj_path}")

        # Skip warmup — set optimizer LR directly to --lr value
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr

        # Fast-forward scheduler to match start_epoch position
        # so LR decays correctly over remaining epochs
        steps_per_epoch  = len(optimizer.param_groups)   # placeholder; corrected below
        # We advance the scheduler by (start_epoch * steps_per_epoch) steps
        # but we don't have the loader here — mark for correction in main()
        start_epoch    = args.start_epoch
        best_agreement = -1.0
        best_metrics   = None
        print(f"  Weights-only resume | start_epoch={start_epoch} | lr={args.lr:.2e}")
        print(f"  Warmup skipped — LR set directly to {args.lr:.2e}")

    return start_epoch, best_agreement, best_metrics


# ==========================================
# 10. Training loop
# ==========================================
def train_distillation(student, projector, teacher,
                       train_loader, val_loader,
                       optimizer, scheduler, args, device,
                       start_epoch=0, best_agreement=-1.0,
                       best_metrics=None):

    total_epochs = args.epochs
    print(f"Training epochs {start_epoch + 1} → {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        student.train()
        if projector is not None:
            projector.train()

        r_total = r_hard = r_kl = r_rep = 0.0

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids      = input_ids.to(device,      non_blocking=True)
            attention_mask = attention_mask.to(device,  non_blocking=True)
            labels         = labels.to(device,          non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                t_out          = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                         output_hidden_states=True)
                teacher_logits = t_out.logits
                t_hidden_last  = t_out.hidden_states[-1]
                mask           = attention_mask.unsqueeze(-1).float()
                t_pooled       = (t_hidden_last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

            student_logits, s_pooled = student(input_ids, attention_mask)
            t_proj = projector(t_pooled) if projector is not None else None

            loss, hard_l, kl_l, rep_l = distillation_loss(
                student_logits, teacher_logits, s_pooled, t_proj,
                labels, args.temperature, args.alpha, args.beta,
                use_hidden_distill=(not args.no_hidden_distill),
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            r_total += loss.item()
            r_hard  += hard_l.item()
            r_kl    += kl_l.item()
            r_rep   += rep_l.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{total_epochs}] | Batch {batch_idx} | "
                    f"Total: {loss.item():.4f} | Hard: {hard_l.item():.4f} | "
                    f"KL: {kl_l.item():.4f} | Rep: {rep_l.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6e}"
                )

        n = len(train_loader)
        print(
            f"=== Epoch {epoch+1} | "
            f"Avg Total: {r_total/n:.4f} | Hard: {r_hard/n:.4f} | "
            f"KL: {r_kl/n:.4f} | Rep: {r_rep/n:.4f} ==="
        )

        metrics = evaluate(student, projector, teacher, val_loader, device, args,
                           label="Validation (clean sim split)")

        if metrics["agreement_rate"] > best_agreement:
            best_agreement = metrics["agreement_rate"]
            best_metrics   = {
                "epoch": epoch + 1, **metrics,
                "avg_total_loss": r_total / n,
                "avg_hard_loss":  r_hard  / n,
                "avg_kl_loss":    r_kl    / n,
                "avg_rep_loss":   r_rep   / n,
            }

            torch.save(student.state_dict(),
                       os.path.join(args.output_dir, "best_student_model.pt"))
            if projector is not None:
                torch.save(projector.state_dict(),
                           os.path.join(args.output_dir, "best_projector.pt"))

            with open(os.path.join(args.output_dir, "best_metrics.txt"), "w") as f:
                for k, v in best_metrics.items():
                    f.write(f"{k}\t{v}\n")
            print(f">>> New best saved (agreement={best_agreement:.2f}%) <<<\n")

        # Save full resume checkpoint every epoch
        save_checkpoint(student, projector, optimizer, scheduler,
                        epoch, best_agreement, best_metrics, args.output_dir)

    if best_metrics:
        print("===== Best Validation Summary =====")
        for k, v in best_metrics.items():
            print(f"  {k}: {v}")


# ==========================================
# 11. Sim-Val Evaluation
# ==========================================
def evaluate_simval(student, projector, teacher, simval_path,
                    tokenizer, args, device):
    print(f"\n{'='*60}")
    print(f"SIM-VAL EVALUATION (held-out illumina5 samples 05-06)")
    print(f"Loading: {simval_path}")
    print(f"{'='*60}")

    df = pd.read_csv(simval_path)
    print(f"Sim-val samples: {len(df):,}")

    dataset = SequenceLabelDataset(
        df.sequence.to_numpy(), df.label.to_numpy(),
        tokenizer, args.max_len,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    best_path = os.path.join(args.output_dir, "best_student_model.pt")
    if os.path.exists(best_path):
        print(f"Loading best student checkpoint: {best_path}")
        student.load_state_dict(torch.load(best_path, map_location=device))
        if projector is not None:
            proj_path = os.path.join(args.output_dir, "best_projector.pt")
            if os.path.exists(proj_path):
                projector.load_state_dict(torch.load(proj_path, map_location=device))

    metrics = evaluate(student, projector, teacher, loader, device, args,
                       label="Sim-Val (illumina5 held-out)",
                       compute_loss=False)

    simval_out = os.path.join(args.output_dir, "simval_metrics.txt")
    with open(simval_out, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}\t{v}\n")
    print(f"Sim-val results saved to: {simval_out}")
    return metrics


# ==========================================
# 12. Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_path   = args.log_file or os.path.join(args.output_dir, "distillation.log")
    log_handle = setup_logging(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────
    print(f"Loading training data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Total rows: {len(df):,}")

    split_kwargs = dict(test_size=args.val_split, random_state=42)
    if args.stratify_split:
        split_kwargs["stratify"] = df["label"]
    train_df, val_df = train_test_split(df, **split_kwargs)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    print(f"Hidden distillation: {'OFF' if args.no_hidden_distill else f'ON (beta={args.beta})'}")
    print("Noise augmentation : OFF (illumina5 data has built-in realistic noise)")

    train_dataset = SequenceLabelDataset(
        train_df.sequence.to_numpy(), train_df.label.to_numpy(),
        tokenizer, args.max_len,
    )
    val_dataset = SequenceLabelDataset(
        val_df.sequence.to_numpy(), val_df.label.to_numpy(),
        tokenizer, args.max_len,
    )

    if args.weighted_sampler:
        print("Using WeightedRandomSampler...")
        label_counts = train_df["label"].value_counts().to_dict()
        weights      = [1.0 / label_counts[l] for l in train_df["label"].tolist()]
        sampler      = WeightedRandomSampler(weights, num_samples=len(weights),
                                             replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Teacher ───────────────────────────────────────────────────────
    teacher, teacher_hidden_size = load_teacher(args.teacher_weights, device)

    # ── Student ───────────────────────────────────────────────────────
    if args.student_type == "transformer":
        student = StudentTransformer(
            vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
            dropout=args.dropout, max_pos_len=args.max_pos_len,
            num_buckets=NUM_LABELS, masked_pooling=args.masked_pooling,
        )
    else:
        student = StudentLiteMLP(
            vocab_size=vocab_size, d_model=args.d_model, hidden_dim=args.hidden_dim,
            max_pos_len=args.max_pos_len, num_buckets=NUM_LABELS,
            dropout=args.dropout, masked_pooling=args.masked_pooling,
        )
    student.to(device)

    # ── Projector ─────────────────────────────────────────────────────
    projector = None
    if not args.no_hidden_distill:
        projector = HiddenProjector(teacher_hidden_size, args.d_model).to(device)
        print(f"Projector: {teacher_hidden_size} -> {args.d_model} "
              f"({count_parameters(projector):,} params)")

    print(f"Student trainable params: {count_parameters(student):,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    params       = list(student.parameters())
    if projector is not None:
        params  += list(projector.parameters())
    optimizer    = AdamW(params, lr=args.lr)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch    = args.start_epoch
    best_agreement = -1.0
    best_metrics   = None

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        start_epoch, best_agreement, best_metrics = load_checkpoint(
            student, projector, optimizer, scheduler,
            args.load_checkpoint, args, device,
        )

        # For weights-only resume: fast-forward scheduler so LR decays
        # correctly over the remaining epochs rather than starting fresh.
        # We simulate the steps that would have been taken up to start_epoch.
        ckpt_probe = torch.load(args.load_checkpoint, map_location="cpu")
        if "student" not in ckpt_probe:
            # Weights-only — advance scheduler by steps already completed
            steps_done = start_epoch * len(train_loader)
            print(f"  Fast-forwarding scheduler by {steps_done:,} steps "
                  f"({start_epoch} epochs × {len(train_loader):,} batches)...")
            for _ in range(steps_done):
                scheduler.step()
            print(f"  Scheduler LR after fast-forward: "
                  f"{scheduler.get_last_lr()[0]:.6e}")
        del ckpt_probe

    elif args.load_checkpoint:
        print(f"[WARN] Checkpoint not found at {args.load_checkpoint} "
              f"— starting from scratch.")

    print("\n===== Config =====")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  total_steps  : {total_steps}")
    print(f"  warmup_steps : {warmup_steps}")
    print(f"  start_epoch  : {start_epoch}")
    print("==================\n")

    # ── Train ─────────────────────────────────────────────────────────
    train_distillation(student, projector, teacher,
                       train_loader, val_loader,
                       optimizer, scheduler, args, device,
                       start_epoch=start_epoch,
                       best_agreement=best_agreement,
                       best_metrics=best_metrics)

    # ── Sim-Val ───────────────────────────────────────────────────────
    if args.simval_path and os.path.exists(args.simval_path):
        evaluate_simval(student, projector, teacher,
                        args.simval_path, tokenizer, args, device)
    else:
        print("\n[INFO] No --simval_path provided, skipping sim-val evaluation.")

    log_handle.close()


if __name__ == "__main__":
    main()