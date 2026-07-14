import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==========================================
# 1. Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Deep-SemP Teacher Training v2")

    parser.add_argument("--data_path",       type=str, required=True)
    parser.add_argument("--model_name",      type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--output_dir",      type=str, default="./checkpoints")

    # resume
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to resume_checkpoint.pt (full state) or "
                             "model_epoch_N.pt (weights only)")
    parser.add_argument("--start_epoch",     type=int, default=0,
                        help="Epoch to resume from (0-based). Only used for "
                             "weights-only resume. Full resume ignores this.")

    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--num_labels",      type=int,   default=50)
    parser.add_argument("--max_len",         type=int,   default=100)
    parser.add_argument("--val_split",       type=float, default=0.1)
    parser.add_argument("--stratify_split",  action="store_true")
    parser.add_argument("--num_workers",     type=int,   default=4)

    return parser.parse_args()


# ==========================================
# 2. Dataset
# ==========================================
class SimulationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==========================================
# 3. Train / Eval
# ==========================================
def train_epoch(model, loader, optimizer, scheduler, device, epoch_idx, total_epochs):
    model.train()
    losses, correct, total = [], 0, 0

    for batch in tqdm(loader, desc=f"Epoch {epoch_idx+1}/{total_epochs} [Train]"):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        preds   = outputs.logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct / total, np.mean(losses)


def eval_model(model, loader, device, epoch_idx, total_epochs):
    model.eval()
    losses, correct, total = [], 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch_idx+1}/{total_epochs} [Val]"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds   = outputs.logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            losses.append(outputs.loss.item())

    return correct / total, np.mean(losses)


# ==========================================
# 4. Checkpoint save / load
# ==========================================
def save_resume_checkpoint(model, optimizer, scheduler,
                            epoch, best_acc, output_dir):
    """Save full training state after every epoch."""
    state = {
        "epoch":     epoch,
        "best_acc":  best_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = os.path.join(output_dir, "resume_checkpoint.pt")
    torch.save(state, path)
    print(f"  [ckpt] Resume state saved → {path}")


def load_resume_checkpoint(model, optimizer, scheduler,
                            checkpoint_path, args, device, steps_per_epoch):
    """
    Load checkpoint. Two modes:
      - resume_checkpoint.pt : full state restore
      - model_epoch_N.pt     : weights only, fast-forwards scheduler
    Returns (start_epoch, best_acc).
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model" in ckpt:
        # ── Full resume ──────────────────────────────────────────────
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"  Full resume from epoch {start_epoch} "
              f"| best acc so far: {best_acc:.4f}")
        print(f"  Scheduler LR restored: {scheduler.get_last_lr()}")

    else:
        # ── Weights-only resume (model_epoch_N.pt / best_model.pt) ──
        # Handle both bare state_dict and {'model_state_dict': ...} formats
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Weights-only resume | Missing: {len(missing)} | "
              f"Unexpected: {len(unexpected)}")

        start_epoch = args.start_epoch
        best_acc    = 0.0

        # Fast-forward scheduler to correct position
        steps_done = start_epoch * steps_per_epoch
        print(f"  Fast-forwarding scheduler {steps_done:,} steps "
              f"({start_epoch} epochs × {steps_per_epoch:,} batches)...")
        for _ in range(steps_done):
            scheduler.step()
        print(f"  LR after fast-forward: {scheduler.get_last_lr()[0]:.6e}")

    return start_epoch, best_acc


# ==========================================
# 5. Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    split_kwargs = dict(test_size=args.val_split, random_state=42)
    if args.stratify_split:
        split_kwargs["stratify"] = df["label"]
    train_df, val_df = train_test_split(df, **split_kwargs)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    print(f"Num labels: {args.num_labels}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Loading Model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.num_labels = args.num_labels
    config.use_cache  = False
    config.__dict__["pad_token_id"] = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    if not hasattr(config, "alibi_starting_size"):
        config.__dict__["alibi_starting_size"] = 512

    # Build from config to avoid meta tensor device conflict,
    # then load pretrained backbone weights from HF cache
    with torch.device("cpu"):
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=True
        )

    import glob as _glob
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M"
    )
    weight_files = (
        _glob.glob(f"{cache_dir}/**/pytorch_model.bin",   recursive=True) +
        _glob.glob(f"{cache_dir}/**/model.safetensors",   recursive=True)
    )
    if weight_files:
        weight_path = weight_files[0]
        print(f"Loading pretrained weights from cache: {weight_path}")
        if weight_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            pretrained_state = load_file(weight_path, device="cpu")
        else:
            pretrained_state = torch.load(weight_path, map_location="cpu")
        # Skip classifier head — size differs with new num_labels
        pretrained_state = {
            k: v for k, v in pretrained_state.items()
            if not k.startswith("classifier")
        }
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        print(f"Pretrained weights loaded | Missing: {len(missing)} | "
              f"Unexpected: {len(unexpected)}")
    else:
        print("[WARN] Pretrained cache not found — training from random init.")

    model = model.to(device)

    # ── Dataloaders ───────────────────────────────────────────────────
    pin = torch.cuda.is_available()
    train_dataset = SimulationDataset(
        train_df.sequence.to_numpy(), train_df.label.to_numpy(),
        tokenizer, args.max_len,
    )
    val_dataset = SimulationDataset(
        val_df.sequence.to_numpy(), val_df.label.to_numpy(),
        tokenizer, args.max_len,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    # No warmup for teacher — LR starts at full value immediately
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = args.start_epoch
    best_acc    = 0.0

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        start_epoch, best_acc = load_resume_checkpoint(
            model, optimizer, scheduler,
            args.load_checkpoint, args, device,
            steps_per_epoch=len(train_loader),
        )
    elif args.load_checkpoint:
        print(f"[WARN] Checkpoint not found at {args.load_checkpoint} "
              f"— starting from scratch.")

    print(f"\n===== Config =====")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  total_steps : {total_steps}")
    print(f"  start_epoch : {start_epoch}")
    print("==================\n")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"Training epochs {start_epoch + 1} → {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.epochs
        )
        val_acc, val_loss = eval_model(
            model, val_loader, device, epoch, args.epochs
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save per-epoch checkpoint
        epoch_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_path)
        print(f"Saved: {epoch_path}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"--> New Best Model! ({val_acc:.4f})")

        # Save full resume checkpoint (overwrites each epoch)
        save_resume_checkpoint(
            model, optimizer, scheduler, epoch, best_acc, args.output_dir
        )

        print("-" * 40)

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()