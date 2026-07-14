import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# --- FIX: Removed AdamW from here ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
# --- FIX: Added AdamW from PyTorch ---
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. CONFIGURATION & ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Deep-SemP Training Script (Save Every Epoch)")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Resume from this .pt file")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_labels", type=int, default=50, help="Total Buckets") # Default updated to 50
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.1)
    
    return parser.parse_args()

# --- 2. DATASET CLASS ---
class SimulationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 3. TRAIN & EVAL FUNCTIONS ---
def train_epoch(model, data_loader, optimizer, device, scheduler, epoch_idx):
    model.train()
    losses = []
    correct_predictions = 0
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch_idx+1} [Train]")
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        loop.set_postfix(loss=loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device, epoch_idx):
    model.eval()
    losses = []
    correct_predictions = 0
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch_idx+1} [Val]")
    
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# --- 4. MAIN LOOP ---
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=args.num_labels,
        trust_remote_code=True
    )

    if args.load_checkpoint:
        print(f"Resuming from {args.load_checkpoint}...")
        checkpoint = torch.load(args.load_checkpoint)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)

    # Dataloaders
    train_dataset = SimulationDataset(train_df.sequence.to_numpy(), train_df.label.to_numpy(), tokenizer, args.max_len)
    val_dataset = SimulationDataset(val_df.sequence.to_numpy(), val_df.label.to_numpy(), tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_acc = 0

    print("Starting Training...")
    for epoch in range(args.epochs):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, epoch)
        val_acc, val_loss = eval_model(model, val_loader, device, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # --- SAVE EVERY EPOCH ---
        epoch_save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Saved checkpoint: {epoch_save_path}")

        # --- SAVE BEST MODEL ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_save_path)
            print(f"--> New Best Model! ({val_acc:.4f})")
            
        print("-" * 30)

if __name__ == "__main__":
    main()