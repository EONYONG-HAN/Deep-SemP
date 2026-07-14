import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ==========================================
# 1. Configuration & Arguments
# ==========================================
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
NUM_LABELS = 50
VOCAB_SIZE = 4096  # DNABERT-2 BPE Vocab Size

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Deep-SemP")
    parser.add_argument("--teacher_weights", type=str, required=True, help="Path to best Deep-SemP model (.pt)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to full training CSV")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--output_dir", type=str, default="./distilled_models")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()

# ==========================================
# 2. Dataset & Tokenization
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
            max_length=self.max_length
        )
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. Model Architectures
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

    # Required for DNABERT-2 custom code
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    config.__dict__["pad_token_id"] = pad_id

    # Some custom remote-code models expect this too
    if not hasattr(config, "alibi_starting_size"):
        config.__dict__["alibi_starting_size"] = 512

    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    print("config.pad_token_id:", config.__dict__.get("pad_token_id"))
    print("torch default device:", torch.get_default_device())

    # IMPORTANT: build from config, not from_pretrained
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

class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, num_buckets=50):
        super(StudentTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_buckets)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.fc(x)
        return logits

# ==========================================
# 4. Training & Evaluation Logic
# ==========================================
def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, true_labels)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher)
    return (alpha * hard_loss) + ((1 - alpha) * kl_loss * (T * T))

def evaluate_student(student, teacher, val_loader, device):
    student.eval()
    correct_student, correct_teacher, agreements, total = 0, 0, 0, 0
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits
            student_logits = student(input_ids)
            
            _, student_preds = torch.max(student_logits, 1)
            _, teacher_preds = torch.max(teacher_logits, 1)
            
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
    return agreement_rate

def train_distillation(student, teacher, train_loader, val_loader, optimizer, scheduler, args, device):
    best_agreement = 0.0
    
    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits
                
            student_logits = student(input_ids)
            
            loss = distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)
            loss.backward()
            
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] | Batch {batch_idx} | Total Loss: {loss.item():.4f}")
                
        print(f"=== End of Epoch {epoch+1} | Avg Loss: {running_loss / len(train_loader):.4f} ===")
        
        agreement_rate = evaluate_student(student, teacher, val_loader, device)
        
        if agreement_rate > best_agreement:
            best_agreement = agreement_rate
            save_path = os.path.join(args.output_dir, "best_student_model.pt")
            torch.save(student.state_dict(), save_path)
            print(f">>> New best Student model saved to {save_path}! <<<\n")

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Integrated Data Loading & Splitting ---
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    train_dataset = SequenceLabelDataset(train_df.sequence.to_numpy(), train_df.label.to_numpy(), tokenizer, args.max_len)
    val_dataset = SequenceLabelDataset(val_df.sequence.to_numpy(), val_df.label.to_numpy(), tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Initialize Models ---
    teacher = load_teacher(args.teacher_weights, device)
    student = StudentTransformer(vocab_size=tokenizer.vocab_size, num_buckets=NUM_LABELS)
    student.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(student.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("Starting Distillation Training...")
    train_distillation(student, teacher, train_loader, val_loader, optimizer, scheduler, args, device)

if __name__ == "__main__":
    main()