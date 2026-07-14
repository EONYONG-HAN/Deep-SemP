import os
import gzip
import argparse
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# --- CONFIG ---
DEFAULT_MODEL_PATH = "/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/track_a_direct_full/best_model.pt"
DEFAULT_R1 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_1.fastq.gz"
DEFAULT_R2 = "/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_2.fastq.gz"
OUTPUT_DIR = "/data3/projects/2025_Assembly/eyh/c_elegans/partitioned_reads"
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
NUM_LABELS = 50
BATCH_SIZE = 128 

def parse_args():
    parser = argparse.ArgumentParser(description="Paired-End Partitioning using Deep-SemP")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--r1", type=str, default=DEFAULT_R1)
    parser.add_argument("--r2", type=str, default=DEFAULT_R2)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()

def load_model(model_path):
    print("1. Initializing Config and Tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = NUM_LABELS
    config.use_cache = False
    config._attn_implementation = "eager"

    print("2. Building model architecture (from_config)...")
    model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)

    print(f"3. Loading weights from {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def process_batch(batch_r1, batch_r2, model, tokenizer, device, file_handles):
    # Perform Inference on R1 sequences only
    inputs = tokenizer(
        [r['seq'] for r in batch_r1], 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=100
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Write BOTH mates to the same bucket
    for r1, r2, bucket_id in zip(batch_r1, batch_r2, preds):
        h1, h2 = file_handles[bucket_id]
        # Write R1
        h1.write(f"@{r1['id']}\n{r1['seq']}\n+\n{r1['qual']}\n")
        # Write R2
        h2.write(f"@{r2['id']}\n{r2['seq']}\n+\n{r2['qual']}\n")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Open 50 pairs of output files (bucket_0_R1.fastq, bucket_0_R2.fastq, etc.)
    print(f"Opening {NUM_LABELS} output pairs...")
    file_handles = {}
    for i in range(NUM_LABELS):
        f1 = open(os.path.join(args.output_dir, f"bucket_{i}_R1.fastq"), "w")
        f2 = open(os.path.join(args.output_dir, f"bucket_{i}_R2.fastq"), "w")
        file_handles[i] = (f1, f2)

    model, tokenizer, device = load_model(args.model_path)
    print(f"Processing pairs: {Path(args.r1).name} & {Path(args.r2).name}...")
    
    batch_r1, batch_r2 = [], []
    total_pairs = 0
    
    # Stream both files in sync
    with gzip.open(args.r1, 'rt') as f1, gzip.open(args.r2, 'rt') as f2:
        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            if not line1 or not line2: break
            
            # Read 4-line FASTQ blocks for R1
            r1 = {'id': line1.strip().replace("@", ""), 'seq': f1.readline().strip()}
            f1.readline(); r1['qual'] = f1.readline().strip()
            
            # Read 4-line FASTQ blocks for R2
            r2 = {'id': line2.strip().replace("@", ""), 'seq': f2.readline().strip()}
            f2.readline(); r2['qual'] = f2.readline().strip()

            batch_r1.append(r1)
            batch_r2.append(r2)
            
            if len(batch_r1) >= BATCH_SIZE:
                process_batch(batch_r1, batch_r2, model, tokenizer, device, file_handles)
                total_pairs += len(batch_r1)
                batch_r1, batch_r2 = [], []
                print(f"Processed {total_pairs} pairs...", end='\r')

        if batch_r1:
            process_batch(batch_r1, batch_r2, model, tokenizer, device, file_handles)
            total_pairs += len(batch_r1)

    for h1, h2 in file_handles.values():
        h1.close(); h2.close()
    print(f"\nDone! Partitioned {total_pairs} pairs into {args.output_dir}")

if __name__ == "__main__":
    main()