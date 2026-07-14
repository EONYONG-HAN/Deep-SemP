import pandas as pd
import torch
import gzip
import subprocess
import random
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda"
ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")

# FASTQ_FILE = ROOT / "real_data/SRR36278209_1.fastq.gz" #rep1 data
FASTQ_FILE = "/data3/projects/2025_Assembly/eyh/c_elegans_rep2/raw_data/SRR36114682_1.fastq.gz" ##rep2 data
REF_CDNA = ROOT / "reference/c_elegans_cdna.fa.gz" 
BUCKET_MAP = ROOT / "buckets/semantic_map.tsv" 
CHECKPOINT_PATH = ROOT / "models/checkpoints/track_a_direct_full/best_model.pt" 
MODEL_BASE_NAME = "zhihan1996/DNABERT-2-117M"

SAMPLE_SIZE = 10000
TOP_K = 5
NUM_LABELS = 50 

def load_bucket_map():
    print(f"1. Loading Map from {BUCKET_MAP}...")
    tx_to_buckets = {}
    with open(BUCKET_MAP, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            b_id = int(parts[0]) 
            for tx in parts[3].split(','):
                clean_tx = tx.strip()
                if clean_tx not in tx_to_buckets: tx_to_buckets[clean_tx] = set()
                tx_to_buckets[clean_tx].add(b_id)
    print(f"   Mapped {len(tx_to_buckets)} transcripts.")
    return tx_to_buckets

def get_real_reads():
    print(f"2. Sampling {SAMPLE_SIZE} real reads...")
    reads = []
    try:
        with gzip.open(FASTQ_FILE, 'rt') as f:
            while len(reads) < SAMPLE_SIZE:
                try:
                    head = f.readline(); seq = f.readline(); f.readline(); f.readline()
                    if not head: break
                    if len(seq.strip()) >= 50:
                        if random.random() < 0.1: 
                            reads.append(seq.strip())
                except ValueError: break
    except Exception as e:
        print(f"Error reading FASTQ: {e}")
    print(f"   Sampled {len(reads)} reads.")
    return reads

def get_ground_truth(reads):
    print("3. Running BLAST to find True Identity...")
    
    # --- FIX: Handle .gz for makeblastdb ---
    # We check if the DB files exist. If not, we unzip to temp and build.
    if not os.path.exists("ref_cdna_db.nhr"):
        print("   Building BLAST DB (unzipping first)...")
        # 1. Unzip to temp file
        subprocess.run(f"gunzip -c {REF_CDNA} > temp_ref.fa", shell=True)
        # 2. Make DB
        subprocess.run(f"makeblastdb -in temp_ref.fa -dbtype nucl -out ref_cdna_db", shell=True)
        # 3. Clean up
        if os.path.exists("temp_ref.fa"):
            os.remove("temp_ref.fa")

    with open("temp_query.fa", "w") as f:
        for i, seq in enumerate(reads):
            f.write(f">read_{i}\n{seq}\n")

    cmd = "blastn -query temp_query.fa -db ref_cdna_db -outfmt '6 qseqid sseqid' -max_target_seqs 1 -evalue 1e-5"
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
    except subprocess.CalledProcessError:
        print("   BLAST failed.")
        return {}

    truth = {}
    for line in output.splitlines():
        qid, sid = line.split()[:2]
        idx = int(qid.split('_')[1])
        truth[idx] = sid 
    
    print(f"   Identified true genes for {len(truth)} reads.")
    return truth

# def evaluate_model(reads, truth, tx_map):
#     print("4. Testing Model Accuracy (DEBUG MODE)...")
    
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_BASE_NAME, num_labels=NUM_LABELS, trust_remote_code=True
#     )
    
#     try:
#         state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#         if 'model_state_dict' in state_dict:
#              model.load_state_dict(state_dict['model_state_dict'])
#         else:
#              model.load_state_dict(state_dict)
#     except Exception as e:
#         print(f"   Error loading weights: {e}")
#         return

#     model.to(DEVICE)
#     model.eval()
    
#     correct_top1 = 0
#     total = 0
    
#     print("   Running Inference...")
    
#     # We will just check the first 20 matches to see what is happening
#     debug_matches_printed = 0 
    
#     for i, seq in enumerate(reads):
#         if i not in truth: continue
#         true_tx = truth[i]
        
#         valid_buckets = None
#         if true_tx in tx_map:
#             valid_buckets = tx_map[true_tx]
#         else:
#             base = true_tx.rsplit('.', 1)[0]
#             if base in tx_map:
#                 valid_buckets = tx_map[base]
        
#         if not valid_buckets: continue 
        
#         total += 1
#         inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=100, padding="max_length").to(DEVICE)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=-1)
#             vals, indices = torch.topk(probs, TOP_K)
#             preds = indices[0].cpu().numpy()
            
#         pred_bucket = preds[0]
        
#         # --- DEBUG PRINT LOGIC ---
#         if pred_bucket in valid_buckets:
#             correct_top1 += 1
#             if debug_matches_printed < 20:
#                 print(f"[MATCH] Tx: {true_tx:<15} | Map says Bucket: {valid_buckets} | Model guessed: {pred_bucket}")
#                 debug_matches_printed += 1
#         else:
#             # If we finally find a mismatch, print it!
#             print(f"[MISMATCH] Tx: {true_tx:<15} | Map says Bucket: {valid_buckets} | Model guessed: {pred_bucket}")
            
#     print("-" * 30)
#     if total > 0:
#         print(f"REAL DATA ACCURACY (N={total})")
#         print(f"Top-1 Accuracy: {correct_top1/total:.2%}")
#     print("-" * 30)

def evaluate_model(reads, truth, tx_map):
    print("4. Testing Model Accuracy...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer + Config (explicitly set pad_token_id)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)

    config = AutoConfig.from_pretrained(MODEL_BASE_NAME, trust_remote_code=True)
    config.num_labels = NUM_LABELS
    config.pad_token_id = tokenizer.pad_token_id  # <-- FIX for your AttributeError
    config.use_cache = False
    config._attn_implementation = "eager"

    # Build model WITHOUT from_pretrained() to avoid meta-tensor init
    model = AutoModelForSequenceClassification.from_config(
        config,
        trust_remote_code=True
    )

    # Load your trained weights
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # If checkpoint saved under DDP/DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys (up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (up to 20): {unexpected[:20]}")

    model.to(device)
    model.eval()

    correct_top1 = 0
    correct_topk = 0
    total = 0

    print("   Running Inference...")
    for i, seq in enumerate(reads):
        if i not in truth:
            continue
        true_tx = truth[i]

        # Mapping Logic
        valid_buckets = None
        if true_tx in tx_map:
            valid_buckets = tx_map[true_tx]
        else:
            base = true_tx.rsplit('.', 1)[0]
            if base in tx_map:
                valid_buckets = tx_map[base]

        if not valid_buckets:
            continue

        total += 1

        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=100,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}  # safer than .to(device) on BatchEncoding

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            _, indices = torch.topk(probs, TOP_K)
            preds = indices[0].cpu().numpy()

        if preds[0] in valid_buckets:
            correct_top1 += 1
        if any(p in valid_buckets for p in preds):
            correct_topk += 1

    print("-" * 30)
    if total > 0:
        print(f"REAL DATA ACCURACY (N={total})")
        print(f"Top-1 Accuracy: {correct_top1/total:.2%}")
        print(f"Top-{TOP_K} Accuracy: {correct_topk/total:.2%}")
    else:
        print("   No reads matched between BLAST and Bucket Map.")
    print("-" * 30)


if __name__ == "__main__":
    tx_map = load_bucket_map()
    reads = get_real_reads()
    truth = get_ground_truth(reads)
    evaluate_model(reads, truth, tx_map)