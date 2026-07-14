import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DEEPSEMP_GPU", "2"))
# --- CONFIG ---
# Use absolute paths as requested
ROOT_DIR = Path(os.environ.get("DEEPSEMP_ELEGANS_ROOT", "/data3/projects/2025_Assembly/eyh/c_elegans"))
INPUT_FASTA = Path(os.environ.get("DEEPSEMP_PROTEOME_FASTA", str(ROOT_DIR / "reference/c_elegans.protein.fa")))
OUTPUT_DIR = Path(os.environ.get("DEEPSEMP_EMBED_OUTDIR", str(ROOT_DIR / "embeddings")))
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D" # Balanced choice (650M params)
BATCH_SIZE = 16  # Adjust based on GPU VRAM (16 fits easily on 24GB)
MAX_LEN = 1022   # ESM limit (1024 - 2 special tokens)

def get_embeddings():
    print(f"Loading Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).cuda().eval()

    print(f"Reading Sequences from {INPUT_FASTA}")
    seqs = []
    ids = []
    
    # Read FASTA
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        # Header: >Y110A7A.10.1 pep ... -> ID: Y110A7A.10.1
        seq_id = record.id 
        seq_str = str(record.seq)
        
        # Truncate if too long (rare for single proteins, but happens)
        if len(seq_str) > MAX_LEN:
            seq_str = seq_str[:MAX_LEN]
            
        seqs.append(seq_str)
        ids.append(seq_id)

    print(f"Total Proteins: {len(seqs)}")
    
    # Storage for embeddings
    all_embeddings = []
    
    # Processing Loop
    for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Embedding"):
        batch_seqs = seqs[i : i + BATCH_SIZE]
        
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN + 2)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract Last Layer (hidden_states[-1])
            # Shape: [Batch, SeqLen, Dim]
            last_hidden = outputs.last_hidden_state
            
            # Mask padding tokens for Mean Pooling
            attention_mask = inputs['attention_mask']
            
            # Create masked mean pooling
            # We assume first token (CLS) and last token (EOS) should be excluded or included? 
            # Standard ESM practice: Mean over all AA tokens (excluding padding)
            
            for j in range(len(batch_seqs)):
                # Get length of this specific sequence (excluding padding)
                # sum(mask) gives total tokens including CLS/EOS
                seq_len = attention_mask[j].sum()
                
                # Slice: [1 : seq_len-1] removes CLS and EOS tokens
                # Result is mean of the actual Amino Acids
                token_embeddings = last_hidden[j, 1 : seq_len-1, :]
                mean_embedding = torch.mean(token_embeddings, dim=0).cpu().numpy()
                
                all_embeddings.append(mean_embedding)

    # Save to disk
    print("Saving to disk...")
    final_array = np.vstack(all_embeddings)
    
    np.save(OUTPUT_DIR / "esm2_embeddings.npy", final_array)
    
    # Save IDs separately to map back later
    with open(OUTPUT_DIR / "protein_ids.txt", "w") as f:
        for pid in ids:
            f.write(f"{pid}\n")

    print(f"Saved {final_array.shape} matrix.")

if __name__ == "__main__":
    get_embeddings()