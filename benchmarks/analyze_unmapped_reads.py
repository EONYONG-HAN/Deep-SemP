import pandas as pd
import gzip
import subprocess
import random
import os
import sys
from pathlib import Path
from collections import Counter

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")

# INPUTS
FASTQ_FILE = ROOT / "real_data/SRR36278209_1.fastq.gz"
BUCKET_MAP = ROOT / "buckets/semantic_map.tsv"

# REFERENCES
REF_CDNA = ROOT / "reference/c_elegans_cdna.fa.gz"   # To check known transcripts
REF_GENOME = ROOT / "reference/c_elegans_dna_fa.gz" # To check Introns/UTRs

SAMPLE_SIZE = 1000  # Sample enough to get a good look at the "rejects"

def load_bucket_map():
    print(f"1. Loading Map from {BUCKET_MAP}...")
    valid_tx_ids = set()
    with open(BUCKET_MAP, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            for tx in parts[3].split(','):
                valid_tx_ids.add(tx.strip())
    print(f"   Loaded {len(valid_tx_ids)} valid transcript IDs.")
    return valid_tx_ids

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
    return reads

def build_blast_dbs():
    # Build cDNA DB if missing
    if not os.path.exists("ref_cdna_db.nhr"):
        print("   Building cDNA BLAST DB...")
        subprocess.run(f"gunzip -c {REF_CDNA} > temp_cdna.fa", shell=True)
        subprocess.run(f"makeblastdb -in temp_cdna.fa -dbtype nucl -out ref_cdna_db", shell=True)
        if os.path.exists("temp_cdna.fa"): os.remove("temp_cdna.fa")

    # Build GENOME DB if missing
    if not os.path.exists("ref_genome_db.nhr"):
        print("   Building GENOME BLAST DB...")
        subprocess.run(f"gunzip -c {REF_GENOME} > temp_genome.fa", shell=True)
        subprocess.run(f"makeblastdb -in temp_genome.fa -dbtype nucl -out ref_genome_db", shell=True)
        if os.path.exists("temp_genome.fa"): os.remove("temp_genome.fa")

def forensic_analysis(reads, valid_tx_ids):
    print("3. Identifying 'Dark Matter' Reads...")
    build_blast_dbs()

    # --- STEP A: BLAST against cDNA (Transcriptome) ---
    print("   Step A: Checking against Transcriptome (cDNA)...")
    with open("temp_query.fa", "w") as f:
        for i, seq in enumerate(reads):
            f.write(f">read_{i}\n{seq}\n")

    # We ask for qseqid and sseqid
    cmd_cdna = "blastn -query temp_query.fa -db ref_cdna_db -outfmt '6 qseqid sseqid' -max_target_seqs 1 -evalue 1e-5"
    try:
        out_cdna = subprocess.check_output(cmd_cdna, shell=True).decode()
    except:
        out_cdna = ""

    # Parse cDNA results
    cdna_hits = {}
    for line in out_cdna.splitlines():
        qid, sid = line.split()[:2]
        idx = int(qid.split('_')[1])
        cdna_hits[idx] = sid

    # --- STEP B: Separate the 'Unmapped' ---
    unmapped_indices = []
    mapped_count = 0
    tx_not_in_bucket_count = 0
    
    for i in range(len(reads)):
        if i in cdna_hits:
            hit_id = cdna_hits[i]
            # Check if this hit is in our bucket map
            if hit_id in valid_tx_ids or hit_id.rsplit('.', 1)[0] in valid_tx_ids:
                mapped_count += 1
            else:
                # It hit a transcript, but one we ignored (e.g., lncRNA, pseudogene)
                tx_not_in_bucket_count += 1
                unmapped_indices.append(i) 
        else:
            # Didn't hit cDNA at all
            unmapped_indices.append(i)

    print(f"   - Total Reads: {len(reads)}")
    print(f"   - Successfully Mapped to Buckets: {mapped_count}")
    print(f"   - Transcript Hit (But not in Buckets): {tx_not_in_bucket_count}")
    print(f"   - Total 'Fuzzy' Reads to Investigate: {len(unmapped_indices)}")

    if not unmapped_indices:
        print("   No fuzzy reads found! (Sample size might be too small).")
        return

    # --- STEP C: BLAST 'Fuzzy' Reads against GENOME ---
    print("\n   Step C: BLASTing Fuzzy reads against Full Genome...")
    
    with open("temp_fuzzy.fa", "w") as f:
        for idx in unmapped_indices:
            f.write(f">read_{idx}\n{reads[idx]}\n")
            
    # BLAST against GENOME
    # We ask for 'stitle' to see "Chromosome I" etc.
    cmd_genome = "blastn -query temp_fuzzy.fa -db ref_genome_db -outfmt '6 qseqid sseqid stitle' -max_target_seqs 1 -evalue 1e-5"
    try:
        out_genome = subprocess.check_output(cmd_genome, shell=True).decode()
    except:
        out_genome = ""

    genome_hits = {}
    for line in out_genome.splitlines():
        parts = line.split('\t')
        qid = parts[0]
        idx = int(qid.split('_')[1])
        # sseqid usually has Chromosome info for genome files
        genome_hits[idx] = parts[2] # Title often contains "Chromosome I", etc.

    # --- STEP D: Report Findings ---
    print("\n" + "="*40)
    print("FORENSIC REPORT: UNMAPPED READS")
    print("="*40)

    genomic_count = 0
    no_hit_count = 0
    
    print(f"{'READ_ID':<10} | {'STATUS':<20} | {'DETAILS'}")
    print("-" * 80)

    for idx in unmapped_indices:
        seq_preview = reads[idx][:20] + "..."
        
        if idx in cdna_hits:
            # Case 1: Hit cDNA, but not in bucket map
            status = "KNOWN_TRANSCRIPT"
            details = f"Hit {cdna_hits[idx]} (Not in Map)"
        elif idx in genome_hits:
            # Case 2: Hit Genome, but NOT cDNA
            status = "GENOMIC (Intron/UTR)"
            details = f"Loc: {genome_hits[idx][:40]}" # Truncate long titles
            genomic_count += 1
        else:
            # Case 3: No Hit at all
            status = "NO_HIT (Contam?)"
            details = "Likely E. coli or Artifact"
            no_hit_count += 1
            
        print(f"Read {idx:<5} | {status:<20} | {details}")

    print("-" * 80)
    print("SUMMARY:")
    print(f"1. Valid Transcript Matches: {mapped_count}")
    print(f"2. Ignored Transcripts (ncRNA?): {tx_not_in_bucket_count}")
    print(f"3. Genomic Regions (Introns/UTRs): {genomic_count}")
    print(f"4. True Unknowns (Bacteria/Junk): {no_hit_count}")
    print("="*40)

if __name__ == "__main__":
    valid_ids = load_bucket_map()
    reads = get_real_reads()
    forensic_analysis(reads, valid_ids)