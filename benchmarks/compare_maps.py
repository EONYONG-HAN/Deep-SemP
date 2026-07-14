import os
from pathlib import Path
from collections import Counter

ROOT = Path("/data3/projects/2025_Assembly/eyh/c_elegans")
MAP_CORRECT = ROOT / "buckets/semantic_map.tsv"
MAP_WRONG = ROOT / "buckets/aa_based/map_track_B.tsv"

def parse_map(map_path, name):
    print(f"\n--- Parsing {name} ---")
    tx_to_bucket = {}
    bucket_counts = Counter()
    
    try:
        with open(map_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split('\t')
                
                if len(parts) < 4: 
                    continue
                    
                try:
                    # Assuming Column 2 is bucket ID, Column 3 is TX list
                    b_id = int(parts[0]) 
                    
                    for tx in parts[3].split(','):
                        clean_tx = tx.strip()
                        if clean_tx:
                            tx_to_bucket[clean_tx] = b_id
                            bucket_counts[b_id] += 1
                except ValueError:
                    if line_idx < 5:
                        print(f"  [Warning] Could not parse line {line_idx}: {line.strip()}")
                    continue
                    
    except Exception as e:
        print(f"Error reading {map_path}: {e}")
        return {}, Counter()

    print(f"Total Transcripts Mapped: {len(tx_to_bucket)}")
    print(f"Total Unique Buckets: {len(bucket_counts)}")
    
    # Print the top 5 largest buckets to see if one bucket dominates
    print("Top 5 Largest Buckets:")
    for b_id, count in bucket_counts.most_common(5):
        print(f"  Bucket {b_id}: {count} transcripts")
        
    return tx_to_bucket, bucket_counts

def main():
    print("Starting Map Comparison...")
    
    map1_tx, map1_dist = parse_map(MAP_CORRECT, "CORRECT MAP (semantic_map.tsv)")
    map2_tx, map2_dist = parse_map(MAP_WRONG, "WRONG MAP (map_track_B.tsv)")
    
    if not map1_tx or not map2_tx:
        print("\nFailed to load one or both maps. Check paths and formats.")
        return

    # Compare Transcripts
    print("\n--- OVERLAP ANALYSIS ---")
    
    common_txs = set(map1_tx.keys()).intersection(set(map2_tx.keys()))
    print(f"Transcripts present in both maps: {len(common_txs)}")
    
    exact_matches = 0
    mismatches = 0
    
    for tx in common_txs:
        if map1_tx[tx] == map2_tx[tx]:
            exact_matches += 1
        else:
            mismatches += 1

    print(f"Exact Bucket Matches: {exact_matches}")
    print(f"Bucket Mismatches: {mismatches}")
    
    if len(common_txs) > 0:
        match_pct = (exact_matches / len(common_txs)) * 100
        print(f"Similarity Score: {match_pct:.2f}%")
        
        if match_pct == 100.0:
            print("\n🚨 CONCLUSION: THE MAPS ARE IDENTICAL! 🚨")
            print("This is why the 'wrong' map got 100% accuracy.")
        elif exact_matches > 0:
            print("\nCONCLUSION: The maps are different, but share some identical assignments.")
        else:
            print("\nCONCLUSION: The maps are completely different (0% overlap).")

if __name__ == "__main__":
    main()