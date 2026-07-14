import pandas as pd
import os

# --- CONFIG ---
base_dir = "/data3/projects/2025_Assembly/eyh/c_elegans/training_data"
files_to_shrink = [
    "simulation_data_phase1.csv", # The Forward Only data
    "simulation_data_full.csv"    # The Mixed data
]

TARGET_SIZE = 500000  # 500k reads is plenty for fine-tuning

for filename in files_to_shrink:
    input_path = os.path.join(base_dir, filename)
    
    # New filename: add "_small"
    name_part, ext = os.path.splitext(filename)
    output_path = os.path.join(base_dir, f"{name_part}_small{ext}")
    
    print(f"Reading {input_path}...")
    
    # Read only the first 5 rows to check, then read randomly? 
    # No, usually safer to read normally if memory allows, or chunk.
    # 38M rows is big for pandas (~4GB RAM). It should fit on cobi3 (usually 128GB+ RAM).
    try:
        df = pd.read_csv(input_path)
        print(f"  - Original shape: {df.shape}")
        
        # Check if we even have enough data (we definitely do)
        if len(df) > TARGET_SIZE:
            # Random Sample
            df_small = df.sample(n=TARGET_SIZE, random_state=42)
            print(f"  - Subsampled to: {df_small.shape}")
        else:
            df_small = df
            print(f"  - Data smaller than target, keeping original.")
            
        # Save
        df_small.to_csv(output_path, index=False)
        print(f"  - Saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Done! Use the *_small.csv files for training.")