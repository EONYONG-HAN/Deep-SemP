library(polyester)
library(Biostrings)

# =========================
# CONFIGURATION
# =========================
fasta_file <- Sys.getenv("DEEPSEMP_CDNA_FASTA", "/data3/projects/2025_Assembly/eyh/c_elegans/reference/c_elegans_cdna.fa.gz")
out_dir    <- Sys.getenv("DEEPSEMP_SIM_OUTDIR", "/data3/projects/2025_Assembly/eyh/c_elegans/simulation/polyester_illumina5")

# =========================
# 1. READ REFERENCE
# =========================
print(paste("Reading reference:", fasta_file))
fasta <- readDNAStringSet(fasta_file)
print(paste("Loaded", length(fasta), "transcripts."))

# Filter out very short transcripts that can't produce 100bp reads
min_len <- 100
fasta <- fasta[width(fasta) >= min_len]
print(paste("After length filter (>= 100bp):", length(fasta), "transcripts remaining."))

# =========================
# 2. WRITE FILTERED FASTA
# =========================
writeXStringSet(fasta, 'sim_input_filtered.fa')

# =========================
# 3. DEFINE EXPRESSION
# Expression weights based on transcript length
# (longer transcripts tend to have more reads in real RNA-seq)
# This makes coverage distribution more realistic than uniform
# =========================
print("Defining expression weights...")

tx_lengths  <- width(fasta)
len_weights <- tx_lengths / mean(tx_lengths)   # length-proportional weight

# reads_per_transcript is a per-transcript vector here
# target ~100 reads per transcript on average (vs 50 before)
# giving ~3M reads total for 30K transcripts
base_reads  <- 100
reads_vec   <- round(base_reads * len_weights)
reads_vec   <- pmax(reads_vec, 10)   # minimum 10 reads per transcript

print(paste("Total reads to simulate:", sum(reads_vec) * 2, "(paired-end)"))
print(paste("Mean reads per transcript:", round(mean(reads_vec))))
print(paste("Min reads per transcript:", min(reads_vec)))
print(paste("Max reads per transcript:", max(reads_vec)))

# fold_changes must be a matrix: nrow = n_transcripts, ncol = n_groups
# With num_reps = c(3,3), we have 2 groups, 3 reps each
# All fold changes = 1 (no DE, just simulate coverage)
n_tx <- length(fasta)
fc_matrix <- matrix(1, nrow = n_tx, ncol = 2)

# =========================
# 4. SIMULATE
# =========================
print("Starting simulation with illumina5 error model...")
print(paste("Output dir:", out_dir))

simulate_experiment(
  fasta            = 'sim_input_filtered.fa',
  outdir           = out_dir,

  # More replicates = more sequence diversity in training data
  num_reps         = c(3, 3),

  fold_changes     = fc_matrix,

  # Length-proportional read counts per transcript
  reads_per_transcript = reads_vec,

  readlen          = 100,
  paired           = TRUE,
  strand_specific  = TRUE,   # keep stranded as before

  # KEY CHANGE: illumina5 error model
  # - Position-dependent error rates (3' end degrades)
  # - Realistic substitution frequencies matching Illumina chemistry
  # - Much closer to real sequencing than 'uniform'
  error_model      = 'illumina5',

  # 0.5% average error rate — matches real Illumina short reads
  # Your previous setting was 0.00001 (100x too low)
  error_rate       = 0.005,

  gzip             = FALSE
)

print("Simulation complete.")
print(paste("Output written to:", out_dir))
print("Next step: parse FASTQ outputs and rebuild training CSV with labels.")