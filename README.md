# Deep-SemP — Deep Semantic Partitioning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-download-blue.svg)](https://cobi.knu.ac.kr/tools/deepsemp/best_student_model.pt)
[![Tutorial](https://img.shields.io/badge/tutorial-quickstart-green.svg)](tutorial/README.md)

Reference-free RNA-seq de novo transcriptome assembly that **partitions reads into
semantically coherent bins using ESM-2 protein embeddings before independent Trinity
assembly**. By segregating biologically unrelated reads before they enter a de Bruijn
graph, Deep-SemP reduces redundancy and computational cost while improving structural
precision and unique-transcript recovery.

Validated across four *Caenorhabditis* species (~20 My of divergence, no retraining)
plus *H. sapiens* as a phylogenetic boundary case.

> Companion code for the manuscript *"Parallelizing transcriptome assembly via deep
> semantic read partitioning"* (Bioinformatics, in prep).

## How it works — the four steps

| Step | What it does | Folder |
|------|--------------|--------|
| **1. Semantic map** | ESM-2 embed the reference proteome → hierarchically cluster into K=50 buckets | `pipeline/step1_semantic_map/` |
| **2. Train + distill** | Fine-tune a DNABERT-2 teacher classifier → distill into a 12.8 M-param, 8-layer student | `pipeline/step2_train_distill/` |
| **3. Partition + assemble** | 4-GPU parallel read partitioning → k-mer-complexity-adaptive Trinity assembly | `pipeline/step3_partition_assemble/` |
| **4. Post-process + evaluate** | CD-HIT-EST → CAP3 → CPC2, then BUSCO / GffCompare / BLASTn evaluation | `pipeline/step4_postprocess_eval/` |

A full walkthrough with commands is in **[`docs/tutorial.md`](docs/tutorial.md)**.

## Repository layout

```
Deep-SemP/
├── pipeline/            # the four-step pipeline (run in order)
│   ├── step1_semantic_map/
│   ├── step2_train_distill/
│   ├── step3_partition_assemble/
│   └── step4_postprocess_eval/
├── analysis/            # scripts that regenerate paper figures & tables
├── benchmarks/          # throughput / batch-size / tokenization benchmarks
├── tests/               # partitioning-accuracy sanity checks
├── tutorial/            # toy end-to-end example (download model → route → assemble)
├── configs/             # paths.example.sh — central path configuration
├── docs/                # tutorial.md
├── models/              # (git-ignored) downloaded student model
└── results/             # (git-ignored) data, models, figures, DBs, logs
```

## Pretrained model

The trained 8-layer student model is hosted on the COBI lab page (it exceeds GitHub's
free-tier size limit and is git-ignored here):

```bash
bash tutorial/download_model.sh          # -> models/best_student_model.pt
```

Direct link: https://cobi.knu.ac.kr/tools/deepsemp/best_student_model.pt

## Try it in 5 minutes (toy example)

New users should start with the toy quickstart — it downloads the model and runs
routing + assembly on a ~few-MB sample, on CPU or GPU:

**→ [`tutorial/README.md`](tutorial/README.md)**

## Quickstart

### 1. Install

```bash
# option A — conda (recommended; pulls Trinity/BUSCO/etc. from bioconda)
conda env create -f environment.yml
conda activate deep-semp

# option B — pip for the Python parts only
pip install -r requirements.txt
```

Install a CUDA build of PyTorch matching your driver (see https://pytorch.org).
CPC2 is installed separately (see `environment.yml` notes).

### 2. Configure paths

```bash
cp configs/paths.example.sh configs/paths.sh
$EDITOR configs/paths.sh                      # point at your data / references / tools
export DEEPSEMP_CONFIG="$PWD/configs/paths.sh"   # pipeline scripts auto-source this
```

Every script reads paths as `${VAR:-default}`, so setting them in `configs/paths.sh`
(or as environment variables) overrides the defaults without editing code.

### 3. Run (C. elegans example)

```bash
# Step 3 — partition reads across GPUs, then assemble
bash pipeline/step3_partition_assemble/parallel_route.sh
bash pipeline/step3_partition_assemble/compute_bucket_complexity.sh
bash pipeline/step3_partition_assemble/run_trinity_adaptive.sh

# Step 4 — post-process + evaluate
bash pipeline/step4_postprocess_eval/postprocess.sh \
     --fasta_dir "$BASE_OUT_TRINITY" --label deepsemp --outdir results/postprocess \
     --r1 "$R1" --r2 "$R2" --ref_dna "$REF_DNA" --ref_gtf "$REF_GTF"
```

Steps 1–2 (building the semantic map and training the models) are only needed to
reproduce or retrain; the shipped student checkpoint can be used directly for Step 3.
See the tutorial for the full sequence.

## Data availability

RNA-seq datasets are on NCBI SRA: `SRR36278209` (*C. elegans*), `SRR31870168` +
`SRR31870169` (*C. briggsae*), `SRR34855585` (*C. remanei*), `SRR34855622`
(*C. latens*), `SRR37112488` (*H. sapiens*).

## Citing

If you use Deep-SemP, please cite the manuscript (citation to be added on publication).

## License

MIT — see [LICENSE](LICENSE).
