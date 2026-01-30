# Where MinerU is Used in AutoReviewer

## Overview
MinerU is used as a **command-line tool** (not a Python library) to convert PDFs to structured markdown. It's invoked via shell scripts and SLURM batch jobs.

## 1. MinerU Installation

**Location**: `pyproject.toml`
- Package: `mineru==2.6.2` (with extras: `["core", "vllm"]`)
- Also uses: `mineru-vl-utils==0.1.18`

## 2. MinerU Command Invocations

### A. SLURM Batch Jobs (Production)

#### 1. **`sbatch/mineru_array.sbatch`** - Main batch processing
- **Purpose**: Processes PDFs in parallel batches (array job 0-18 = 19 batches)
- **Command**: 
  ```bash
  mineru -p "$INPUT_DIR" -o "$OUTPUT_DIR" -b vlm-vllm-engine --end 14
  ```
- **Input**: `data/mineru_batches/batch_XX/` (symlinks to PDFs)
- **Output**: `data/full_run/md_mineru/batch_XX/`
- **Backend**: `vlm-vllm-engine` (uses VLLM server on port 30000)
- **Limit**: Processes first 14 pages only (`--end 14`)

#### 2. **`sbatch/mineru_2020_fixed.sbatch`** - Fixed 2020 batch
- **Purpose**: Reprocesses fixed 2020 PDFs
- **Command**: Same as above
- **Input**: `data/mineru_batches/batch_2020_fixed/`
- **Output**: `data/full_run/md_mineru/batch_2020_fixed/`

#### 3. **`sbatch/mineru_2020_fixed_pt2.sbatch`** - Fixed 2020 part 2
- **Purpose**: Processes second batch of fixed 2020 PDFs (Dec 16 cutoff)
- **Command**: Same as above
- **Input**: `data/mineru_batches/batch_2020_fixed_pt2/`
- **Output**: `data/full_run/md_mineru/batch_2020_fixed_pt2/`

### B. Manual/Testing Commands

**Location**: `sh/mineru_cmds.txt` (reference commands)

#### Pipeline Mode (OCR + VLM):
```bash
uv run mineru -p /path/to/paper.pdf -o /output/dir \
  --source local -b pipeline -m ocr --formula false
```

#### VLM-Only Mode:
```bash
uv run mineru -p /path/to/paper.pdf -o /output/dir \
  --source local -b vlm-transformers
```

#### VLLM Engine Mode (Production):
```bash
# Start VLLM server first
nohup mineru-openai-server \
  --engine vllm \
  --port 30000 \
  --data-parallel-size 1 \
  > mineru-server.log 2>&1 &

# Then run MinerU
nohup mineru \
  -p /path/to/pdf.pdf \
  -o /output/dir \
  -b vlm-vllm-engine \
  > mineru.log 2>&1
```

## 3. MinerU Processing Pipeline

### Step 1: Prepare Batches
**Script**: `scripts/prepare_mineru_batches.py`
- Creates symlink batches in `data/mineru_batches/batch_XX/`
- Filters out withdrawn/desk-rejected papers
- Batch size: ~2200 PDFs per batch
- Creates symlinks (not copies) to save disk space

### Step 2: Run MinerU
**Scripts**: `sbatch/mineru_array.sbatch` (or fixed batch scripts)
- Runs MinerU on each batch directory
- Processes PDFs → Markdown + JSON
- Outputs to `data/full_run/md_mineru/batch_XX/`

### Step 3: Merge Outputs
**Script**: `scripts/merge_mineru_outputs.py`
- Merges batch outputs into year-based structure
- Final location: `data/full_run/mds/{year}/{submission_id}_{download_id}/`

### Step 4: Normalize Content
**Script**: `lib/normalize_mineru.py`
- Processes MinerU output (`content_list.json`)
- Filters content (removes title pages, footnotes, etc.)
- Generates clean markdown, filtered JSON, chopped PDF
- **SLURM jobs**: 
  - `sbatch/normalize_mineru.sbatch` (single job)
  - `sbatch/normalize_mineru_array.sbatch` (array job, 40 workers)

## 4. MinerU Output Structure

After MinerU runs, each PDF produces:
```
{output_dir}/{submission_id}_{download_id}/
├── vlm/
│   ├── {submission_id}_{download_id}_content_list.json  # Structured content
│   └── {submission_id}_{download_id}.md                # Markdown
└── images/  # Extracted images
```

## 5. MinerU Usage in Code

### A. Reading MinerU Outputs

**File**: `lib/normalize_mineru.py`
- **Function**: `build_mineru_index()` - Builds index of MinerU output files
- **Function**: `find_mineru_by_prefix()` - Finds MinerU output for a submission ID
- Reads `content_list.json` files from MinerU output directories

**File**: `scripts/generate_hf_dataset.py`
- Uses `build_mineru_index()` and `find_mineru_by_prefix()` from `lib.normalize_mineru`
- Links submissions to their MinerU outputs
- Includes MinerU paths in HuggingFace dataset

**File**: `scripts/analyze_mineru_stats.py`
- Analyzes MinerU conversion statistics
- Checks content quality (abstract presence, references, etc.)

### B. Processing MinerU Outputs

**File**: `lib/normalize_mineru.py`
- Main normalization pipeline
- Processes `content_list.json` files
- Filters and cleans content
- Generates normalized outputs

## 6. MinerU Configuration

### Backend Options:
- **`vlm-vllm-engine`**: Uses VLLM server (production, requires GPU)
- **`vlm-transformers`**: Uses transformers library directly
- **`pipeline`**: OCR + VLM pipeline mode

### Command Flags:
- `-p`: Input PDF path or directory
- `-o`: Output directory
- `-b`: Backend (`vlm-vllm-engine`, `vlm-transformers`, `pipeline`)
- `--end 14`: Process only first 14 pages
- `--source local`: Local file source
- `-m ocr`: OCR mode
- `--formula false`: Disable formula extraction

## 7. Key Directories

- **PDF Input**: `data/full_run/pdfs/{year}/`
- **Batch Preparation**: `data/mineru_batches/batch_XX/` (symlinks)
- **MinerU Output**: `data/full_run/md_mineru/batch_XX/`
- **Merged Output**: `data/full_run/mds/{year}/{submission_id}/`
- **Normalized Output**: `data/full_run/normalized/{year}/{submission_id}/`

## 8. Summary

**MinerU is NOT called from Python code directly**. Instead:
1. **Shell scripts** (`sbatch/*.sbatch`) invoke the `mineru` command-line tool
2. **Python scripts** (`scripts/prepare_mineru_batches.py`) prepare input batches
3. **Python scripts** (`lib/normalize_mineru.py`) process MinerU outputs
4. **SLURM** manages parallel execution across batches

The workflow is:
```
PDFs → Prepare Batches → MinerU CLI → Merge Outputs → Normalize → Final Dataset
```
