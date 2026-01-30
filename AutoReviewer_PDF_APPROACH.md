# AutoReviewer PDF Handling Summary

## Overview
AutoReviewer downloads PDFs from OpenReview and uses **MinerU** (a PDF-to-Markdown conversion tool) to parse them into structured markdown format.

## 1. PDF Storage Structure

### Directory Layout
```
data/full_run/pdfs/
├── 2020/
│   ├── {submission_id}_{download_id}.pdf
│   ├── {submission_id}_.pdf  (if no download_id)
│   └── ...
├── 2021/
├── 2022/
└── ...
```

### Naming Convention
- **Format**: `{submission_id}_{download_id}.pdf`
- **Example**: `HkgTTh4FDH_HyBhJshoH.pdf` or `B1erJJrYPH_.pdf`
- The `download_id` is:
  - For API v1 (2020-2023): revision_id (for pre-camera-ready PDFs)
  - For API v2 (2024+): edit_id (for rebuttal revisions) or empty (use submission_id)

### PDF Manifest (`pdf_manifest.csv`)
A CSV file tracks all PDFs with:
- `submission_id`: OpenReview paper ID
- `year`: Conference year
- `decision`: Accept/Reject/Withdrawn
- `pdf_path`: Path in OpenReview (e.g., `/pdf/abc123.pdf`)
- `pdf_hash`: Hash extracted from path
- `pdf_source`: `'revision'`, `'note'`, or `'rebuttal_edit'`
- `download_id`: ID needed to download (revision_id, edit_id, or empty)
- `title`: Paper title

## 2. PDF Download Process

### API v1 (2020-2023)
```python
import openreview
client = openreview.Client(baseurl='https://api.openreview.net')

# For pre-camera-ready PDFs (revisions)
if download_id:
    pdf_bytes = client.get_pdf(download_id, is_reference=True)
else:
    # Fallback to submission PDF
    pdf_bytes = client.get_pdf(submission_id)
```

### API v2 (2024+)
```python
import openreview.api
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# For note PDFs (2024, 2026)
pdf_bytes = client.get_pdf(submission_id)

# For edit PDFs (2025 rebuttals)
url = client.baseurl + '/notes/edits/attachment'
params = {'id': edit_id, 'name': 'pdf'}
response = client.session.get(url, params=params)
pdf_bytes = response.content
```

### Download Script (`lib/download_pdfs.py`)
- Reads `pdf_manifest.csv`
- Downloads PDFs to `data/full_run/pdfs/{year}/`
- Handles rate limiting with `time.sleep(0.5)` between requests
- Supports `--skip-existing` flag to resume interrupted downloads

## 3. PDF Parsing with MinerU

### What is MinerU?
MinerU is a PDF-to-Markdown conversion tool that:
- Extracts text, tables, figures, and formulas from PDFs
- Uses VLM (Vision Language Models) for better accuracy
- Outputs structured JSON (`content_list.json`) and Markdown files

### MinerU Usage
```bash
# Pipeline mode (OCR + VLM)
mineru -p /path/to/paper.pdf -o /output/dir --source local -b pipeline -m ocr --formula false

# VLM-only mode
mineru -p /path/to/paper.pdf -o /output/dir --source local -b vlm-transformers
```

### MinerU Output Structure
```
{output_dir}/
└── {submission_id}_{download_id}/
    ├── vlm/
    │   ├── {submission_id}_{download_id}_content_list.json  # Structured content
    │   └── {submission_id}_{download_id}.md                # Markdown
    ├── images/  # Extracted images
    └── ...
```

### Processing Pipeline
1. **Prepare Batches** (`scripts/prepare_mineru_batches.py`):
   - Creates symlink batches of PDFs for parallel processing
   - Filters out withdrawn/desk-rejected papers
   - Batch size: ~2200 PDFs per batch

2. **Run MinerU** (via SLURM/sbatch scripts):
   - Processes batches in parallel
   - Outputs to `data/full_run/md_mineru/batch_XX/`

3. **Merge Outputs** (`scripts/merge_mineru_outputs.py`):
   - Merges batch outputs into year-based structure
   - Final location: `data/full_run/mds/{year}/{submission_id}_{download_id}/`

4. **Normalize Content** (`lib/normalize_mineru.py`):
   - Filters content (removes before abstract, after references, footnotes)
   - Standardizes headers (uppercase level-1 headers only)
   - Removes GitHub links and git-related sentences
   - Generates:
     - Filtered JSON (`content_list_filtered.json`)
     - Clean Markdown (`{id}.md`)
     - Chopped PDF (redacted PDF with only main content)
     - Metadata JSON

## 4. PDF Content Extraction Details

### Pre-Camera-Ready PDF Selection
- **2020-2023**: Uses last revision PDF from year before conference (e.g., ICLR 2020 → last PDF from 2019)
- **2024**: Uses submission PDF directly (no rebuttals stored)
- **2025**: Uses last Rebuttal_Revision edit PDF, or falls back to submission PDF
- **2026**: Uses submission PDF directly

### Content Filtering
The normalization process (`normalize_mineru.py`) removes:
- Content before the first header (title page, etc.)
- Content after "References" section
- Footnotes
- Sentences containing GitHub/GitLab URLs
- Reproducibility and Acknowledgments sections (between body and references)

### Output Files
After normalization, each paper has:
- `{id}.md`: Clean markdown version
- `content_list_filtered.json`: Structured content with text blocks, images, tables
- `{id}_chopped.pdf`: PDF with only main content (redacted)
- `metadata.json`: Paper metadata

## 5. Key Files

- **`lib/download_pdfs.py`**: Downloads PDFs from OpenReview
- **`lib/extract_pdf_manifest.py`**: Generates PDF manifest CSV
- **`lib/normalize_mineru.py`**: Normalizes MinerU output
- **`scripts/prepare_mineru_batches.py`**: Prepares PDF batches for MinerU
- **`scripts/merge_mineru_outputs.py`**: Merges MinerU batch outputs
- **`lib/utils.py`**: Utility functions for finding PDF/MD paths

## 6. Summary

**PDF Storage**:
- Raw PDFs stored as: `data/full_run/pdfs/{year}/{submission_id}_{download_id}.pdf`
- Tracked in `pdf_manifest.csv`

**PDF Parsing**:
- Uses **MinerU** tool to convert PDFs to structured markdown
- Outputs stored as: `data/full_run/mds/{year}/{submission_id}_{download_id}/`
- Content is filtered and normalized to remove non-paper content

**Key Points**:
- PDFs are downloaded using OpenReview API clients
- Pre-camera-ready PDFs are selected based on year-specific heuristics
- MinerU handles the actual PDF parsing (text extraction, OCR, VLM)
- Normalization pipeline cleans and structures the extracted content
