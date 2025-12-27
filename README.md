# Systematic Review Pipeline with Nanopublications

A complete workflow for conducting PRISMA-compliant systematic reviews using nanopublications for transparent, traceable, and machine-readable research documentation.

## Overview

This pipeline implements a 5-stage PRISMA workflow:

1. **PICO Research Question** - Define your research question
2. **Search Strategy** - Document your search methodology
3. **Search Execution** - Query academic databases via APIs
4. **Study Screening** - AI-assisted title/abstract screening with ASReview
5. **Study Assessment** - Document eligibility criteria and methodology

Each stage produces nanopublications that create a complete provenance chain from research question to included studies.

---

## Installation

### Prerequisites

```bash
# Create conda environment
conda create -n systematic-review python=3.11
conda activate systematic-review

# Install required packages
pip install jupyter pandas numpy requests
pip install rdflib nanopub
pip install asreview asreview-insights
```

### Verify Installation

```bash
# Check ASReview
asreview --version

# Check nanopub
python -c "from nanopub import Nanopub; print('nanopub OK')"
```

---

## Jupyter Notebooks

### Stage 1: PICO Research Question

**Notebook:** `pico-nanopub-from-json.ipynb`

Generates a nanopublication defining your research question using the PICO framework (Population, Intervention, Comparison, Outcome).

**Input:** `pico-config.json`  
**Output:** PICO nanopub (e.g., `https://w3id.org/np/RA...`)

---

### Stage 2: Search Strategy

**Notebook:** `search-strategy-nanopub-from-json.ipynb`

Documents your search methodology including databases, search terms, Boolean operators, and filters.

**Input:** `search-strategy-config.json`  
**Output:** Search Strategy nanopub

---

### Stage 3: Search Execution

**Notebook:** `search-execution-api-queries.ipynb`

Programmatically queries academic databases via their APIs:
- OpenAlex
- arXiv
- PubMed (E-utilities)
- Europe PMC
- Semantic Scholar

#### Output Files

| File | Format | Purpose |
|------|--------|---------|
| `search_results_combined.csv` | CSV | Data analysis, deduplication |
| `search_results_combined.ris` | RIS | Import to screening tools (Rayyan, Zotero, Covidence) |
| `search_results_combined.bib` | BibTeX | Citation management (LaTeX, JabRef) |
| `search_summary.json` | JSON | Machine-readable search metadata |

**Nanopub Notebook:** `search-execution-nanopub-from-json.ipynb`

---

### Stage 4: Study Screening with ASReview LAB

AI-assisted title/abstract screening using active learning.

#### Starting ASReview LAB

```bash
# Start the web interface
asreview lab

# Opens http://localhost:5001 in your browser
```

#### Screening Workflow

1. **Create Project**: Click FILE → Upload your `search_results_combined.ris`
2. **Start Screening**: Go to Reviewer tab
3. **Make Decisions**: Click "Relevant" or "Irrelevant" for each paper
4. **AI Learning**: After ~10-20 decisions, ASReview prioritizes likely-relevant papers
5. **Stop Screening**: When you see mostly irrelevant papers (AI has found most relevant ones)
6. **Export Results**: FILE → Export project as `.asreview` file

#### How ASReview Works

- **You** always make the final decision (Relevant/Irrelevant)
- **AI** learns from your decisions and prioritizes which paper to show next
- **Active Learning** means relevant papers are shown first, saving screening time
- **Typical efficiency**: Review ~30-40% of papers to find >95% of relevant studies

#### Exclusion Reason Codes (Optional - for Notes field)

| Code | Meaning |
|------|---------|
| `offtopic_quantum` | No quantum computing content |
| `offtopic_biodiversity` | No biodiversity application |
| `wrong_type_review` | Review/editorial only |
| `wrong_type_abstract` | Conference abstract only |
| `duplicate` | Duplicate study |
| `language` | Not in English |
| `unavailable` | Full text not accessible |
| `retracted` | Retracted paper |

#### Export Screening Results

**Notebook:** `asreview-to-nanopub.ipynb`

Extracts screening decisions from `.asreview` file and generates:
- `study_inclusion.json` - Ready for nanopub generation
- `included_studies.csv` / `.ris` - Included papers
- `excluded_studies.csv` / `.ris` - Excluded papers
- `prisma_flow_data.json` - PRISMA diagram numbers

---

### Stage 5a: Study Inclusion Nanopubs

**Notebook:** `study-inclusion-nanopub-asreview.ipynb`

Generates ONE nanopub per included study, linking each to your systematic review.

**Input:** `study_inclusion.json` (from ASReview export)  
**Output:** Multiple `.trig` files (one per study with DOI)

---

### Stage 5b: Study Assessment Dataset

**Notebook:** `study-assessment-nanopub-from-json.ipynb`

Generates ONE nanopub summarizing your entire assessment methodology:
- Eligibility criteria (PRISMA Item 5)
- Assessment technique (PRISMA Item 11)
- Study characteristics (PRISMA Item 17)
- Quality assessment (PRISMA Item 18)
- Study results (PRISMA Item 19)

**Input:** `study-assessment-config.json`  
**Output:** Study Assessment nanopub

---

## Example: Quantum Computing × Biodiversity Review

### Published Nanopublications

| Stage | Nanopub URI |
|-------|-------------|
| PICO Research Question | https://w3id.org/np/RA8B3ptXUOsN7obpkFGtA0FBmsh0OnID53wOsUIpSKTcg |
| Search Strategy | https://w3id.org/np/RAJW9kn9Syx7y_1Okl4HPwqUlUssxi0daadJNM1AT8-PU |
| Search Execution | https://w3id.org/np/RAMPy96eCLCXlGR9VvCVf6rJmpN_DlxxarMGm91_5n-O8 |
| Study Inclusion (238 nanopubs) | See [published_uris.json](https://zenodo.org/records/18070378/files/published_uris.json) |
| Study Assessment Dataset | https://w3id.org/np/RAx_ZQScbvsz7Rvqk8scSYx06zojCc6Gjcvkxjj_MKwVM |

### Zenodo Dataset

All screening data deposited at: https://zenodo.org/records/18070378

### Screening Summary

| Metric | Count |
|--------|-------|
| Total records | 1,649 |
| Records screened | 569 |
| Included | 283 |
| Excluded | 286 |
| Not screened (AI stopped) | 1,080 |
| Nanopubs published | 238 |

---

## Files to Commit to GitHub

### Notebooks (commit these)

```
├── pico-nanopub-from-json.ipynb
├── search-strategy-nanopub-from-json.ipynb
├── search-execution-api-queries.ipynb
├── search-execution-nanopub-from-json.ipynb
├── asreview-to-nanopub.ipynb
├── study-inclusion-nanopub-asreview.ipynb
├── study-assessment-nanopub-from-json.ipynb
└── README.md
```

### Configuration Templates (commit these)

```
├── pico-config-template.json
├── search-strategy-config-template.json
├── search-execution-config-template.json
└── study-assessment-config-template.json
```

### DO NOT Commit (add to .gitignore)

```
# Large data files (deposit in Zenodo instead)
*.csv
*.ris
*.bib
*.asreview

# Generated nanopubs
*.trig
*.signed.trig

# Output directories
screening_results/
nanopubs_study_inclusion/

# Personal config with real data
*-config.json
!*-template.json

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
```

### Example .gitignore

```gitignore
# Data files (deposit in Zenodo)
*.csv
*.ris
*.bib
*.asreview
search_summary.json

# Generated nanopubs
*.trig

# Output directories
screening_results/
nanopubs_study_inclusion/

# Jupyter checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc

# Config files with real data (keep templates)
study_inclusion.json
published_uris.json
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEMATIC REVIEW PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PICO Question ──► Nanopub                                   │
│         │                                                        │
│         ▼                                                        │
│  2. Search Strategy ──► Nanopub                                 │
│         │                                                        │
│         ▼                                                        │
│  3. Search Execution ──► .ris/.csv files ──► Nanopub            │
│         │                                                        │
│         ▼                                                        │
│  4. ASReview Screening ──► .asreview project                    │
│         │                                                        │
│         ▼                                                        │
│  5a. Study Inclusion ──► 1 Nanopub per study                    │
│         │                                                        │
│         ▼                                                        │
│  5b. Study Assessment ──► 1 Summary Nanopub                     │
│         │                                                        │
│         ▼                                                        │
│  Zenodo: Deposit all data files                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## License

CC-BY 4.0

## Contact

Anne Fouilloux (ORCID: 0000-0002-1784-2920)
