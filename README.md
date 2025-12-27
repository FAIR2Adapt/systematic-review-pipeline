# Systematic Review pipeline

## Jupyter Notebooks:

### PICO Research Questions

- pico-nanopub-from-json.ipynb

### Search Strategy

- search-strategy-nanopub-from-json.ipynb

### Execute Research Strategy against different databases

- search-execution-api-queries.ipynb 

#### Output Files

##### `search_results_combined.csv`

Tabular format containing all retrieved records with fields: title, authors, year, journal, DOI, abstract, source database, and open access status. Use for data analysis, deduplication checks, or custom filtering.

##### `search_results_combined.ris`

RIS (Research Information Systems) format for import into systematic review screening tools. Compatible with:
- Rayyan (https://rayyan.ai)
- Zotero
- Covidence
- EPPI-Reviewer
- DistillerSR

##### `search_results_combined.bib`

BibTeX format for citation management in LaTeX workflows or import into reference managers like JabRef or Zotero.

##### `search_summary.json`

Machine-readable summary containing:
- Total record count
- Records per database
- Search execution date
- Query parameters used

### PRISMA Search Execution Dataset Nanopublication Generator

- search-execution-nanopub-from-json.ipynb

## Example with Biodiversity and Quantum Computing
 
- PICO Research Question: https://w3id.org/np/RA8B3ptXUOsN7obpkFGtA0FBmsh0OnID53wOsUIpSKTcg
- Search Strategy: https://w3id.org/np/RAJW9kn9Syx7y_1Okl4HPwqUlUssxi0daadJNM1AT8-PU
- Search Execution: https://w3id.org/np/RAMPy96eCLCXlGR9VvCVf6rJmpN_DlxxarMGm91_5n-O8
- All nanopublications related to individual assessment are gathered in: https://zenodo.org/records/18070378/files/published_uris.json
- A PRISMA study assessment dataset : https://w3id.org/np/RAx_ZQScbvsz7Rvqk8scSYx06zojCc6Gjcvkxjj_MKwVM
