"""
Science Live - Systematic Review Screening Module
==================================================

Screens papers against a PICO research question defined as a nanopublication.
Optimized for Qwen2.5:14b via Ollama.

Usage:
    from screening import PICOScreener
    
    # Create screener from nanopub URL
    screener = PICOScreener.from_nanopub_url(
        "https://w3id.org/np/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0"
    )
    
    # Screen a single paper
    result = screener.screen_paper(paper)
    
    # Screen multiple papers
    results = screener.screen_papers(papers)
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, List


# =============================================================================
# CONFIGURATION
# =============================================================================

PDF_DIR = "pdfs"
DEFAULT_CHAR_LIMIT = 25000
DEFAULT_MODEL = "qwen2.5:14b"
UNPAYWALL_EMAIL = "contact@vitenhub.no"
ZENODO_CSV_URL = "https://zenodo.org/records/18070378/files/included_studies.csv"


# DATA CLASSES
# =============================================================================

@dataclass
class PICOQuestion:
    """Structured PICO research question."""
    uri: str
    label: str
    description: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    creator_orcid: Optional[str] = None
    creator_name: Optional[str] = None


@dataclass
class Paper:
    """Paper metadata for screening."""
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    authors: Optional[str] = None  # Changed to str for CSV compatibility
    year: Optional[str] = None     # Changed to str for CSV compatibility
    source: Optional[str] = None
    keywords: Optional[List[str]] = None
    source_database: Optional[str] = None  # Track where paper came from


@dataclass
class ScreeningResult:
    """Result of screening a single paper."""
    paper_doi: Optional[str]
    paper_title: str
    decision: str  # "INCLUDE", "EXCLUDE", or "ERROR"
    confidence: float
    reason: str
    matched_population: Optional[str] = None
    matched_intervention: Optional[str] = None
    exclusion_code: Optional[str] = None
    screening_time_ms: Optional[int] = None
    model_used: str = DEFAULT_MODEL
    text_length: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "paper_doi": self.paper_doi,
            "paper_title": self.paper_title,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "matched_population": self.matched_population,
            "matched_intervention": self.matched_intervention,
            "exclusion_code": self.exclusion_code,
            "screening_time_ms": self.screening_time_ms,
            "model_used": self.model_used,
            "text_length": self.text_length
        }


# =============================================================================
# PDF DOWNLOAD AND TEXT EXTRACTION
# =============================================================================

os.makedirs(PDF_DIR, exist_ok=True)

# =============================================================================
# DATA LOADERS (Zenodo, OpenAlex, CrossRef)
# =============================================================================

def load_papers_from_zenodo_csv(
    url: str = ZENODO_CSV_URL,
    enrich_missing_abstracts: bool = True,
    limit: Optional[int] = None
) -> List[Paper]:
    """
    Load papers directly from a Zenodo CSV file.
    
    Args:
        url: URL to the CSV file
        enrich_missing_abstracts: If True, fetch abstracts from OpenAlex for papers without them
        limit: Maximum number of papers to load (None for all)
    
    Returns:
        List of Paper objects
        
    Example:
        papers = load_papers_from_zenodo_csv()
        papers = load_papers_from_zenodo_csv(limit=10, enrich_missing_abstracts=False)
    """
    import csv
    import ast
    
    print(f"Fetching papers from Zenodo CSV...")
    print(f"URL: {url}")
    
    # Download CSV
    req = urllib.request.Request(url, headers={"User-Agent": "ScienceLive/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read().decode('utf-8')
    
    # Parse CSV
    reader = csv.DictReader(content.splitlines())
    rows = list(reader)
    
    print(f"Found {len(rows)} papers in CSV")
    
    if limit:
        rows = rows[:limit]
        print(f"Limited to {limit} papers")
    
    papers = []
    missing_abstracts = 0
    
    for row in rows:
        # Parse authors (stored as string representation of list)
        authors_str = row.get('authors', '[]')
        try:
            authors_list = ast.literal_eval(authors_str) if authors_str else []
            authors = ', '.join(authors_list) if authors_list else ''
        except:
            authors = authors_str
        
        # Extract DOI (remove https://doi.org/ prefix if present)
        doi = row.get('doi', '')
        if doi and doi.startswith('https://doi.org/'):
            doi = doi[16:]  # Remove prefix
        
        # Get abstract
        abstract = row.get('abstract', '')
        if not abstract or abstract.strip() == '':
            missing_abstracts += 1
            abstract = None
        
        paper = Paper(
            doi=doi if doi else None,
            title=row.get('title', ''),
            authors=authors if authors else None,
            year=row.get('year', ''),
            abstract=abstract,
            source_database='Zenodo CSV'
        )
        papers.append(paper)
    
    print(f"Loaded {len(papers)} papers")
    print(f"  With abstracts: {len(papers) - missing_abstracts}")
    print(f"  Missing abstracts: {missing_abstracts}")
    
    # Enrich missing abstracts from OpenAlex
    if enrich_missing_abstracts and missing_abstracts > 0:
        print(f"\nEnriching {missing_abstracts} papers with missing abstracts from OpenAlex...")
        enriched = 0
        
        for i, paper in enumerate(papers):
            if not paper.abstract and paper.doi:
                enriched_paper = enrich_paper_from_openalex(paper)
                if enriched_paper.abstract:
                    papers[i] = enriched_paper
                    enriched += 1
                
                # Rate limit
                time.sleep(0.1)
        
        print(f"  Enriched {enriched} papers with abstracts")
    
    return papers


def enrich_paper_from_openalex(paper: Paper) -> Paper:
    """
    Enrich paper with metadata from OpenAlex API (FREE, no key required).
    Fetches abstract, authors, year if missing.
    
    Args:
        paper: Paper object (must have DOI)
        
    Returns:
        Paper with enriched metadata
    """
    if not paper.doi:
        return paper
    
    try:
        # OpenAlex API
        url = f"https://api.openalex.org/works/https://doi.org/{paper.doi}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ScienceLive/1.0 (mailto:contact@vitenhub.no)",
                "Accept": "application/json"
            }
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        # Extract abstract (OpenAlex stores as inverted index)
        abstract = paper.abstract
        if not abstract and 'abstract_inverted_index' in data and data['abstract_inverted_index']:
            # Reconstruct abstract from inverted index
            inv_index = data['abstract_inverted_index']
            words = []
            for word, positions in inv_index.items():
                for pos in positions:
                    words.append((pos, word))
            words.sort(key=lambda x: x[0])
            abstract = ' '.join(w[1] for w in words)
        
        # Extract authors if missing
        authors = paper.authors
        if not authors and 'authorships' in data:
            author_names = []
            for auth in data['authorships'][:10]:  # Limit to first 10
                if auth.get('author', {}).get('display_name'):
                    author_names.append(auth['author']['display_name'])
            authors = ', '.join(author_names)
        
        # Extract year if missing
        year = paper.year
        if not year and 'publication_year' in data:
            year = str(data['publication_year'])
        
        # Extract title if missing
        title = paper.title or data.get('title', '')
        
        return Paper(
            doi=paper.doi,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            source=paper.source,
            keywords=paper.keywords,
            source_database=paper.source_database
        )
        
    except Exception as e:
        return paper


def load_papers_from_csv_file(
    filepath: str,
    doi_column: str = 'doi',
    title_column: str = 'title',
    abstract_column: str = 'abstract',
    authors_column: str = 'authors',
    year_column: str = 'year',
    enrich_missing: bool = True
) -> List[Paper]:
    """
    Load papers from a local CSV file with configurable column names.
    
    Args:
        filepath: Path to CSV file
        doi_column: Column name for DOI
        title_column: Column name for title
        abstract_column: Column name for abstract
        authors_column: Column name for authors
        year_column: Column name for year
        enrich_missing: Enrich papers missing abstracts from OpenAlex
        
    Returns:
        List of Paper objects
    """
    import csv
    import ast
    
    papers = []
    missing_abstracts = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Extract DOI
            doi = row.get(doi_column, '')
            if doi and doi.startswith('https://doi.org/'):
                doi = doi[16:]
            
            # Parse authors
            authors_str = row.get(authors_column, '')
            try:
                if authors_str.startswith('['):
                    authors_list = ast.literal_eval(authors_str)
                    authors = ', '.join(authors_list)
                else:
                    authors = authors_str
            except:
                authors = authors_str
            
            # Get abstract
            abstract = row.get(abstract_column, '')
            if not abstract or abstract.strip() == '':
                missing_abstracts += 1
                abstract = None
            
            paper = Paper(
                doi=doi if doi else None,
                title=row.get(title_column, ''),
                authors=authors if authors else None,
                year=row.get(year_column, ''),
                abstract=abstract,
                source_database=f'CSV: {filepath}'
            )
            papers.append(paper)
    
    print(f"Loaded {len(papers)} papers from {filepath}")
    print(f"  Missing abstracts: {missing_abstracts}")
    
    if enrich_missing and missing_abstracts > 0:
        print(f"Enriching from OpenAlex...")
        enriched = 0
        for i, paper in enumerate(papers):
            if not paper.abstract and paper.doi:
                enriched_paper = enrich_paper_from_openalex(paper)
                if enriched_paper.abstract:
                    papers[i] = enriched_paper
                    enriched += 1
                time.sleep(0.1)
        print(f"  Enriched {enriched} papers")
    
    return papers



def get_paper_metadata_from_crossref(doi: str) -> Optional[dict]:
    """
    Fetch paper metadata from CrossRef API (FREE, no key required).
    Returns dict with title, abstract, authors, year, etc.
    """
    try:
        url = f"https://api.crossref.org/works/{doi}"
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "ScienceLive/1.0 (mailto:contact@vitenhub.no)"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        work = data.get("message", {})
        
        # Extract title
        title = work.get("title", ["Unknown"])[0] if work.get("title") else "Unknown"
        
        # Extract abstract (may not be available)
        abstract = work.get("abstract", "")
        # Clean HTML tags from abstract
        if abstract:
            abstract = re.sub(r'<[^>]+>', '', abstract)
        
        # Extract authors
        authors = []
        for author in work.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)
        
        # Extract year
        year = None
        if work.get("published-print"):
            year = work["published-print"].get("date-parts", [[None]])[0][0]
        elif work.get("published-online"):
            year = work["published-online"].get("date-parts", [[None]])[0][0]
        
        # Extract journal/source
        source = work.get("container-title", [""])[0] if work.get("container-title") else ""
        
        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "source": source,
            "doi": doi
        }
    except Exception as e:
        return None


def paper_from_doi(doi: str) -> Paper:
    """
    Create a Paper object from just a DOI by fetching metadata from CrossRef.
    
    Example:
        paper = paper_from_doi("10.1038/s41586-021-03819-2")
    """
    metadata = get_paper_metadata_from_crossref(doi)
    
    if metadata:
        return Paper(
            title=metadata["title"],
            abstract=metadata["abstract"],
            doi=doi,
            authors=', '.join(metadata["authors"]) if metadata["authors"] else None,
            year=str(metadata["year"]) if metadata["year"] else None,
            source=metadata["source"],
            source_database="CrossRef"
        )
    else:
        # Fallback: create minimal paper with just DOI
        return Paper(title=f"Paper {doi}", doi=doi, source_database="DOI only")


def papers_from_dois(dois: List[str], verbose: bool = True) -> List[Paper]:
    """
    Create Paper objects from a list of DOIs.
    Fetches metadata from CrossRef with rate limiting.
    
    Example:
        papers = papers_from_dois(["10.1234/abc", "10.5678/xyz"])
    """
    papers = []
    
    if verbose:
        print(f"Fetching metadata for {len(dois)} DOIs from CrossRef...")
    
    for i, doi in enumerate(dois):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(dois)}")
        
        paper = paper_from_doi(doi)
        papers.append(paper)
        
        # Rate limit: be nice to CrossRef
        time.sleep(0.5)
    
    if verbose:
        print(f"✓ Loaded {len(papers)} papers")
    
    return papers


def get_pdf_url_from_unpaywall(doi: str, email: str = UNPAYWALL_EMAIL) -> Optional[str]:
    """
    Get open access PDF URL from Unpaywall API (FREE).
    Returns URL to PDF if available, None otherwise.
    """
    if not doi:
        return None
    
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        req = urllib.request.Request(url, headers={"User-Agent": "ScienceLive/1.0"})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        # Check for best open access location
        best_oa = data.get('best_oa_location')
        if best_oa and best_oa.get('url_for_pdf'):
            return best_oa['url_for_pdf']
        
        # Check all OA locations
        for location in data.get('oa_locations', []):
            if location.get('url_for_pdf'):
                return location['url_for_pdf']
        
        return None
    except Exception:
        return None


def download_pdf(url: str, filepath: str) -> bool:
    """Download PDF from URL to filepath."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ScienceLive/1.0)",
                "Accept": "application/pdf"
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        
        # Verify it's actually a PDF
        with open(filepath, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                os.remove(filepath)
                return False
        
        return True
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def extract_text_from_pdf(filepath: str, max_pages: int = 50) -> str:
    """
    Extract text from PDF using PyMuPDF.
    
    Args:
        filepath: Path to PDF file
        max_pages: Maximum pages to extract (to limit context size)
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(filepath)
        text_parts = []
        
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        return "\n\n".join(text_parts)
    
    except ImportError:
        print("Warning: PyMuPDF not installed. Run: pip install pymupdf")
        return ""
    except Exception as e:
        print(f"Warning: Could not extract text from {filepath}: {e}")
        return ""


def get_fulltext_for_paper(
    paper: Paper, 
    pdf_dir: str = PDF_DIR,
    download_if_missing: bool = True,
    rate_limit_seconds: float = 1.0
) -> str:
    """
    Get full text for a paper.
    
    1. Check if PDF already downloaded
    2. If not, try to get from Unpaywall
    3. Extract text from PDF
    4. Fall back to abstract if no PDF available
    
    Returns: Full text or abstract
    """
    if not paper.doi:
        return paper.abstract or ""
    
    # Safe filename from DOI
    safe_doi = paper.doi.replace("/", "_").replace(":", "_").replace("<", "").replace(">", "")
    pdf_path = os.path.join(pdf_dir, f"{safe_doi}.pdf")
    
    # Check if already downloaded
    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        if text:
            return text
    
    # Try to download
    if download_if_missing:
        pdf_url = get_pdf_url_from_unpaywall(paper.doi)
        
        if pdf_url:
            if download_pdf(pdf_url, pdf_path):
                text = extract_text_from_pdf(pdf_path)
                if text:
                    time.sleep(rate_limit_seconds)  # Rate limit
                    return text
        
        time.sleep(rate_limit_seconds)  # Rate limit even on failure
    
    # Fall back to abstract
    return paper.abstract or ""


# =============================================================================
# NANOPUBLICATION PARSER
# =============================================================================

class NanopubPICOParser:
    """Parse PICO research questions from nanopublication TriG format."""
    
    NANOPUB_SERVERS = [
        "https://np.knowledgepixels.com",
        "https://server.np.dumontierlab.com",
        "https://np.petapico.org",
    ]
    
    @classmethod
    def fetch_nanopub(cls, nanopub_uri: str) -> str:
        """Fetch nanopublication content from the network."""
        # Extract artifact code from URI
        if "/np/" in nanopub_uri:
            artifact_code = nanopub_uri.split("/np/")[-1].split("/")[0]
        else:
            artifact_code = nanopub_uri
        
        for server in cls.NANOPUB_SERVERS:
            try:
                url = f"{server}/{artifact_code}"
                req = urllib.request.Request(
                    url,
                    headers={"Accept": "application/trig", "User-Agent": "ScienceLive/1.0"}
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    return response.read().decode('utf-8')
            except Exception:
                continue
        
        raise ValueError(f"Could not fetch nanopublication: {nanopub_uri}")
    
    @classmethod
    def parse_pico_from_trig(cls, trig_content: str) -> PICOQuestion:
        """Parse PICO elements from TriG content."""
        
        def extract_description(prefix: str) -> str:
            pattern = rf'{prefix}\s+dcterms:description\s+"([^"]+)"'
            match = re.search(pattern, trig_content, re.IGNORECASE)
            return match.group(1) if match else ""
        
        # Extract main question (ends with ?)
        main_desc_pattern = r'dcterms:description\s+"([^"]*\?)"'
        main_match = re.search(main_desc_pattern, trig_content)
        main_description = main_match.group(1) if main_match else ""
        
        # Fallback for question
        if not main_description:
            fallback_pattern = r'dcterms:description\s+"((?:What|How|Which)[^"]+)"'
            fallback_match = re.search(fallback_pattern, trig_content)
            main_description = fallback_match.group(1) if fallback_match else ""
        
        # Extract label
        label_pattern = r'rdfs:label\s+"([^"]+(?:Review|Question)[^"]*)"'
        label_match = re.search(label_pattern, trig_content, re.IGNORECASE)
        label = label_match.group(1) if label_match else "Research Question"
        
        # Extract URI
        prefix_pattern = r'@prefix\s+this:\s*<([^>]+)>'
        prefix_match = re.search(prefix_pattern, trig_content)
        uri = prefix_match.group(1) if prefix_match else ""
        
        # Extract PICO components
        population = extract_description("sub1:population")
        intervention = extract_description("sub1:interventionGroup")
        comparator = extract_description("sub1:comparatorGroup")
        outcome = extract_description("sub1:outcomeGroup")
        
        # Extract creator
        creator_pattern = r'orcid:(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])'
        creator_match = re.search(creator_pattern, trig_content)
        creator_orcid = f"https://orcid.org/{creator_match.group(1)}" if creator_match else None
        
        name_pattern = r'foaf:name\s+"([^"]+)"'
        name_match = re.search(name_pattern, trig_content)
        creator_name = name_match.group(1) if name_match else None
        
        return PICOQuestion(
            uri=uri,
            label=label,
            description=main_description,
            population=population,
            intervention=intervention,
            comparator=comparator,
            outcome=outcome,
            creator_orcid=creator_orcid,
            creator_name=creator_name
        )
    
    @classmethod
    def from_url(cls, nanopub_uri: str) -> PICOQuestion:
        """Fetch and parse PICO from nanopub network."""
        trig_content = cls.fetch_nanopub(nanopub_uri)
        return cls.parse_pico_from_trig(trig_content)
    
    @classmethod
    def from_file(cls, filepath: str) -> PICOQuestion:
        """Load PICO from a local TriG file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.parse_pico_from_trig(f.read())


# =============================================================================
# PROMPT GENERATOR
# =============================================================================

def generate_screening_system_prompt(pico: PICOQuestion) -> str:
    """Generate the screening system prompt from a PICO question."""
    
    def make_bullets(text: str, max_items: int = 5) -> str:
        items = re.split(r'[,;]|\band\b', text)
        items = [item.strip() for item in items if item.strip()][:max_items]
        return "\n   ".join(f"- {item}" for item in items) if items else f"- {text}"
    
    population_bullets = make_bullets(pico.population)
    intervention_bullets = make_bullets(pico.intervention)
    
    return f'''You are a systematic review screener. Your ONLY task is to decide: INCLUDE or EXCLUDE.

## RESEARCH QUESTION

**Title:** {pico.label}

**Question:** {pico.description}

### PICO Framework

| Element | Description |
|---------|-------------|
| **Population** | {pico.population} |
| **Intervention** | {pico.intervention} |
| **Comparator** | {pico.comparator} |
| **Outcome** | {pico.outcome} |

## INCLUSION CRITERIA

A paper should be INCLUDED if it addresses BOTH:

1. **Population Match**: The paper studies or relates to:
   {population_bullets}

2. **Intervention Match**: The paper uses, proposes, or evaluates:
   {intervention_bullets}

## EXCLUSION CODES

- **E1**: No connection to the population/domain described above
- **E2**: No connection to the intervention/methods described above
- **E3**: Tangentially related but not actually about the research question
- **E4**: Not a research paper (news, editorial, dataset description only)
- **E5**: Duplicate or retracted publication

## RESPONSE FORMAT

Return ONLY valid JSON (no markdown, no explanation outside JSON):

{{
  "decision": "INCLUDE" or "EXCLUDE",
  "confidence": 0.0 to 1.0,
  "reason": "One sentence explanation",
  "matched_population": "Which population aspect matched, or null",
  "matched_intervention": "Which intervention aspect matched, or null", 
  "exclusion_code": "E1-E5 if EXCLUDE, null if INCLUDE"
}}

## DECISION RULES

1. Paper CLEARLY matches population AND intervention → INCLUDE (confidence ≥ 0.8)
2. Paper CLEARLY missing population OR intervention → EXCLUDE (confidence ≥ 0.8)
3. Paper UNCERTAIN but seems relevant → INCLUDE (confidence 0.5-0.7)
4. When genuinely uncertain, lean toward INCLUDE with low confidence

Think step by step before answering:
1. What domain/population does this paper address?
2. What methods/interventions does this paper use?
3. Do these match the research question criteria?'''


def generate_user_prompt(paper: Paper, content: str, char_limit: int = DEFAULT_CHAR_LIMIT) -> str:
    """Generate the user prompt with paper content."""
    truncated = content[:char_limit]
    if len(content) > char_limit:
        truncated += f"\n\n[... truncated at {char_limit} characters ...]"
    
    return f'''Screen this paper for inclusion in the systematic review.

**Title:** {paper.title or "Unknown"}

**DOI:** {paper.doi or "Not available"}

**Content (first {char_limit} characters):**
{truncated}

---
Respond with JSON only. No markdown code blocks.'''


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """Simple client for Ollama API."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: int = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> dict:
        """Generate a response from Ollama."""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            url = f"{self.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False


# =============================================================================
# MAIN SCREENER CLASS
# =============================================================================

class PICOScreener:
    """
    Screen papers against a PICO research question.
    
    Example:
        # Create screener with a folder for caching PDFs
        screener = PICOScreener.from_nanopub_url(
            "https://w3id.org/np/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0",
            pdf_folder="./my_review_pdfs"
        )
        
        # Screen papers from DOIs only
        results = screener.screen_dois(["10.1234/abc", "10.5678/xyz"])
        
        # Or screen Paper objects
        results = screener.screen_papers(papers)
    """
    
    def __init__(
        self,
        pico: PICOQuestion,
        model: str = DEFAULT_MODEL,
        char_limit: int = DEFAULT_CHAR_LIMIT,
        pdf_folder: str = PDF_DIR,
        ollama_url: str = "http://localhost:11434",
        debug_dir: Optional[str] = None
    ):
        self.pico = pico
        self.model = model
        self.char_limit = char_limit
        self.pdf_folder = pdf_folder
        self.debug_dir = debug_dir  # Directory to save raw responses for debugging
        
        # Generate system prompt from PICO
        self.system_prompt = generate_screening_system_prompt(pico)
        
        # Initialize Ollama client
        self.client = OllamaClient(base_url=ollama_url, model=model)
        
        # Ensure PDF folder exists
        os.makedirs(pdf_folder, exist_ok=True)
        
        # Create debug dir if specified
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
    
    @classmethod
    def from_nanopub_url(
        cls,
        nanopub_uri: str,
        model: str = DEFAULT_MODEL,
        char_limit: int = DEFAULT_CHAR_LIMIT,
        pdf_folder: str = PDF_DIR,
        debug_dir: Optional[str] = None
    ) -> "PICOScreener":
        """
        Create screener from a nanopublication URL.
        
        Args:
            nanopub_uri: Full URI to PICO nanopublication
                         e.g., "https://w3id.org/np/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0"
            model: Ollama model to use
            char_limit: Max characters of PDF to include (default 25000)
            pdf_folder: Directory to store/cache downloaded PDFs
            debug_dir: Optional directory to save raw LLM responses for debugging
        """
        print(f"Fetching PICO from: {nanopub_uri}")
        pico = NanopubPICOParser.from_url(nanopub_uri)
        print(f"✓ Loaded: {pico.label}")
        return cls(pico=pico, model=model, char_limit=char_limit, pdf_folder=pdf_folder, debug_dir=debug_dir)
    
    @classmethod
    def from_trig_file(
        cls,
        filepath: str,
        model: str = DEFAULT_MODEL,
        char_limit: int = DEFAULT_CHAR_LIMIT,
        pdf_folder: str = PDF_DIR
    ) -> "PICOScreener":
        """Create screener from a local TriG file."""
        pico = NanopubPICOParser.from_file(filepath)
        return cls(pico=pico, model=model, char_limit=char_limit, pdf_folder=pdf_folder)
    
    def screen_doi(self, doi: str, download_pdf: bool = True) -> ScreeningResult:
        """
        Screen a single paper by DOI.
        Fetches metadata from CrossRef and PDF from Unpaywall.
        
        Args:
            doi: The DOI (e.g., "10.1038/s41586-021-03819-2")
            download_pdf: If True, download PDF from Unpaywall
            
        Returns:
            ScreeningResult with decision
        """
        paper = paper_from_doi(doi)
        return self.screen_paper(paper, download_pdf=download_pdf)
    
    def screen_dois(
        self,
        dois: List[str],
        verbose: bool = True,
        delay_between: float = 0.5,
        checkpoint_file: Optional[str] = None
    ) -> List[ScreeningResult]:
        """
        Screen multiple papers by DOI.
        
        Args:
            dois: List of DOIs to screen
            verbose: Print progress
            delay_between: Seconds between papers
            checkpoint_file: Optional path to save progress (enables resume)
            
        Returns:
            List of ScreeningResults
        """
        if verbose:
            print(f"Fetching metadata for {len(dois)} DOIs...")
        
        papers = papers_from_dois(dois, verbose=verbose)
        return self.screen_papers(papers, verbose=verbose, delay_between=delay_between, checkpoint_file=checkpoint_file)
    
    def screen_paper(
        self,
        paper: Paper,
        content: Optional[str] = None,
        download_pdf: bool = True
    ) -> ScreeningResult:
        """
        Screen a single paper.
        
        Args:
            paper: Paper to screen
            content: Optional pre-loaded content. If None, will fetch PDF.
            download_pdf: If True and content is None, download PDF from Unpaywall
            
        Returns:
            ScreeningResult with decision
        """
        start_time = time.time()
        
        # Get content if not provided
        if content is None:
            content = get_fulltext_for_paper(
                paper, 
                pdf_dir=self.pdf_folder,
                download_if_missing=download_pdf
            )
        
        # Handle empty content
        if not content or not content.strip():
            return ScreeningResult(
                paper_doi=paper.doi,
                paper_title=paper.title,
                decision="EXCLUDE",
                confidence=0.0,
                reason="No content available for screening (no PDF or abstract)",
                exclusion_code="E4",
                model_used=self.model,
                text_length=0
            )
        
        # Generate user prompt
        user_prompt = generate_user_prompt(paper, content, self.char_limit)
        
        max_retries = 2
        last_response_text = ""
        
        for attempt in range(max_retries + 1):
            try:
                # Call Ollama
                response = self.client.generate(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt
                )
                
                # Parse response
                response_text = response.get("message", {}).get("content", "")
                last_response_text = response_text
                
                # Use debug mode on retry
                debug_mode = attempt > 0
                result_data = self._parse_json_response(response_text, debug=debug_mode)
                
                # Check if parsing actually succeeded
                reason = result_data.get("reason", "")
                confidence = result_data.get("confidence", 0.5)
                
                # Save raw response for debugging if parsing seems uncertain
                if self.debug_dir and (confidence <= 0.5 or "Insufficient" in reason or not reason):
                    self._save_debug_response(paper, response_text, result_data)
                
                if "Could not parse" in reason and attempt < max_retries:
                    if debug_mode:
                        print(f"\n[DEBUG] Retry {attempt + 1}/{max_retries} for: {paper.title[:40]}...")
                    time.sleep(0.5)
                    continue
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                return ScreeningResult(
                    paper_doi=paper.doi,
                    paper_title=paper.title,
                    decision=result_data.get("decision", "EXCLUDE"),
                    confidence=float(result_data.get("confidence", 0.5)),
                    reason=result_data.get("reason", "No reason provided"),
                    matched_population=result_data.get("matched_population"),
                    matched_intervention=result_data.get("matched_intervention"),
                    exclusion_code=result_data.get("exclusion_code"),
                    screening_time_ms=elapsed_ms,
                    model_used=self.model,
                    text_length=len(content)
                )
                
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                    
                elapsed_ms = int((time.time() - start_time) * 1000)
                return ScreeningResult(
                    paper_doi=paper.doi,
                    paper_title=paper.title,
                    decision="ERROR",
                    confidence=0.0,
                    reason=f"Screening error: {str(e)}",
                    screening_time_ms=elapsed_ms,
                    model_used=self.model,
                    text_length=len(content) if content else 0
                )
        
        # Should not reach here, but just in case
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ScreeningResult(
            paper_doi=paper.doi,
            paper_title=paper.title,
            decision="ERROR",
            confidence=0.0,
            reason="Max retries exceeded",
            screening_time_ms=elapsed_ms,
            model_used=self.model,
            text_length=len(content) if content else 0
        )
    
    def screen_papers(
        self,
        papers: List[Paper],
        verbose: bool = True,
        delay_between: float = 0.5,
        checkpoint_file: Optional[str] = None
    ) -> List[ScreeningResult]:
        """
        Screen multiple papers with optional checkpointing.
        
        Args:
            papers: List of papers to screen
            verbose: Print progress
            delay_between: Seconds between papers (rate limiting)
            checkpoint_file: Path to save results after each paper (JSON lines format).
                            If file exists, already-screened DOIs/titles are skipped.
            
        Returns:
            List of ScreeningResults (only newly screened papers)
        """
        results = []
        screened_dois = set()
        screened_titles = set()
        
        # Get DOIs and titles already in checkpoint
        if checkpoint_file and os.path.exists(checkpoint_file):
            screened_dois, screened_titles = self._get_checkpoint_identifiers(checkpoint_file)
            if verbose and (screened_dois or screened_titles):
                print(f"📂 Checkpoint found: {len(screened_dois)} DOIs + {len(screened_titles)} titles already screened")
        
        # Also deduplicate within input list (track by DOI or title)
        seen_dois = set()
        seen_titles = set()
        
        # Screen papers
        total = len(papers)
        skipped_checkpoint = 0
        skipped_duplicate = 0
        
        for i, paper in enumerate(papers):
            # Normalize title for comparison
            norm_title = paper.title.lower().strip() if paper.title else ""
            
            # Skip if already screened (by DOI or title)
            if paper.doi and paper.doi in screened_dois:
                skipped_checkpoint += 1
                continue
            if not paper.doi and norm_title and norm_title in screened_titles:
                skipped_checkpoint += 1
                continue
            
            # Skip duplicates within input list
            if paper.doi:
                if paper.doi in seen_dois:
                    skipped_duplicate += 1
                    if verbose:
                        print(f"[{i+1}/{total}] Skipping duplicate DOI: {paper.doi}")
                    continue
                seen_dois.add(paper.doi)
            else:
                if norm_title in seen_titles:
                    skipped_duplicate += 1
                    if verbose:
                        print(f"[{i+1}/{total}] Skipping duplicate title: {paper.title[:40]}...")
                    continue
                if norm_title:
                    seen_titles.add(norm_title)
            
            if verbose:
                print(f"[{i+1}/{total}] Screening: {paper.title[:50]}...", end="")
            
            result = self.screen_paper(paper)
            results.append(result)
            
            if verbose:
                status = "✓" if result.decision == "INCLUDE" else "✗"
                print(f" {status} {result.decision} ({result.confidence:.2f}): {result.reason[:50]}")
            
            # Save to checkpoint immediately
            if checkpoint_file:
                self._append_to_checkpoint(checkpoint_file, result)
            
            if i < total - 1:
                time.sleep(delay_between)
        
        if verbose:
            if skipped_checkpoint > 0:
                print(f"⏭️  Skipped {skipped_checkpoint} already-screened papers (from checkpoint)")
            if skipped_duplicate > 0:
                print(f"⏭️  Skipped {skipped_duplicate} duplicate papers in input")
        
        return results
    
    def _get_checkpoint_dois(self, checkpoint_file: str) -> set:
        """Extract DOIs from checkpoint file (fast scan)."""
        dois = set()
        try:
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    if line.strip():
                        # Quick extraction - just get DOI field
                        data = json.loads(line)
                        doi = data.get("paper_doi")
                        if doi:
                            dois.add(doi)
        except Exception:
            pass
        return dois
    
    def _get_checkpoint_identifiers(self, checkpoint_file: str) -> tuple:
        """Extract DOIs and titles from checkpoint file.
        
        Returns:
            Tuple of (set of DOIs, set of normalized titles for papers without DOIs)
        """
        dois = set()
        titles = set()
        try:
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        doi = data.get("paper_doi")
                        title = data.get("paper_title", "")
                        
                        if doi:
                            dois.add(doi)
                        elif title:
                            # For papers without DOI, track by normalized title
                            titles.add(title.lower().strip())
        except Exception:
            pass
        return dois, titles
    
    def _append_to_checkpoint(self, checkpoint_file: str, result: ScreeningResult):
        """Append a single result to checkpoint file (JSON lines format)."""
        try:
            os.makedirs(os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) else ".", exist_ok=True)
            with open(checkpoint_file, 'a') as f:
                data = {
                    "paper_doi": result.paper_doi,
                    "paper_title": result.paper_title,
                    "decision": result.decision,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "matched_population": result.matched_population,
                    "matched_intervention": result.matched_intervention,
                    "exclusion_code": result.exclusion_code,
                    "screening_time_ms": result.screening_time_ms,
                    "model_used": result.model_used,
                    "text_length": result.text_length
                }
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")
    
    def _save_debug_response(self, paper: Paper, raw_response: str, parsed_result: dict):
        """Save raw LLM response for debugging parsing issues."""
        if not self.debug_dir:
            return
        try:
            # Use DOI or title hash for filename
            identifier = paper.doi.replace("/", "_").replace(":", "_") if paper.doi else f"title_{hash(paper.title) % 100000}"
            filepath = os.path.join(self.debug_dir, f"{identifier}.txt")
            
            with open(filepath, 'w') as f:
                f.write(f"=== PAPER ===\n")
                f.write(f"DOI: {paper.doi}\n")
                f.write(f"Title: {paper.title}\n")
                f.write(f"\n=== RAW LLM RESPONSE ===\n")
                f.write(raw_response)
                f.write(f"\n\n=== PARSED RESULT ===\n")
                f.write(json.dumps(parsed_result, indent=2))
        except Exception as e:
            pass  # Silent fail for debug logging
    
    def _parse_json_response(self, response_text: str, debug: bool = False) -> dict:
        """Parse JSON from model response with robust extraction."""
        result = None
        parse_method = None
        
        if debug:
            print(f"\n[DEBUG] Raw response ({len(response_text)} chars):")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        # Method 1: Direct parse
        try:
            result = json.loads(response_text.strip())
            parse_method = "direct"
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code block
        if not result:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    parse_method = "markdown"
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Find JSON object with balanced braces
        if not result:
            # Find start of JSON object
            start = response_text.find('{')
            if start != -1:
                # Find matching closing brace
                depth = 0
                end = start
                for i, c in enumerate(response_text[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                
                json_str = response_text[start:end]
                try:
                    result = json.loads(json_str)
                    parse_method = "balanced_braces"
                except json.JSONDecodeError:
                    # Try to fix common issues
                    fixed = self._fix_json_string(json_str)
                    try:
                        result = json.loads(fixed)
                        parse_method = "fixed_json"
                    except json.JSONDecodeError:
                        pass
        
        # Method 4: Extract key-value pairs with regex
        if not result:
            result = self._extract_fields_from_text(response_text)
            if result.get("decision"):
                parse_method = "regex_extraction"
        
        # Method 5: Last resort - look for decision keyword
        if not result or not result.get("decision"):
            decision = None
            confidence = 0.5
            reason = ""
            
            upper_text = response_text.upper()
            if '"INCLUDE"' in upper_text or "'INCLUDE'" in upper_text or ": INCLUDE" in upper_text:
                decision = "INCLUDE"
            elif '"EXCLUDE"' in upper_text or "'EXCLUDE'" in upper_text or ": EXCLUDE" in upper_text:
                decision = "EXCLUDE"
            elif "INCLUDE" in upper_text and "EXCLUDE" not in upper_text:
                decision = "INCLUDE"
            elif "EXCLUDE" in upper_text:
                decision = "EXCLUDE"
            
            # Try to extract reason
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', response_text)
            if reason_match:
                reason = reason_match.group(1)
            
            # Try to extract confidence
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response_text)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except ValueError:
                    pass
            
            if decision:
                result = {"decision": decision, "confidence": confidence, "reason": reason}
                parse_method = "keyword_search"
            else:
                result = {"decision": "EXCLUDE", "confidence": 0.3, "reason": ""}
                parse_method = "failed"
        
        # Method 6: Extract reasoning from prose text (for non-JSON responses)
        if parse_method == "failed" or (result.get("reason", "") in ["", "Could not parse model response"]):
            prose_reason = self._extract_reason_from_prose(response_text)
            if prose_reason:
                result["reason"] = prose_reason
                # Also try to infer decision from prose if not found
                if parse_method == "failed":
                    lower_prose = prose_reason.lower()
                    if any(kw in lower_prose for kw in ["relevant", "addresses", "applies quantum", "uses quantum", "includes"]):
                        result["decision"] = "INCLUDE"
                        result["confidence"] = 0.6
                    elif any(kw in lower_prose for kw in ["not relevant", "does not", "doesn't", "tangential", "unrelated"]):
                        result["decision"] = "EXCLUDE"
                        result["confidence"] = 0.6
                    parse_method = "prose_extraction"
        
        if debug:
            print(f"[DEBUG] Parse method: {parse_method}")
            print(f"[DEBUG] Result: {result}")
        
        # Ensure reason is never empty - construct from other fields if needed
        reason = result.get("reason", "")
        decision = result.get("decision", "EXCLUDE")
        matched_pop = result.get("matched_population")
        matched_int = result.get("matched_intervention")
        exclusion_code = result.get("exclusion_code")
        confidence = result.get("confidence", 0.5)
        
        # VALIDATION: Suspicious INCLUDE - low confidence with no evidence
        # If INCLUDE but confidence ≤ 0.5 AND no matched fields AND no reason, 
        # this is likely a parsing failure - downgrade to EXCLUDE
        if decision == "INCLUDE" and confidence <= 0.5:
            has_evidence = bool(matched_pop) or bool(matched_int) or bool(reason and reason.strip())
            if not has_evidence:
                result["decision"] = "EXCLUDE"
                result["reason"] = "Insufficient evidence for inclusion (low confidence, no matched criteria)"
                result["exclusion_code"] = "E3"
                result["confidence"] = 0.4
                return result
        
        if not reason or not str(reason).strip():
            if decision == "INCLUDE":
                parts = []
                if matched_pop and matched_pop not in [True, False, "true", "false"]:
                    parts.append(f"Population: {matched_pop}")
                if matched_int and matched_int not in [True, False, "true", "false"]:
                    parts.append(f"Intervention: {matched_int}")
                if parts:
                    result["reason"] = "; ".join(parts)
                else:
                    # No specific matches but still INCLUDE - provide generic reason
                    result["reason"] = "Matches PICO criteria based on content analysis"
            else:
                code_reasons = {
                    "E1": "Does not address target population (biodiversity/conservation)",
                    "E2": "Does not involve quantum computing methods",
                    "E3": "Only tangentially related to research question",
                    "E4": "Not a research article or no accessible content",
                    "E5": "Duplicate or retracted"
                }
                if exclusion_code and exclusion_code in code_reasons:
                    result["reason"] = code_reasons[exclusion_code]
                elif matched_pop is False or str(matched_pop).lower() == "false":
                    result["reason"] = "Does not address target population"
                    result["exclusion_code"] = result.get("exclusion_code") or "E1"
                elif matched_int is False or str(matched_int).lower() == "false":
                    result["reason"] = "Does not involve target intervention"
                    result["exclusion_code"] = result.get("exclusion_code") or "E2"
                else:
                    result["reason"] = "Insufficient relevance to research question"
                    result["exclusion_code"] = result.get("exclusion_code") or "E3"
        
        return result
    
    def _fix_json_string(self, json_str: str) -> str:
        """Try to fix common JSON issues."""
        fixed = json_str
        
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix single quotes to double quotes (careful with apostrophes)
        # Only do this if there are no double quotes
        if '"' not in fixed and "'" in fixed:
            fixed = fixed.replace("'", '"')
        
        # Fix unquoted keys
        fixed = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
        
        # Fix True/False to true/false
        fixed = re.sub(r'\bTrue\b', 'true', fixed)
        fixed = re.sub(r'\bFalse\b', 'false', fixed)
        fixed = re.sub(r'\bNone\b', 'null', fixed)
        
        return fixed
    
    def _extract_fields_from_text(self, text: str) -> dict:
        """Extract screening fields from text using regex."""
        result = {}
        
        # Decision
        decision_match = re.search(r'"?decision"?\s*[:\=]\s*"?(INCLUDE|EXCLUDE)"?', text, re.IGNORECASE)
        if decision_match:
            result["decision"] = decision_match.group(1).upper()
        
        # Confidence
        conf_match = re.search(r'"?confidence"?\s*[:\=]\s*"?([\d.]+)"?', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Reason - handle multi-line
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text)
        if reason_match:
            result["reason"] = reason_match.group(1).replace('\\"', '"').replace('\\n', ' ')
        
        # Exclusion code
        code_match = re.search(r'"?exclusion_code"?\s*[:\=]\s*"?(E[1-5])"?', text, re.IGNORECASE)
        if code_match:
            result["exclusion_code"] = code_match.group(1).upper()
        
        # Matched population
        pop_match = re.search(r'"?matched_population"?\s*[:\=]\s*"?([^",}\]]+)"?', text)
        if pop_match:
            val = pop_match.group(1).strip()
            if val.lower() == 'true':
                result["matched_population"] = True
            elif val.lower() == 'false':
                result["matched_population"] = False
            else:
                result["matched_population"] = val
        
        # Matched intervention
        int_match = re.search(r'"?matched_intervention"?\s*[:\=]\s*"?([^",}\]]+)"?', text)
        if int_match:
            val = int_match.group(1).strip()
            if val.lower() == 'true':
                result["matched_intervention"] = True
            elif val.lower() == 'false':
                result["matched_intervention"] = False
            else:
                result["matched_intervention"] = val
        
        return result
    
    def _extract_reason_from_prose(self, text: str) -> str:
        """Extract a meaningful reason from prose/non-JSON response text."""
        # Clean up the text
        text = text.strip()
        
        # Skip if it looks like JSON
        if text.startswith('{') or '```' in text:
            return ""
        
        # Try to find sentences that explain the decision
        reason_indicators = [
            r"(?:because|since|as)\s+(.{20,150}?)[.!?\n]",
            r"(?:this paper|the paper|the study|this study|it)\s+(is|does|addresses|discusses|focuses|examines|presents|applies|uses|proposes)(.{20,150}?)[.!?\n]",
            r"(?:relevant|included?|excluded?)(?:\s+because)?\s*[:\-]?\s*(.{20,150}?)[.!?\n]",
            r"(?:reason|rationale|justification)[:\s]+(.{20,150}?)[.!?\n]",
        ]
        
        for pattern in reason_indicators:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                reason = match.group(1) if match.lastindex else match.group(0)
                reason = reason.strip()
                # Clean up
                reason = re.sub(r'\s+', ' ', reason)
                if len(reason) > 20:
                    # Capitalize first letter
                    return reason[0].upper() + reason[1:] if reason else reason
        
        # Fallback: Use first substantive sentence (skip very short ones)
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip JSON-like content, very short sentences, or headers
            if len(sentence) > 30 and not sentence.startswith('{') and ':' not in sentence[:15]:
                # Truncate if too long
                if len(sentence) > 200:
                    sentence = sentence[:197] + "..."
                return sentence[0].upper() + sentence[1:] if sentence else sentence
        
        return ""
    
    def get_system_prompt(self) -> str:
        """Return the generated system prompt for inspection."""
        return self.system_prompt
    
    def get_pico_summary(self) -> str:
        """Return a summary of the PICO question."""
        return f"""PICO Research Question
======================
Title: {self.pico.label}
URI: {self.pico.uri}

Question: {self.pico.description}

Population: {self.pico.population}
Intervention: {self.pico.intervention}
Comparator: {self.pico.comparator}
Outcome: {self.pico.outcome}

Creator: {self.pico.creator_name} ({self.pico.creator_orcid})"""
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        return self.client.is_available()
    
    def screen_from_zenodo_csv(
        self,
        url: str = ZENODO_CSV_URL,
        limit: Optional[int] = None,
        enrich_abstracts: bool = True,
        verbose: bool = True,
        checkpoint_file: Optional[str] = None
    ) -> List[ScreeningResult]:
        """
        Load papers from Zenodo CSV and screen them.
        
        Args:
            url: URL to Zenodo CSV file
            limit: Max papers to load (None for all)
            enrich_abstracts: Fetch missing abstracts from OpenAlex
            verbose: Print progress
            checkpoint_file: Optional path to save progress (enables resume)
            
        Returns:
            List of ScreeningResults
        """
        papers = load_papers_from_zenodo_csv(
            url=url,
            enrich_missing_abstracts=enrich_abstracts,
            limit=limit
        )
        return self.screen_papers(papers, verbose=verbose, checkpoint_file=checkpoint_file)


def fetch_pdfs_for_dois(
    dois: List[str],
    pdf_folder: str = PDF_DIR,
    skip_existing: bool = True
) -> dict:
    """
    Batch download PDFs for a list of DOIs.
    
    Args:
        dois: List of DOIs
        pdf_folder: Directory to save PDFs (created if doesn't exist)
        skip_existing: Skip DOIs where PDF already downloaded
        
    Returns:
        Stats dict with counts
    """
    os.makedirs(pdf_folder, exist_ok=True)
    
    stats = {
        "total": len(dois),
        "already_had": 0,
        "downloaded": 0,
        "not_available": 0
    }
    
    print(f"\nFetching PDFs for {len(dois)} DOIs...")
    print(f"Saving to: {pdf_folder}")
    print("(Using Unpaywall API - only open access PDFs available)\n")
    
    for i, doi in enumerate(dois):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(dois)} "
                  f"(downloaded: {stats['downloaded']}, "
                  f"not available: {stats['not_available']})")
        
        safe_doi = doi.replace("/", "_").replace(":", "_").replace("<", "").replace(">", "")
        pdf_path = os.path.join(pdf_folder, f"{safe_doi}.pdf")
        
        # Skip if already exists
        if skip_existing and os.path.exists(pdf_path):
            stats["already_had"] += 1
            continue
        
        # Try to download
        pdf_url = get_pdf_url_from_unpaywall(doi)
        
        if pdf_url:
            if download_pdf(pdf_url, pdf_path):
                stats["downloaded"] += 1
            else:
                stats["not_available"] += 1
        else:
            stats["not_available"] += 1
        
        # Rate limit: 1 request per second
        time.sleep(1.0)
    
    print(f"\n✓ PDF fetching complete:")
    print(f"  Already had: {stats['already_had']}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Not available (no OA): {stats['not_available']}")
    
    return stats


def fetch_pdfs_for_papers(
    papers: List[Paper],
    pdf_dir: str = PDF_DIR,
    skip_existing: bool = True
) -> dict:
    """
    Batch download PDFs for all papers.
    Returns stats about downloads.
    """
    stats = {
        "total": len(papers),
        "already_had": 0,
        "downloaded": 0,
        "not_available": 0,
        "no_doi": 0
    }
    
    print(f"\nFetching PDFs for {len(papers)} papers...")
    print("(Using Unpaywall API - only open access PDFs available)\n")
    
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(papers)} "
                  f"(downloaded: {stats['downloaded']}, "
                  f"not available: {stats['not_available']})")
        
        if not paper.doi:
            stats["no_doi"] += 1
            continue
        
        safe_doi = paper.doi.replace("/", "_").replace(":", "_").replace("<", "").replace(">", "")
        pdf_path = os.path.join(pdf_dir, f"{safe_doi}.pdf")
        
        # Skip if already exists
        if skip_existing and os.path.exists(pdf_path):
            stats["already_had"] += 1
            continue
        
        # Try to download
        pdf_url = get_pdf_url_from_unpaywall(paper.doi)
        
        if pdf_url:
            if download_pdf(pdf_url, pdf_path):
                stats["downloaded"] += 1
            else:
                stats["not_available"] += 1
        else:
            stats["not_available"] += 1
        
        # Rate limit: 1 request per second
        time.sleep(1.0)
    
    print(f"\n✓ PDF fetching complete:")
    print(f"  Already had: {stats['already_had']}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Not available (no OA): {stats['not_available']}")
    print(f"  No DOI: {stats['no_doi']}")
    
    return stats


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def load_results_from_checkpoint(checkpoint_file: str) -> List[ScreeningResult]:
    """
    Load screening results from a checkpoint file.
    
    Args:
        checkpoint_file: Path to checkpoint file (JSON lines format)
        
    Returns:
        List of ScreeningResult objects
    """
    results = []
    if not os.path.exists(checkpoint_file):
        return results
    
    with open(checkpoint_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                result = ScreeningResult(
                    paper_doi=data.get("paper_doi"),
                    paper_title=data.get("paper_title"),
                    decision=data.get("decision"),
                    confidence=data.get("confidence", 0.5),
                    reason=data.get("reason"),
                    matched_population=data.get("matched_population"),
                    matched_intervention=data.get("matched_intervention"),
                    exclusion_code=data.get("exclusion_code"),
                    screening_time_ms=data.get("screening_time_ms"),
                    model_used=data.get("model_used"),
                    text_length=data.get("text_length")
                )
                results.append(result)
    return results


def clear_checkpoint(checkpoint_file: str) -> bool:
    """
    Delete a checkpoint file to start fresh.
    
    Returns:
        True if file was deleted, False if it didn't exist
    """
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        return True
    return False


def checkpoint_summary(checkpoint_file: str) -> dict:
    """
    Get summary statistics from a checkpoint file.
    
    Returns:
        Dict with counts: total, include, exclude, error, suspicious, no_reason
    """
    stats = {
        "total": 0,
        "include": 0,
        "exclude": 0,
        "error": 0,
        "suspicious": 0,  # INCLUDE with low confidence and no evidence
        "no_reason": 0,
        "file": checkpoint_file
    }
    
    try:
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    stats["total"] += 1
                    decision = data.get("decision", "").upper()
                    
                    if decision == "INCLUDE":
                        stats["include"] += 1
                    elif decision == "EXCLUDE":
                        stats["exclude"] += 1
                    elif decision == "ERROR":
                        stats["error"] += 1
                    
                    # Check for suspicious entries
                    reason = data.get("reason", "")
                    confidence = data.get("confidence", 0.5)
                    matched_pop = data.get("matched_population")
                    matched_int = data.get("matched_intervention")
                    
                    if not reason or not str(reason).strip():
                        stats["no_reason"] += 1
                    
                    # Suspicious: INCLUDE with low confidence and no evidence
                    if decision == "INCLUDE" and confidence <= 0.5:
                        has_evidence = bool(matched_pop) or bool(matched_int) or bool(reason and reason.strip())
                        if not has_evidence:
                            stats["suspicious"] += 1
    except FileNotFoundError:
        pass
    
    return stats


def find_suspicious_entries(checkpoint_file: str) -> List[dict]:
    """
    Find suspicious entries in checkpoint that should be re-screened.
    
    Returns list of entries that are:
    - INCLUDE with confidence ≤ 0.5 AND no reason AND no matched fields
    - Any entry with empty reason
    """
    suspicious = []
    
    try:
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    decision = data.get("decision", "").upper()
                    reason = data.get("reason", "")
                    confidence = data.get("confidence", 0.5)
                    matched_pop = data.get("matched_population")
                    matched_int = data.get("matched_intervention")
                    
                    is_suspicious = False
                    
                    # Empty reason is always suspicious
                    if not reason or not str(reason).strip():
                        is_suspicious = True
                    
                    # INCLUDE with low confidence and no evidence
                    if decision == "INCLUDE" and confidence <= 0.5:
                        has_evidence = bool(matched_pop) or bool(matched_int) or bool(reason and reason.strip())
                        if not has_evidence:
                            is_suspicious = True
                    
                    if is_suspicious:
                        suspicious.append(data)
    except FileNotFoundError:
        pass
    
    return suspicious


def remove_entries_from_checkpoint(checkpoint_file: str, dois_to_remove: List[str]) -> int:
    """
    Remove specific DOIs from checkpoint file so they can be re-screened.
    
    Args:
        checkpoint_file: Path to checkpoint file
        dois_to_remove: List of DOIs to remove
        
    Returns:
        Number of entries removed
    """
    if not dois_to_remove:
        return 0
    
    dois_set = set(d.lower() for d in dois_to_remove if d)
    kept_lines = []
    removed_count = 0
    
    try:
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    doi = data.get("paper_doi", "")
                    if doi and doi.lower() in dois_set:
                        removed_count += 1
                    else:
                        kept_lines.append(line)
        
        # Write back
        with open(checkpoint_file, 'w') as f:
            f.writelines(kept_lines)
        
        return removed_count
    except FileNotFoundError:
        return 0


def clean_checkpoint(checkpoint_file: str, remove_suspicious: bool = True) -> dict:
    """
    Clean checkpoint file by removing suspicious/bad entries so they can be re-screened.
    
    Args:
        checkpoint_file: Path to checkpoint file
        remove_suspicious: If True, remove INCLUDE entries with no evidence
        
    Returns:
        Dict with counts of what was removed
    """
    results = {
        "removed_no_reason": 0,
        "removed_suspicious_include": 0,
        "kept": 0
    }
    
    kept_lines = []
    
    try:
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    decision = data.get("decision", "").upper()
                    reason = data.get("reason", "")
                    confidence = data.get("confidence", 0.5)
                    matched_pop = data.get("matched_population")
                    matched_int = data.get("matched_intervention")
                    
                    should_remove = False
                    
                    # Remove entries with empty reason
                    if not reason or not str(reason).strip():
                        should_remove = True
                        results["removed_no_reason"] += 1
                    
                    # Remove suspicious INCLUDEs
                    elif remove_suspicious and decision == "INCLUDE" and confidence <= 0.5:
                        has_evidence = bool(matched_pop) or bool(matched_int)
                        if not has_evidence:
                            should_remove = True
                            results["removed_suspicious_include"] += 1
                    
                    if not should_remove:
                        kept_lines.append(line)
                        results["kept"] += 1
        
        # Write back
        with open(checkpoint_file, 'w') as f:
            f.writelines(kept_lines)
        
        print(f"✓ Cleaned checkpoint: removed {results['removed_no_reason']} with no reason, "
              f"{results['removed_suspicious_include']} suspicious INCLUDEs, kept {results['kept']}")
        
        return results
    except FileNotFoundError:
        print(f"⚠️  Checkpoint file not found: {checkpoint_file}")
        return results


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def screen_papers_with_pico(
    nanopub_uri: str,
    papers: List[Paper],
    model: str = DEFAULT_MODEL,
    char_limit: int = DEFAULT_CHAR_LIMIT,
    pdf_folder: str = PDF_DIR,
    checkpoint_file: Optional[str] = None,
    verbose: bool = True
) -> List[ScreeningResult]:
    """
    Screen Paper objects against a PICO nanopublication.
    
    Args:
        nanopub_uri: URI of the PICO nanopublication
        papers: List of Paper objects to screen
        model: Ollama model name
        char_limit: Max characters to send to model
        pdf_folder: Folder for PDF caching
        checkpoint_file: Optional path to save progress (enables resume)
        verbose: Print progress
    """
    screener = PICOScreener.from_nanopub_url(
        nanopub_uri, model=model, char_limit=char_limit, pdf_folder=pdf_folder
    )
    return screener.screen_papers(papers, verbose=verbose, checkpoint_file=checkpoint_file)


def screen_dois_with_pico(
    nanopub_uri: str,
    dois: List[str],
    model: str = DEFAULT_MODEL,
    char_limit: int = DEFAULT_CHAR_LIMIT,
    pdf_folder: str = PDF_DIR,
    checkpoint_file: Optional[str] = None,
    verbose: bool = True
) -> List[ScreeningResult]:
    """
    One-liner to screen DOIs against a PICO nanopublication.
    
    Example:
        results = screen_dois_with_pico(
            "https://w3id.org/np/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0",
            ["10.1234/abc", "10.5678/xyz"],
            pdf_folder="./my_review",
            checkpoint_file="./my_review/checkpoint.jsonl"  # Auto-resume on restart
        )
    """
    screener = PICOScreener.from_nanopub_url(
        nanopub_uri, model=model, char_limit=char_limit, pdf_folder=pdf_folder
    )
    papers = papers_from_dois(dois, verbose=verbose)
    return screener.screen_papers(papers, verbose=verbose, checkpoint_file=checkpoint_file)


# =============================================================================
# CLI / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Science Live - PICO Screener")
    print("=" * 70)
    
    # Try local file first, then fetch from network
    local_file = "/mnt/user-data/uploads/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0.trig"
    nanopub_uri = "https://w3id.org/np/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0"
    
    if os.path.exists(local_file):
        print(f"\nLoading PICO from local file: {local_file}")
        screener = PICOScreener.from_trig_file(local_file, pdf_folder="./review_pdfs")
    else:
        print(f"\nFetching PICO from: {nanopub_uri}")
        screener = PICOScreener.from_nanopub_url(nanopub_uri, pdf_folder="./review_pdfs")
    
    print("\n" + screener.get_pico_summary())
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("""
# 1. LOAD FROM ZENODO CSV (your papers)
from screening import load_papers_from_zenodo_csv, PICOScreener

papers = load_papers_from_zenodo_csv(
    limit=10,                        # Optional: limit for testing
    enrich_missing_abstracts=True    # Fetch from OpenAlex if missing
)

# 2. CREATE SCREENER
screener = PICOScreener.from_nanopub_url(
    "https://w3id.org/np/RAvk9pmoZ2IberoDe7zUWV0bVithiy6CnbSG5y06YuKM0",
    pdf_folder="./my_review_pdfs"
)

# 3. SCREEN PAPERS
results = screener.screen_papers(papers)

# OR: All-in-one from Zenodo CSV
results = screener.screen_from_zenodo_csv(limit=10)

# 4. EXPORT RESULTS
import json
with open("screening_results.json", "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)

# Summary
included = [r for r in results if r.decision == "INCLUDE"]
excluded = [r for r in results if r.decision == "EXCLUDE"]
print(f"Included: {len(included)}, Excluded: {len(excluded)}")
""")
    
    # Test loading from Zenodo
    print("=" * 70)
    print("TEST: Loading 3 papers from Zenodo CSV")
    print("=" * 70)
    
    try:
        papers = load_papers_from_zenodo_csv(limit=3, enrich_missing_abstracts=True)
        for p in papers:
            print(f"\n  DOI: {p.doi}")
            print(f"  Title: {p.title[:60]}...")
            print(f"  Abstract: {'Yes' if p.abstract else 'No'} ({len(p.abstract) if p.abstract else 0} chars)")
    except Exception as e:
        print(f"  Could not load: {e}")
    
    print("\n" + "=" * 70)
    print("OLLAMA STATUS")
    print("=" * 70)
    if screener.is_ollama_available():
        print(f"✓ Ollama ready with model: {screener.model}")
    else:
        print(f"✗ Ollama not available")
        print(f"  Start with: ollama run {screener.model}")