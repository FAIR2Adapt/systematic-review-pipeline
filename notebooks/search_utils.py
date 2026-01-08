"""
search_utils.py - Reusable utilities for systematic review database searches

This module provides functions to search multiple academic databases via their APIs,
export results in various formats, and deduplicate records.

Supported Databases:
- OpenAlex (free, comprehensive coverage)
- arXiv (preprints)
- Semantic Scholar (AI-enhanced discovery)
- PubMed (biomedical via Entrez)
- Europe PMC (European biomedical)
- CORE (open access aggregator)
- BASE (Bielefeld Academic Search Engine)

Usage:
    from search_utils import SearchExecutor, load_search_terms_from_json
    
    search_terms = load_search_terms_from_json("search-strategy.json")
    executor = SearchExecutor(search_terms=search_terms, ...)
    results = executor.run_all_searches()
    executor.export_results(results, output_dir="./results")
"""

import requests
import json
import time
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# =============================================================================
# ONTOLOGY LABEL FETCHING
# =============================================================================

def fetch_wikidata_label(qid: str, lang: str = "en") -> Optional[str]:
    """
    Fetch label for a Wikidata entity.
    
    Args:
        qid: Wikidata ID (e.g., "Q169950" or full URI)
        lang: Language code for label
    
    Returns:
        Label string or None if not found
    """
    # Extract QID from URI if needed
    original_input = qid
    if "wikidata.org" in qid:
        match = re.search(r'Q\d+', qid)
        if match:
            qid = match.group()
    
    # Method 1: Try SPARQL endpoint first (different subdomain, often more accessible)
    try:
        sparql_url = "https://query.wikidata.org/sparql"
        query = f"""
        SELECT ?label WHERE {{
            wd:{qid} rdfs:label ?label .
            FILTER(LANG(?label) = "{lang}")
        }}
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": "SystematicReviewBot/1.0"
        }
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                return bindings[0]["label"]["value"]
    except Exception:
        pass  # Try next method
    
    # Method 2: Try the REST API (EntityData)
    try:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        headers = {"User-Agent": "SystematicReviewBot/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            entities = data.get("entities", {})
            if qid in entities:
                labels = entities[qid].get("labels", {})
                if lang in labels:
                    return labels[lang]["value"]
    except Exception:
        pass  # Try next method
    
    # Method 3: Try the Action API
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "labels",
            "languages": lang,
            "format": "json"
        }
        headers = {"User-Agent": "SystematicReviewBot/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            entities = data.get("entities", {})
            if qid in entities:
                labels = entities[qid].get("labels", {})
                if lang in labels:
                    return labels[lang]["value"]
    except Exception:
        pass
    
    return None


def fetch_obo_label(uri: str) -> Optional[str]:
    """
    Fetch label for an OBO Foundry ontology term using OLS API.
    
    Args:
        uri: Full OBO PURL (e.g., "http://purl.obolibrary.org/obo/ENVO_01000174")
    
    Returns:
        Label string or None if not found
    """
    # OLS4 API endpoint
    base_url = "https://www.ebi.ac.uk/ols4/api/terms"
    
    params = {
        "iri": uri
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # OLS returns embedded terms
            terms = data.get("_embedded", {}).get("terms", [])
            if terms:
                return terms[0].get("label")
    except Exception as e:
        print(f"  Warning: Could not fetch OBO label for {uri}: {e}")
    
    return None


def fetch_label(uri: str) -> Optional[str]:
    """
    Fetch label for any supported ontology URI.
    
    Supports:
    - Wikidata (wikidata.org)
    - OBO Foundry (purl.obolibrary.org)
    
    Args:
        uri: Full URI
    
    Returns:
        Label string or None if not found
    """
    if "wikidata.org" in uri:
        return fetch_wikidata_label(uri)
    elif "purl.obolibrary.org" in uri:
        return fetch_obo_label(uri)
    else:
        print(f"  Warning: Unknown ontology for URI: {uri}")
        return None


def fetch_all_labels(uris: List[str], delay: float = 0.2, use_manual_fallback: bool = True) -> Dict[str, str]:
    """
    Fetch labels for a list of URIs.
    
    Args:
        uris: List of ontology URIs
        delay: Delay between API calls (rate limiting)
        use_manual_fallback: Use built-in manual mappings if API fails
    
    Returns:
        Dict mapping URI -> label
    """
    labels = {}
    
    # Separate Wikidata and other URIs
    wikidata_uris = [u for u in uris if "wikidata.org" in u]
    other_uris = [u for u in uris if "wikidata.org" not in u]
    
    print(f"Fetching labels for {len(uris)} terms...")
    
    # Try batch fetch for Wikidata first
    if wikidata_uris:
        print(f"  Trying batch fetch for {len(wikidata_uris)} Wikidata terms...")
        batch_labels = fetch_wikidata_labels_batch(wikidata_uris)
        
        # Check which ones succeeded
        succeeded = len(batch_labels)
        if succeeded > 0:
            print(f"  Batch fetch: {succeeded}/{len(wikidata_uris)} labels found")
            labels.update(batch_labels)
        
        # Handle remaining/failed Wikidata URIs
        for uri in wikidata_uris:
            if uri in labels:
                print(f"  ✓ {labels[uri]}")
            else:
                # Try individual fetch
                label = fetch_wikidata_label(uri)
                if label:
                    labels[uri] = label
                    print(f"  ✓ {label}")
                elif use_manual_fallback:
                    # Try manual fallback
                    for key, manual_label in MANUAL_LABELS.items():
                        if key in uri:
                            labels[uri] = manual_label
                            print(f"  ✓ {manual_label} (from manual mapping)")
                            break
                    else:
                        match = re.search(r'Q\d+', uri)
                        fallback = match.group() if match else uri.split("/")[-1]
                        labels[uri] = fallback
                        print(f"  ? {fallback} (no label found)")
                else:
                    match = re.search(r'Q\d+', uri)
                    fallback = match.group() if match else uri.split("/")[-1]
                    labels[uri] = fallback
                    print(f"  ? {fallback} (no label found)")
                time.sleep(delay)
    
    # Fetch other URIs individually
    for uri in other_uris:
        label = fetch_label(uri)
        if label:
            labels[uri] = label
            print(f"  ✓ {label}")
        elif use_manual_fallback:
            # Try manual fallback
            for key, manual_label in MANUAL_LABELS.items():
                if key in uri:
                    labels[uri] = manual_label
                    print(f"  ✓ {manual_label} (from manual mapping)")
                    break
            else:
                fallback = uri.split("/")[-1]
                labels[uri] = fallback
                print(f"  ? {fallback} (no label found)")
        else:
            fallback = uri.split("/")[-1]
            labels[uri] = fallback
            print(f"  ? {fallback} (no label found)")
        time.sleep(delay)
    
    return labels


def fetch_wikidata_labels_batch(uris: List[str], lang: str = "en") -> Dict[str, str]:
    """
    Fetch labels for multiple Wikidata entities in one request.
    
    Args:
        uris: List of Wikidata URIs
        lang: Language code
    
    Returns:
        Dict mapping URI -> label
    """
    labels = {}
    
    # Extract QIDs
    qid_to_uri = {}
    for uri in uris:
        match = re.search(r'Q\d+', uri)
        if match:
            qid_to_uri[match.group()] = uri
    
    if not qid_to_uri:
        return labels
    
    qids = list(qid_to_uri.keys())
    
    # Try SPARQL batch query first
    try:
        sparql_url = "https://query.wikidata.org/sparql"
        values = " ".join([f"wd:{qid}" for qid in qids])
        query = f"""
        SELECT ?item ?label WHERE {{
            VALUES ?item {{ {values} }}
            ?item rdfs:label ?label .
            FILTER(LANG(?label) = "{lang}")
        }}
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": "SystematicReviewBot/1.0"
        }
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            for binding in data.get("results", {}).get("bindings", []):
                item_uri = binding["item"]["value"]
                label = binding["label"]["value"]
                # Find original URI
                qid = item_uri.split("/")[-1]
                if qid in qid_to_uri:
                    labels[qid_to_uri[qid]] = label
            return labels
    except Exception:
        pass
    
    # Fallback: Action API batch (up to 50 at a time)
    try:
        url = "https://www.wikidata.org/w/api.php"
        for i in range(0, len(qids), 50):
            batch = qids[i:i+50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": lang,
                "format": "json"
            }
            headers = {"User-Agent": "SystematicReviewBot/1.0"}
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for qid, entity in data.get("entities", {}).items():
                    if qid in qid_to_uri:
                        label_data = entity.get("labels", {}).get(lang)
                        if label_data:
                            labels[qid_to_uri[qid]] = label_data["value"]
    except Exception:
        pass
    
    return labels


# Manual label mapping as fallback
MANUAL_LABELS = {
    # Wikidata - Wildfire/Fire related
    "Q169950": "wildfire",
    "Q107434304": "forest fire",
    "Q33374019": "burned area",
    
    # Wikidata - Sentinel-2
    "Q4302480": "Sentinel-2",
    "Q131985539": "Sentinel-2 data",
    
    # Wikidata - ML/AI
    "Q2539": "machine learning",
    "Q197536": "deep learning",
    "Q192776": "artificial neural network",
    "Q245748": "random forest",
    "Q7397": "software",
    "Q11660": "artificial intelligence",
    
    # ENVO
    "ENVO_01000174": "forest biome",
}


def get_label_with_fallback(uri: str) -> str:
    """Get label from API or fall back to manual mapping."""
    # Try API first
    label = fetch_label(uri)
    if label:
        return label
    
    # Check manual mapping
    for key, manual_label in MANUAL_LABELS.items():
        if key in uri:
            return manual_label
    
    # Last resort: return the ID
    return uri.split("/")[-1]


def load_search_terms_from_json(json_path: str, group_by: str = "auto") -> Dict[str, List[str]]:
    """
    Load search strategy JSON and fetch labels for all search term URIs.
    
    Args:
        json_path: Path to search strategy JSON file
        group_by: How to group terms - "auto" (by URI pattern), "flat" (single list)
    
    Returns:
        Dict of grouped search terms with labels
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    strategy = data.get("search_strategy", data)
    uris = strategy.get("search_terms", [])
    
    # Fetch all labels
    uri_to_label = fetch_all_labels(uris)
    
    if group_by == "flat":
        return {"terms": list(uri_to_label.values())}
    
    # Auto-group by URI pattern/ontology
    groups = {
        "wikidata": [],
        "envo": [],
        "other": []
    }
    
    for uri, label in uri_to_label.items():
        if "wikidata.org" in uri:
            groups["wikidata"].append(label)
        elif "ENVO" in uri:
            groups["envo"].append(label)
        else:
            groups["other"].append(label)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def load_config_from_json(json_path: str) -> Dict[str, Any]:
    """
    Load full configuration from search strategy JSON.
    
    Supports two formats for search_terms:
    - New: [{"uri": "...", "label": "..."}]
    - Old: ["uri1", "uri2"]  (will use URI as label)
    
    Returns dict with:
    - databases: list of database URLs
    - start_year: int
    - end_year: int  
    - search_terms: list of {"uri": ..., "label": ...}
    - search_groups: dict of grouped labels for Boolean query
    - labels: list of just the labels
    - label: search strategy label
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    strategy = data.get("search_strategy", data)
    time_period = strategy.get("time_period", {})
    
    start_date = time_period.get("start", "2015-01-01")
    end_date = time_period.get("end", "2025-12-31")
    
    # Handle both old and new search_terms formats
    raw_terms = strategy.get("search_terms", [])
    search_terms = []
    
    for term in raw_terms:
        if isinstance(term, dict):
            # New format: {"uri": "...", "label": "..."}
            search_terms.append(term)
        else:
            # Old format: just URI string
            search_terms.append({"uri": term, "label": term.split("/")[-1]})
    
    return {
        "label": strategy.get("label", "Systematic Review"),
        "databases": strategy.get("databases", []),
        "start_year": int(start_date.split("-")[0]),
        "end_year": int(end_date.split("-")[0]),
        "search_terms": search_terms,
        "search_groups": strategy.get("search_groups", {}),
        "labels": [t["label"] for t in search_terms],
        "search_term_uris": [t["uri"] for t in search_terms],
        "languages": strategy.get("languages", []),
        "methodology_notes": strategy.get("methodology_notes", "")
    }


# Check for optional dependencies
def check_dependencies():
    """Check which optional packages are available."""
    deps = {}
    
    try:
        import arxiv
        deps["arxiv"] = True
    except ImportError:
        deps["arxiv"] = False
    
    try:
        from Bio import Entrez
        deps["entrez"] = True
    except ImportError:
        deps["entrez"] = False
    
    return deps


AVAILABLE_DEPS = check_dependencies()


@dataclass
class SearchConfig:
    """Configuration for a systematic review search."""
    label: str
    search_terms: Dict[str, List[str]]  # e.g., {"topic1": [...], "topic2": [...]}
    start_year: int
    end_year: int
    databases: List[str]
    email: str = "user@example.com"
    max_results: int = 500
    output_dir: Path = field(default_factory=lambda: Path("./search_results"))
    
    @classmethod
    def from_json(cls, json_path: str, email: str = "user@example.com", 
                  search_terms: Dict[str, List[str]] = None, max_results: int = 500):
        """Load configuration from a search strategy JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        
        strategy = data.get("search_strategy", data)
        
        time_period = strategy.get("time_period", {})
        start_date = time_period.get("start", "2015-01-01")
        end_date = time_period.get("end", "2025-12-31")
        
        return cls(
            label=strategy.get("label", "Systematic Review Search"),
            search_terms=search_terms or {},
            start_year=int(start_date.split("-")[0]),
            end_year=int(end_date.split("-")[0]),
            databases=strategy.get("databases", []),
            email=email,
            max_results=max_results
        )


@dataclass
class SearchRecord:
    """Standardized record from any database."""
    source: str
    id: str
    doi: str
    title: str
    year: Optional[int]
    authors: str
    journal: str
    abstract: str
    type: str
    is_oa: bool
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "id": self.id,
            "doi": self.doi,
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
            "journal": self.journal,
            "abstract": self.abstract,
            "type": self.type,
            "is_oa": self.is_oa,
            **self.extra
        }


# =============================================================================
# DATABASE SEARCH FUNCTIONS
# =============================================================================

def build_boolean_query(search_terms: Dict[str, List[str]], 
                        or_operator: str = " OR ", 
                        and_operator: str = " AND ",
                        quote_terms: bool = False,
                        field_prefix: str = "") -> str:
    """
    Build a consistent Boolean query from grouped search terms.
    
    Query structure: (group1_term1 OR group1_term2) AND (group2_term1 OR group2_term2) ...
    
    Args:
        search_terms: Dict mapping group names to lists of terms
        or_operator: Operator for OR (e.g., " OR ", " | ")
        and_operator: Operator for AND (e.g., " AND ", " ")
        quote_terms: Whether to wrap terms in quotes
        field_prefix: Prefix for each term (e.g., "all:" for arXiv)
    
    Returns:
        Boolean query string
    """
    query_parts = []
    for topic, terms in search_terms.items():
        if terms:
            if quote_terms:
                formatted_terms = [f'{field_prefix}"{term}"' for term in terms]
            else:
                formatted_terms = [f'{field_prefix}{term}' for term in terms]
            topic_query = or_operator.join(formatted_terms)
            query_parts.append(f"({topic_query})")
    return and_operator.join(query_parts)


def print_query_for_database(db_name: str, query: str) -> None:
    """Print the query being used for a database."""
    if len(query) > 100:
        print(f"Searching {db_name}: {query[:97]}...")
    else:
        print(f"Searching {db_name}: {query}")

def search_openalex(search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                    max_results: int = 500, email: str = None) -> Tuple[List[Dict], int]:
    """
    Search OpenAlex using their REST API.
    
    OpenAlex doesn't support Boolean AND/OR in the search parameter.
    Instead, we use multiple title.search filters (which are ANDed together).
    
    Args:
        search_terms: Dict with topic keys and lists of search terms
        start_year: Start year for date filter
        end_year: End year for date filter
        max_results: Maximum results to retrieve
        email: Email for polite API access
    
    Returns:
        Tuple of (list of records, total count available)
    """
    # Build display query for logging
    display_query = build_boolean_query(search_terms, or_operator=" OR ", and_operator=" AND ")
    
    # For OpenAlex, we need to build filters differently
    # Each group becomes a title_and_abstract.search filter with OR (|) between terms
    # Multiple filters are ANDed together
    filter_parts = [f"publication_year:{start_year}-{end_year}"]
    
    for group_name, terms in search_terms.items():
        # Join terms with | for OR within group
        # Use quotes for multi-word terms
        quoted_terms = []
        for term in terms:
            if " " in term:
                quoted_terms.append(f'"{term}"')
            else:
                quoted_terms.append(term)
        group_filter = "|".join(quoted_terms)
        filter_parts.append(f"title_and_abstract.search:{group_filter}")
    
    base_url = "https://api.openalex.org/works"
    
    params = {
        "filter": ",".join(filter_parts),
        "per_page": 200,
        "cursor": "*"
    }
    
    if email:
        params["mailto"] = email
    
    results = []
    total_count = 0
    
    print_query_for_database("OpenAlex", display_query)
    print(f"  (OpenAlex filter: {params['filter'][:100]}...)")
    
    while True:
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"OpenAlex Error: {response.status_code} - {response.text[:200]}")
            break
            
        data = response.json()
        
        if total_count == 0:
            total_count = data.get("meta", {}).get("count", 0)
            print(f"OpenAlex: Total results available: {total_count}")
        
        works = data.get("results", [])
        if not works:
            break
            
        for work in works:
            # Safely extract journal name
            journal = ""
            primary_loc = work.get("primary_location")
            if primary_loc:
                source_obj = primary_loc.get("source")
                if source_obj:
                    journal = source_obj.get("display_name", "")
            
            record = {
                "source": "OpenAlex",
                "id": work.get("id", ""),
                "doi": work.get("doi", ""),
                "title": work.get("title", ""),
                "year": work.get("publication_year"),
                "authors": "; ".join([
                    name for name in [
                        (a.get("author") or {}).get("display_name") 
                        for a in work.get("authorships", [])
                    ] if name
                ]),
                "journal": journal,
                "abstract": work.get("abstract", "") or "",
                "type": work.get("type", ""),
                "is_oa": work.get("open_access", {}).get("is_oa", False)
            }
            results.append(record)
            
        if max_results and len(results) >= max_results:
            results = results[:max_results]
            break
            
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        
        time.sleep(0.1)
    
    print(f"OpenAlex: Retrieved {len(results)} records")
    return results, total_count


def search_arxiv(search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                 max_results: int = 500, categories: List[str] = None) -> Tuple[List[Dict], int]:
    """
    Search arXiv using their API.
    
    Args:
        search_terms: Dict with topic keys and lists of search terms
        start_year: Start year for date filter
        end_year: End year for date filter
        max_results: Maximum results to retrieve
        categories: Optional list of arXiv categories to filter
    
    Returns:
        Tuple of (list of records, total count)
    """
    if not AVAILABLE_DEPS.get("arxiv"):
        print("arXiv package not available. Install with: pip install arxiv")
        return [], 0
    
    import arxiv
    
    # Build Boolean query with arXiv syntax: all:"term"
    query = build_boolean_query(search_terms, or_operator=" OR ", and_operator=" AND ", 
                                quote_terms=True, field_prefix="all:")
    
    print_query_for_database("arXiv", query)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results or 1000,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    for paper in client.results(search):
        pub_year = paper.published.year
        if start_year <= pub_year <= end_year:
            record = {
                "source": "arXiv",
                "id": paper.entry_id,
                "doi": paper.doi or "",
                "title": paper.title,
                "year": pub_year,
                "authors": "; ".join([a.name for a in paper.authors]),
                "journal": "arXiv preprint",
                "abstract": paper.summary,
                "type": "preprint",
                "is_oa": True,
                "categories": ", ".join(paper.categories)
            }
            results.append(record)
    
    print(f"arXiv: Retrieved {len(results)} records")
    return results, len(results)


def search_semantic_scholar(search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                            max_results: int = 500) -> Tuple[List[Dict], int]:
    """
    Search Semantic Scholar using their API.
    
    Note: S2 API has limited Boolean support. Uses + for required terms.
    
    Args:
        search_terms: Dict with topic keys and lists of search terms
        start_year: Start year for date filter
        end_year: End year for date filter
        max_results: Maximum results to retrieve
    
    Returns:
        Tuple of (list of records, total count available)
    """
    # S2 API works best with simpler queries
    # Build: +group1_term1 +group2_term1 (required terms from each group)
    query_parts = []
    for topic, terms in search_terms.items():
        if terms:
            # Take first term from each group as required
            query_parts.append(f"+{terms[0]}")
    
    # Add optional terms
    for topic, terms in search_terms.items():
        for term in terms[1:3]:  # Add up to 2 more terms per group
            query_parts.append(term)
    
    query = " ".join(query_parts)
    
    # Also build the Boolean version for display
    bool_query = build_boolean_query(search_terms, or_operator=" OR ", and_operator=" AND ")
    print_query_for_database("Semantic Scholar", bool_query)
    print(f"  (S2 query: {query[:60]}...)")
    
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    params = {
        "query": query,
        "year": f"{start_year}-{end_year}",
        "fields": "paperId,externalIds,title,year,authors,venue,abstract,isOpenAccess",
        "limit": min(max_results or 100, 100)
    }
    
    headers = {"Accept": "application/json"}
    
    results = []
    offset = 0
    total_count = 0
    
    while True:
        params["offset"] = offset
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 429:
            print("Semantic Scholar: Rate limited, waiting 10s...")
            time.sleep(10)
            continue
            
        if response.status_code != 200:
            print(f"Semantic Scholar Error: {response.status_code}")
            break
        
        data = response.json()
        
        if total_count == 0:
            total_count = data.get("total", 0)
            print(f"Semantic Scholar: Total results available: {total_count}")
        
        papers = data.get("data", [])
        if not papers:
            break
        
        for paper in papers:
            record = {
                "source": "Semantic Scholar",
                "id": paper.get("paperId", ""),
                "doi": paper.get("externalIds", {}).get("DOI", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "authors": "; ".join([a.get("name", "") for a in paper.get("authors", [])]),
                "journal": paper.get("venue", ""),
                "abstract": paper.get("abstract", "") or "",
                "type": "article",
                "is_oa": paper.get("isOpenAccess", False)
            }
            results.append(record)
        
        if max_results and len(results) >= max_results:
            results = results[:max_results]
            break
            
        offset += len(papers)
        if offset >= total_count:
            break
            
        time.sleep(3)  # Longer delay to avoid rate limiting
    
    print(f"Semantic Scholar: Retrieved {len(results)} records")
    return results, total_count


def search_pubmed(search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                  max_results: int = 500, email: str = None) -> Tuple[List[Dict], int]:
    """
    Search PubMed using Entrez E-utilities.
    
    Args:
        search_terms: Dict with topic keys and lists of search terms
        start_year: Start year for date filter
        end_year: End year for date filter
        max_results: Maximum results to retrieve
        email: Required email for NCBI API
    
    Returns:
        Tuple of (list of records, total count available)
    """
    if not AVAILABLE_DEPS.get("entrez"):
        print("Biopython not available. Install with: pip install biopython")
        return [], 0
    
    from Bio import Entrez
    
    Entrez.email = email or "user@example.com"
    
    # Build PubMed query with field tags
    query_parts = []
    for topic, terms in search_terms.items():
        topic_terms = " OR ".join([f'"{t}"[Title/Abstract]' for t in terms])
        query_parts.append(f"({topic_terms})")
    query = " AND ".join(query_parts) + f" AND ({start_year}:{end_year}[pdat])"
    
    # Print Boolean version for consistency
    bool_query = build_boolean_query(search_terms, or_operator=" OR ", and_operator=" AND ")
    print_query_for_database("PubMed", bool_query)
    
    # Search
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results or 10000)
    record = Entrez.read(handle)
    handle.close()
    
    id_list = record["IdList"]
    total_count = int(record["Count"])
    
    print(f"PubMed: Found {total_count} records, retrieving {len(id_list)}...")
    
    if not id_list:
        return [], total_count
    
    # Fetch details
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    
    results = []
    for article in records.get("PubmedArticle", []):
        medline = article.get("MedlineCitation", {})
        art = medline.get("Article", {})
        
        # Get authors
        authors = []
        for author in art.get("AuthorList", []):
            if "LastName" in author:
                name = f"{author.get('LastName', '')} {author.get('ForeName', '')}".strip()
                authors.append(name)
        
        # Get abstract
        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join([str(p) for p in abstract_parts]) if abstract_parts else ""
        
        # Get year
        pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")
        
        # Get DOI
        doi = ""
        for eid in article.get("PubmedData", {}).get("ArticleIdList", []):
            if eid.attributes.get("IdType") == "doi":
                doi = str(eid)
                break
        
        rec = {
            "source": "PubMed",
            "id": f"PMID:{medline.get('PMID', '')}",
            "doi": doi,
            "title": str(art.get("ArticleTitle", "")),
            "year": int(year) if year.isdigit() else None,
            "authors": "; ".join(authors),
            "journal": art.get("Journal", {}).get("Title", ""),
            "abstract": abstract,
            "type": "article",
            "is_oa": False
        }
        results.append(rec)
    
    print(f"PubMed: Retrieved {len(results)} records")
    return results, total_count


def search_europepmc(search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                     max_results: int = 500) -> Tuple[List[Dict], int]:
    """
    Search Europe PMC using their REST API.
    
    Args:
        search_terms: Dict with topic keys and lists of search terms
        start_year: Start year for date filter
        end_year: End year for date filter
        max_results: Maximum results to retrieve
    
    Returns:
        Tuple of (list of records, total count available)
    """
    # Build query with quoted terms
    query_parts = []
    for topic, terms in search_terms.items():
        topic_terms = " OR ".join([f'"{t}"' for t in terms])
        query_parts.append(f"({topic_terms})")
    query = " AND ".join(query_parts) + f" AND (PUB_YEAR:[{start_year} TO {end_year}])"
    
    # Print Boolean version for consistency
    bool_query = build_boolean_query(search_terms, or_operator=" OR ", and_operator=" AND ")
    print_query_for_database("Europe PMC", bool_query)
    
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    params = {
        "query": query,
        "format": "json",
        "pageSize": 1000,
        "resultType": "core"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Europe PMC Error: {response.status_code}")
        return [], 0
    
    data = response.json()
    total_count = data.get("hitCount", 0)
    
    print(f"Europe PMC: Found {total_count} records")
    
    results = []
    for item in data.get("resultList", {}).get("result", [])[:max_results]:
        record = {
            "source": "Europe PMC",
            "id": item.get("id", ""),
            "doi": item.get("doi", ""),
            "title": item.get("title", ""),
            "year": item.get("pubYear"),
            "authors": item.get("authorString", ""),
            "journal": item.get("journalTitle", ""),
            "abstract": item.get("abstractText", ""),
            "type": item.get("pubType", ""),
            "is_oa": item.get("isOpenAccess", "N") == "Y"
        }
        results.append(record)
    
    print(f"Europe PMC: Retrieved {len(results)} records")
    return results, total_count


def search_core(search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                max_results: int = 500, api_key: str = None) -> Tuple[List[Dict], int]:
    """
    Search CORE using their API (requires free API key).
    
    Args:
        search_terms: Dict with topic keys and lists of search terms
        start_year: Start year for date filter
        end_year: End year for date filter
        max_results: Maximum results to retrieve
        api_key: CORE API key (get free at https://core.ac.uk/services/api)
    
    Returns:
        Tuple of (list of records, total count available)
    """
    if not api_key:
        print("CORE: API key required. Get free key at https://core.ac.uk/services/api")
        return [], 0
    
    # Build query
    all_terms = []
    for topic, terms in search_terms.items():
        all_terms.extend(terms)
    query = " AND ".join(all_terms[:5])
    
    base_url = "https://api.core.ac.uk/v3/search/works"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {
        "q": query,
        "limit": min(max_results or 100, 100)
    }
    
    print(f"Searching CORE...")
    
    response = requests.get(base_url, params=params, headers=headers)
    
    if response.status_code != 200:
        print(f"CORE Error: {response.status_code}")
        return [], 0
    
    data = response.json()
    total_count = data.get("totalHits", 0)
    
    print(f"CORE: Found {total_count} records")
    
    results = []
    for item in data.get("results", []):
        year = item.get("yearPublished")
        if year and start_year <= year <= end_year:
            record = {
                "source": "CORE",
                "id": item.get("id", ""),
                "doi": item.get("doi", ""),
                "title": item.get("title", ""),
                "year": year,
                "authors": "; ".join(item.get("authors", [])),
                "journal": item.get("publisher", ""),
                "abstract": item.get("abstract", "") or "",
                "type": "article",
                "is_oa": True
            }
            results.append(record)
    
    print(f"CORE: Retrieved {len(results)} records")
    return results, total_count


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_csv(records: List[Dict], filepath: Path) -> None:
    """Export records to CSV format."""
    if not records:
        print("No records to export")
        return
    
    fieldnames = ["source", "id", "doi", "title", "year", "authors", 
                  "journal", "abstract", "type", "is_oa"]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    
    print(f"Exported {len(records)} records to {filepath}")


def export_to_ris(records: List[Dict], filepath: Path) -> None:
    """Export records to RIS format (for Zotero/Rayyan import)."""
    if not records:
        print("No records to export")
        return
    
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write("TY  - JOUR\n")
            f.write(f"TI  - {rec.get('title', '')}\n")
            
            for author in rec.get("authors", "").split("; "):
                if author:
                    f.write(f"AU  - {author}\n")
            
            f.write(f"PY  - {rec.get('year', '')}\n")
            f.write(f"JO  - {rec.get('journal', '')}\n")
            
            if rec.get("doi"):
                f.write(f"DO  - {rec.get('doi')}\n")
            
            if rec.get("abstract"):
                f.write(f"AB  - {rec.get('abstract', '')[:1000]}\n")
            
            f.write(f"N1  - Source: {rec.get('source', '')}\n")
            f.write("ER  - \n\n")
    
    print(f"Exported {len(records)} records to {filepath}")


def export_to_bibtex(records: List[Dict], filepath: Path) -> None:
    """Export records to BibTeX format."""
    if not records:
        print("No records to export")
        return
    
    with open(filepath, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            first_author = rec.get("authors", "Unknown").split(";")[0].split()[-1] if rec.get("authors") else "Unknown"
            year = rec.get("year", "XXXX")
            key = f"{first_author}{year}_{i}"
            
            f.write(f"@article{{{key},\n")
            f.write(f"  title = {{{rec.get('title', '')}}},\n")
            f.write(f"  author = {{{rec.get('authors', '')}}},\n")
            f.write(f"  year = {{{year}}},\n")
            f.write(f"  journal = {{{rec.get('journal', '')}}},\n")
            if rec.get("doi"):
                f.write(f"  doi = {{{rec.get('doi')}}},\n")
            f.write(f"  note = {{Source: {rec.get('source', '')}}},\n")
            f.write("}\n\n")
    
    print(f"Exported {len(records)} records to {filepath}")


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate_by_doi(records: List[Dict]) -> Tuple[List[Dict], int, int]:
    """
    Remove duplicates based on DOI.
    
    Returns:
        Tuple of (unique records, duplicate count, records without DOI count)
    """
    seen_dois = set()
    unique = []
    duplicates = 0
    no_doi = 0
    
    for rec in records:
        doi = (rec.get("doi") or "").strip().lower()
        if not doi:
            no_doi += 1
            unique.append(rec)
        elif doi not in seen_dois:
            seen_dois.add(doi)
            unique.append(rec)
        else:
            duplicates += 1
    
    return unique, duplicates, no_doi


# =============================================================================
# SEARCH EXECUTOR CLASS
# =============================================================================

class SearchExecutor:
    """
    Main class to execute systematic review searches across multiple databases.
    
    Example:
        executor = SearchExecutor(
            search_terms={
                "satellite": ["Sentinel-2", "Sentinel 2"],
                "application": ["wildfire", "forest fire", "burned area"],
                "method": ["machine learning", "deep learning"]
            },
            start_year=2015,
            end_year=2025,
            email="user@example.com"
        )
        results = executor.run_all_searches()
        executor.export_results(results)
    """
    
    # Mapping of database URL patterns to search functions
    DATABASE_FUNCTIONS = {
        "openalex": search_openalex,
        "arxiv": search_arxiv,
        "semanticscholar": search_semantic_scholar,
        "pubmed": search_pubmed,
        "europepmc": search_europepmc,
        "core.ac.uk": search_core,
    }
    
    # Databases that don't have API support
    MANUAL_DATABASES = {
        "scholar.google": "Google Scholar requires manual search or Publish or Perish software",
        "base-search": "BASE requires manual search at https://www.base-search.net/",
        "webofscience": "Web of Science requires institutional subscription",
        "scopus": "Scopus requires institutional subscription",
    }
    
    def __init__(self, search_terms: Dict[str, List[str]], start_year: int, end_year: int,
                 email: str = "user@example.com", max_results: int = 500,
                 databases: List[str] = None):
        """
        Initialize the search executor.
        
        Args:
            search_terms: Dict mapping topic names to lists of search terms
            start_year: Start year for date filter
            end_year: End year for date filter
            email: Email for API access
            max_results: Maximum results per database
            databases: List of database URLs to search (searches all if None)
        """
        self.search_terms = search_terms
        self.start_year = start_year
        self.end_year = end_year
        self.email = email
        self.max_results = max_results
        self.databases = databases or list(self.DATABASE_FUNCTIONS.keys())
        
        self.results_summary = {
            "search_date": datetime.now().isoformat(),
            "databases": {}
        }
    
    def _get_search_function(self, database_url: str):
        """Get the search function for a database URL."""
        db_lower = database_url.lower()
        
        # Check for manual databases first
        for key, message in self.MANUAL_DATABASES.items():
            if key in db_lower:
                return None, key
        
        # Check for supported databases
        for key, func in self.DATABASE_FUNCTIONS.items():
            if key in db_lower:
                return func, key
        
        return None, None
    
    def search_database(self, database: str) -> List[Dict]:
        """Search a single database."""
        func, db_name = self._get_search_function(database)
        
        # Check if it's a manual database
        if db_name in self.MANUAL_DATABASES:
            print(f"\n⚠ MANUAL: {self.MANUAL_DATABASES[db_name]}")
            self.results_summary["databases"][db_name] = {
                "total_available": "manual",
                "retrieved": 0,
                "note": self.MANUAL_DATABASES[db_name]
            }
            return []
        
        if not func:
            print(f"⚠ Unknown database: {database}")
            return []
        
        # Special handling for different function signatures
        if db_name in ["openalex.org", "pubmed.ncbi.nlm.nih.gov"]:
            results, total = func(self.search_terms, self.start_year, self.end_year,
                                  self.max_results, self.email)
        else:
            results, total = func(self.search_terms, self.start_year, self.end_year,
                                  self.max_results)
        
        self.results_summary["databases"][db_name] = {
            "total_available": total,
            "retrieved": len(results)
        }
        
        return results
    
    def run_all_searches(self) -> List[Dict]:
        """Run searches across all configured databases."""
        all_records = []
        
        for db in self.databases:
            try:
                records = self.search_database(db)
                all_records.extend(records)
            except Exception as e:
                print(f"Error searching {db}: {e}")
        
        return all_records
    
    def print_summary(self) -> None:
        """Print a summary of search results."""
        print("\n" + "=" * 60)
        print("SEARCH RESULTS SUMMARY")
        print("=" * 60)
        print(f"Search Date: {self.results_summary['search_date']}")
        print(f"Date Range: {self.start_year}-{self.end_year}")
        print()
        
        total_retrieved = 0
        for db, counts in self.results_summary["databases"].items():
            print(f"{db}:")
            print(f"  - Total available: {counts['total_available']}")
            print(f"  - Retrieved: {counts['retrieved']}")
            total_retrieved += counts['retrieved']
        
        print()
        print(f"TOTAL RECORDS (before deduplication): {total_retrieved}")
        print("=" * 60)
    
    def export_results(self, records: List[Dict], output_dir: Path = None,
                       formats: List[str] = None) -> None:
        """
        Export results in multiple formats.
        
        Args:
            records: List of search result records
            output_dir: Output directory path
            formats: List of formats to export ("csv", "ris", "bibtex", "json")
        """
        output_dir = output_dir or Path("./search_results")
        output_dir.mkdir(exist_ok=True)
        formats = formats or ["csv", "ris", "bibtex", "json"]
        
        base_name = "search_results_combined"
        
        if "csv" in formats:
            export_to_csv(records, output_dir / f"{base_name}.csv")
        
        if "ris" in formats:
            export_to_ris(records, output_dir / f"{base_name}.ris")
        
        if "bibtex" in formats:
            export_to_bibtex(records, output_dir / f"{base_name}.bib")
        
        if "json" in formats:
            with open(output_dir / f"{base_name}.json", "w") as f:
                json.dump(records, f, indent=2)
            print(f"Exported {len(records)} records to {output_dir / f'{base_name}.json'}")
        
        # Save summary
        with open(output_dir / "search_summary.json", "w") as f:
            json.dump(self.results_summary, f, indent=2)
        print(f"Saved search summary to {output_dir / 'search_summary.json'}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_available_databases() -> Dict[str, str]:
    """Check which databases can be searched with current dependencies."""
    available = {
        "OpenAlex": "✓ API (no key required)",
        "Semantic Scholar": "✓ API (no key required)",
        "Europe PMC": "✓ API (no key required)",
        "arXiv": "✓ API" if AVAILABLE_DEPS.get("arxiv") else "✗ needs: pip install arxiv",
        "PubMed": "✓ API" if AVAILABLE_DEPS.get("entrez") else "✗ needs: pip install biopython",
        "CORE": "✓ API (requires free key from core.ac.uk)",
        "Google Scholar": "⚠ Manual only (use Publish or Perish)",
        "BASE": "⚠ Manual only",
        "Web of Science": "⚠ Requires institutional access",
        "Scopus": "⚠ Requires institutional access",
    }
    return available


def print_dependency_status():
    """Print status of optional dependencies."""
    print("Optional Dependencies:")
    print(f"  arxiv: {'✓ installed' if AVAILABLE_DEPS.get('arxiv') else '✗ pip install arxiv'}")
    print(f"  biopython: {'✓ installed' if AVAILABLE_DEPS.get('entrez') else '✗ pip install biopython'}")
    print()
    print("Database Support:")
    for db, status in check_available_databases().items():
        print(f"  {db}: {status}")
