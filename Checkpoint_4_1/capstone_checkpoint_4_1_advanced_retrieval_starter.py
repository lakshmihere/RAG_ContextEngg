r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# file_name: trial_cap4_final_3.py

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod
import networkx as nx
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
# LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-coder-6.7b-instruct")
LLM_MODEL = os.getenv("LLM_MODEL","meta-llama-3.1-8b-instruct")
WIKIPEDIA_DIR = Path(os.getenv("WIKIPEDIA_TEXT_DIR", "Wikipedia_10_text"))
TEXT_DIR = WIKIPEDIA_DIR
CHROMA_DIR = os.getenv("WIKIPEDIA_CHROMA_DIR", "Wikipedia_chroma_topics_full")
TOPICS_METADATA_PATH = Path(os.getenv("WIKIPEDIA_TOPICS_METADATA", "wikipedia_topics_10.json"))
LOG_PATH = Path.cwd() / "checkpoint_4_1_advanced_retrieval_final.log"
NUM_RETRIEVED = int(os.getenv("NUM_RETRIEVED", "6"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.5"))
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "0.5"))
TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.0"))
MIN_TAGGED_CANDIDATES = int(os.getenv("MIN_TAGGED_CANDIDATES", "1"))
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "hybrid_tagged").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DECOMPOSITION_MODEL = os.getenv("DECOMPOSITION_MODEL","deepseek-coder-6.7b-instruct")
ANSWER_MODEL = os.getenv("ANSWER_MODEL","meta-llama-3.1-8b-instruct")

CATEGORY_LABELS: dict[str, str] = {
    "person": "a person or biography",
    "place_geography": "a place, city, country, region, or geographic feature",
    "building_landmark": "a building, landmark, monument, or structure",
    "organization_company": "a company, organization, institution, or business",
    "politics_government": "politics, government, a political party, or public administration",
    "arts_entertainment": "arts, entertainment, film, television, music, or literature",
    "sports": "sports, an athlete, team, league, or competition",
    "education": "education, a school, college, university, or academic institution",
    "science_technology": "science, engineering, computing, medicine, or technology",
    "history_event": "a historical event, war, battle, disaster, incident, or movement",
    "chronology_year": "a calendar year or chronological overview of events, births, and deaths",
    "transportation": "transportation, aviation, rail, roads, vehicles, or transit",
    "nature_biology": "nature, an animal, plant, species, ecology, or biology",
    "other": "another topic not covered by the listed categories",
}

CATEGORY_QUERY_HINTS: dict[str, tuple[str, ...]] = {
    "person": (
        "who was", "who is", "biography", "born", "died", "career", "leader",
        "prime minister", "president", "founder", "co-founder", "actor", "author",
    ),
    "place_geography": (
        "where is", "where was", "located", "city", "country", "region", "village",
        "geography", "population",
    ),
    "building_landmark": (
        "building", "landmark", "tower", "monument", "skyscraper", "bridge", "palace",
        "completed", "opened",
    ),
    "organization_company": (
        "company", "organization", "business", "corporation", "owned by", "founded",
        "headquartered", "headquarters",
    ),
    "politics_government": (
        "government", "political", "election", "parliament", "party", "administration",
        "prime minister", "president",
    ),
    "arts_entertainment": (
        "film", "movie", "album", "song", "television", "novel", "music", "actor",
        "actress", "director",
    ),
    "sports": (
        "sport", "football", "soccer", "basketball", "baseball", "tennis", "athlete",
        "team", "league", "championship",
    ),
    "education": (
        "university", "college", "school", "education", "academic",
    ),
    "science_technology": (
        "science", "technology", "engineering", "computer", "software", "medicine",
        "chemical", "physics", "biology", "algorithm",
    ),
    "history_event": (
        "war", "battle", "historical event", "disaster", "incident", "movement",
        "conflict", "revolution",
    ),
    "chronology_year": (
        "year", "events in", "births in", "deaths in", "century",
    ),
    "transportation": (
        "airline", "airport", "aircraft", "aviation", "rail", "railway", "train",
        "road", "vehicle", "transit",
    ),
    "nature_biology": (
        "species", "animal", "plant", "bird", "mammal", "fish", "ecology", "habitat",
    ),
}


# # === SET THIS to the scenario you chose in Checkpoint 1.1 ===

SCENARIO = "wikipedia"   

# # DECOMPOSE_SYSTEM = (
# #     "You are a query decomposition assistant for a document retrieval system. "
# #     "Break the user's question into 2-4 focused sub-queries that together cover "
# #     "everything needed to answer it. Each sub-query should target a distinct aspect. "
# #     'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
# # )

def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    """
    Decompose a Wikipedia question into focused,
    independently retrievable sub-queries.

    Simple questions may remain as one query.
    Complex, comparative, or multi-part questions should
    normally produce 2-4 sub-queries.
    """

    system_prompt = """
You are a query decomposition component for a Wikipedia
retrieval-augmented generation system.

Your task is NOT to answer the question.

Your task is to convert the user's question into focused
retrieval queries.

RULES:

1. Each sub-query must represent ONE distinct information need.

2. Each sub-query must be independently useful for retrieving
   evidence from Wikipedia articles.

3. Preserve important named entities exactly.

4. For comparison questions:
   create a retrieval query for EACH entity separately.

5. For multi-part questions:
   create separate queries for the major requested facts.

6. For relationship or multi-hop questions:
   retrieve evidence about each entity/event separately before
   attempting to establish the relationship.

7. Do NOT merely paraphrase the original question when it
   contains multiple information needs.

8. Simple single-fact questions may remain as ONE query.

9. Produce between 1 and 4 sub-queries.

Return ONLY the sub-queries, one per line.
Do not number them.
Do not answer the question.
"""

    user_prompt = f"""
Wikipedia question:

{query}

Generate the retrieval sub-queries.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    raw = response.content.strip()

    sub_queries = []

    for line in raw.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove numbering/bullets if model adds them
        line = re.sub(
            r"^[\-•*\d\.\)\s]+",
            "",
            line
        ).strip()

        if line:
            sub_queries.append(line)

    # Safety fallback
    if not sub_queries:
        return [query]

    # ----------------------------------------------------------
    # DECOMPOSITION QUALITY CHECK
    # ----------------------------------------------------------

    complex_markers = [
        " compare ",
        " versus ",
        " vs ",
        " relationship ",
        " connected ",
        " difference ",
        " differences ",
        " similarity ",
        " similarities ",
        " relationship between ",
        " connection between ",
    ]

    query_lower = query.lower()

    looks_complex = any(
        marker in query_lower
        for marker in complex_markers
    )

    # If a complex question was merely returned as one query,
    # ask the model once more with stronger instructions.
    if looks_complex and len(sub_queries) == 1:

        retry_prompt = f"""
The following Wikipedia question contains multiple information needs:

{query}

Your previous output did not sufficiently decompose the question.

Break it into 2 to 4 INDEPENDENT Wikipedia retrieval queries.

Rules:

- For comparisons, create a retrieval query for each named entity.
- For multi-part questions, separate the requested facts.
- For multi-hop questions, retrieve evidence for each step separately.
- Preserve the important named entities.
- Do NOT answer the question.
- Do NOT simply repeat or paraphrase the complete original question.

Return ONLY one retrieval query per line.
"""

        retry_response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=retry_prompt),
        ])

        retry_queries = []

        for line in retry_response.content.strip().splitlines():

            line = re.sub(
                r"^[\-•*\d\.\)\s]+",
                "",
                line.strip()
            ).strip()

            if line:
                retry_queries.append(line)

        if len(retry_queries) >= 2:
            sub_queries = retry_queries

    return sub_queries[:4]



ANSWER_SYSTEM = """
You are a grounded question-answering system.

Answer ONLY from the supplied evidence.

Rules:
1. Do not use outside knowledge.
2. Do not infer facts that are not explicitly supported.
3. Do not invent dates, names, organizations, relationships,
   classifications, quotations, or historical events.
4. If evidence conflicts, describe the conflict.
5. If the supplied evidence is insufficient, say:
   "The retrieved evidence is insufficient to answer this question."
6. For comparisons, discuss each entity only from its corresponding evidence.
7. Never claim that you cannot answer because the topic is outside
   programming or your expertise.
8. When possible, identify which source supports each factual claim.
"""


def load_wikipedia_corpus(text_dir: Path) -> list[dict]:
    """Load the pre-converted Wikipedia .txt corpus recursively.
    The TXT files are assumed to have been produced from the original HTML corpus
    (for example with html2text). Keeping the relative path as the document ID
    prevents filename collisions when the corpus contains subdirectories.
    """
    text_files = sorted(text_dir.rglob("*.txt"))
    print(f"Found {len(text_files)} Wikipedia TXT files.")
    documents: list[dict] = []
    for file_path in text_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                print(f"WARNING: Empty document: {file_path}")
                continue
            relative_path = file_path.relative_to(text_dir)
            document_id = relative_path.as_posix()
            documents.append({"id": document_id, "text": text})
        except Exception as exc:
            print(f"WARNING: Could not load {file_path}: {exc}")
    print(f"Successfully loaded {len(documents)} Wikipedia text documents.")
    for doc in documents[:10]:
        print(f"{doc['id']:<45} {len(doc['text']):>8} characters")
    if len(documents) > 10:
        print(f"... plus {len(documents) - 10} additional documents.")
    if not documents:
        raise RuntimeError(
            f"No Wikipedia TXT files were loaded from {text_dir.resolve()}. "
            "Check WIKIPEDIA_TEXT_DIR / TEXT_DIR."
        )
    return documents


# %%
def check_lm_studio(model_name: str | None = None) -> list[str]:
    """Verify LM Studio is reachable and print the model IDs it exposes."""
    import urllib.request

    selected = model_name or LLM_MODEL
    url = f"{LM_STUDIO_BASE_URL.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to LM Studio at {LM_STUDIO_BASE_URL}. "
            "Start the LM Studio local server and load a model. "
            f"Original error: {exc}"
        ) from exc

    models = [item.get("id") for item in data.get("data", []) if item.get("id")]
    print(f"LM Studio reachable at {LM_STUDIO_BASE_URL}")
    print("Available LM Studio model(s): " + (", ".join(models) if models else "none reported"))
    if models and selected not in models:
        raise RuntimeError(
            f"Requested LM Studio model {selected!r} is not available. "
            f"Choose one of: {', '.join(models)}"
        )
    print(f"Selected LM Studio model: {selected}")
    return models


def make_llm(model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or LLM_MODEL,
        temperature=TEMPERATURE,
        api_key="lm-studio",
        base_url=LM_STUDIO_BASE_URL,
    )


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def keyword_score(query: str, text: str) -> float:
    """A tiny, dependency-free relevance score: shared word count. This stands in for
    your real hybrid retriever, so the retrieval step needs no extra dependencies. The
    demo uses the selected local LM Studio model for LLM calls."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return float(len(q & t))


ACTIVE_RETRIEVER = None

def baseline_retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Compatibility adapter used by the Lab 4.1/4.2 code.

    It now delegates to the selected Checkpoint 2.1/3.1 retriever instead of
    whole-document keyword overlap. Returned scores are higher-is-better.
    """
    if ACTIVE_RETRIEVER is None:
        raise RuntimeError("Retriever is not initialized. Call initialize_retriever() first.")
    results = ACTIVE_RETRIEVER.getTopK(query, k)
    return [(doc_id, float(score)) for doc_id, _content, score in results]


# Load the Wikipedia corpus once and build an ID lookup.
SAMPLE_DOCS = load_wikipedia_corpus(TEXT_DIR)
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}

# Attach topic metadata when available. Graph fields always exist so graph
# construction cannot fail simply because metadata is missing.
def attach_graph_metadata(docs: list[dict]) -> list[dict]:
    metadata = {}
    if TOPICS_METADATA_PATH.exists():
        try:
            loaded = json.loads(TOPICS_METADATA_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except Exception as exc:
            print(f"WARNING: Could not load topic metadata: {exc}")

    known_ids = {d["id"] for d in docs}
    known_stems = {Path(d["id"]).stem: d["id"] for d in docs}

    for d in docs:
        item = metadata.get(d["id"], metadata.get(Path(d["id"]).stem, {}))
        if isinstance(item, str):
            topics = [item]
            links = []
        elif isinstance(item, dict):
            raw_topics = item.get("topics", item.get("categories", item.get("category", [])))
            topics = [raw_topics] if isinstance(raw_topics, str) else list(raw_topics or [])
            raw_links = item.get("links", item.get("linked_articles", []))
            links = list(raw_links or []) if not isinstance(raw_links, str) else [raw_links]
        else:
            topics, links = [], []

        normalized_links = []
        for link in links:
            candidate = str(link).replace(".html", ".txt")
            if candidate in known_ids:
                normalized_links.append(candidate)
            elif Path(candidate).stem in known_stems:
                normalized_links.append(known_stems[Path(candidate).stem])

        d["topics"] = [str(t) for t in topics if str(t).strip()]

        if isinstance(item, dict):
            category = str(item.get("category", item.get("broad_category", "other"))).strip() or "other"
        elif isinstance(item, str) and item in CATEGORY_LABELS:
            category = item
        else:
            category = "other"
        # if isinstance(item, dict):
        #     category = str(item.get("category", "other")).strip() or "other"
        # elif isinstance(item, str) and item in CATEGORY_LABELS:
        #     category = item
        # else:
        #     category = "other"
        d["category"] = category if category in CATEGORY_LABELS else "other"
        d["links"] = normalized_links
    return docs

def attach_graph_metadata(docs: list[dict]) -> list[dict]:
    metadata_by_id: dict[str, dict] = {}
    if not TOPICS_METADATA_PATH.exists():
        print(
            f"WARNING: Topic metadata file not found: "
            f"{TOPICS_METADATA_PATH.resolve()}"
        )
    else:
        try:
            loaded = json.loads(
                TOPICS_METADATA_PATH.read_text(
                    encoding="utf-8"
                )
            )
            if isinstance(loaded, list):

                for item in loaded:
                    if not isinstance(item, dict):
                        continue

                    # Try the common filename/title fields.
                    source = (
                        item.get("source")
                        or item.get("filename")
                        or item.get("file")
                        or item.get("id")
                        or item.get("article")
                        or item.get("title")
                    )

                    if not source:
                        continue

                    source = str(source).strip()

                    # Convert HTML filename if necessary.
                    source = source.replace(".html", ".txt")

                    # Store several lookup forms.
                    metadata_by_id[source] = item
                    metadata_by_id[Path(source).name] = item
                    metadata_by_id[Path(source).stem] = item

            elif isinstance(loaded, dict):

                # Some JSON files may wrap records in a field.
                records = None

                for key in (
                    "documents",
                    "articles",
                    "records",
                    "metadata",
                    "items",
                ):
                    if isinstance(loaded.get(key), list):
                        records = loaded[key]
                        break

                if records is not None:

                    for item in records:
                        if not isinstance(item, dict):
                            continue

                        source = (
                            item.get("source")
                            or item.get("filename")
                            or item.get("file")
                            or item.get("id")
                            or item.get("article")
                            or item.get("title")
                        )

                        if not source:
                            continue

                        source = str(source).strip()
                        source = source.replace(".html", ".txt")

                        metadata_by_id[source] = item
                        metadata_by_id[Path(source).name] = item
                        metadata_by_id[Path(source).stem] = item

                else:
                    # Already keyed by article/file name.
                    for key, value in loaded.items():
                        if not isinstance(value, dict):
                            continue

                        normalized_key = (
                            str(key)
                            .strip()
                            .replace(".html", ".txt")
                        )

                        metadata_by_id[normalized_key] = value
                        metadata_by_id[Path(normalized_key).name] = value
                        metadata_by_id[Path(normalized_key).stem] = value

            print(
                f"Loaded topic metadata lookup with "
                f"{len(metadata_by_id)} keys from "
                f"{TOPICS_METADATA_PATH}"
            )

        except Exception as exc:
            print(
                f"WARNING: Could not load topic metadata: {exc}"
            )
    known_ids = {d["id"] for d in docs}
    known_stems = {
        Path(d["id"]).stem: d["id"]
        for d in docs
    }
    matched = 0
    for d in docs:
        doc_id = d["id"]
        doc_name = Path(doc_id).name
        doc_stem = Path(doc_id).stem
        item = (
            metadata_by_id.get(doc_id)
            or metadata_by_id.get(doc_name)
            or metadata_by_id.get(doc_stem)
            or {}
        )
        if item:
            matched += 1
        if isinstance(item, dict):
            category = str(
                item.get(
                    "category",
                    item.get(
                        "broad_category",
                        "other"
                    )
                )
            ).strip() or "other"

        else:
            category = "other"

        if category not in CATEGORY_LABELS:
            category = "other"

        d["category"] = category
        raw_topics = []
        if isinstance(item, dict):

            raw_topics = item.get(
                "topics",
                item.get(
                    "categories",
                    []
                )
            )
        if isinstance(raw_topics, str):
            topics = [
                x.strip()
                for x in raw_topics.split("|")
                if x.strip()
            ]
        elif isinstance(raw_topics, (list, tuple, set)):
            topics = [
                str(x).strip()
                for x in raw_topics
                if str(x).strip()
            ]
        else:
            topics = []

        d["topics"] = topics
        raw_links = []
        if isinstance(item, dict):
            raw_links = item.get(
                "links",
                item.get(
                    "linked_articles",
                    []
                )
            )

        if isinstance(raw_links, str):
            raw_links = [raw_links]
        normalized_links = []
        for link in raw_links or []:
            candidate = (
                str(link)
                .strip()
                .replace(".html", ".txt")
            )

            if candidate in known_ids:
                normalized_links.append(candidate)

            elif Path(candidate).stem in known_stems:
                normalized_links.append(
                    known_stems[Path(candidate).stem]
                )

        d["links"] = normalized_links

    print(
        f"Metadata matched to "
        f"{matched}/{len(docs)} Wikipedia documents."
    )

    return docs

SAMPLE_DOCS = attach_graph_metadata(SAMPLE_DOCS)
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}
print("\nCATEGORY METADATA CHECK")
for doc in SAMPLE_DOCS:
    print(
        doc["id"],
        "category=", doc.get("category"),
        "topics=", doc.get("topics")
    )

# ============================================================================
# RETRIEVAL HELPERS
# ============================================================================

_EMBEDDINGS = None

def get_embeddings():
    """Create the local Hugging Face embedding model once and reuse it."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDINGS

def infer_query_categories(query: str, documents: list[dict]) -> list[str]:
    """Infer likely metadata categories from explicit query hints."""
    q = query.lower()
    scored = []
    for category, hints in CATEGORY_QUERY_HINTS.items():
        score = sum(1 for hint in hints if hint in q)
        if score:
            scored.append((category, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [category for category, _ in scored[:2]]

def filter_documents_by_categories(documents: list[dict], categories: list[str]) -> list[dict]:
    if not categories:
        return list(documents)
    wanted = set(categories)
    return [d for d in documents if d.get("category", "other") in wanted]

# ============================================================================
# STEP 6 — VECTOR DATABASE
# ============================================================================


def build_or_load_db(documents: list[dict], chroma_dir: str = CHROMA_DIR) -> Chroma:
    """
    Build/load persistent Chroma.

    Category is stored in Chroma metadata so Tagged Hybrid can apply metadata
    filters to semantic retrieval as well as BM25 retrieval.

    IMPORTANT: If an older Chroma directory was built before category metadata
    was added, delete that directory once so it can be rebuilt with categories.
    """
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(),
        )

    print("Building vector DB (first run — embedding Wikipedia articles)...")
    docs = [
        Document(
            page_content=d["text"],
            metadata={
                "source": d["id"],
                "category": d.get("category", "other"),
                "topics": " | ".join(d.get("topics", [])),
            },
        )
        for d in documents
    ]

    db = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=chroma_dir,
    )
    print(f"Indexed {len(docs)} Wikipedia articles into {chroma_dir}/")
    return db


# ============================================================================
# STEP 7 — RETRIEVERS
# ============================================================================

_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    ]


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max normalize to [0, 1]; optionally invert distance scores."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    normalized = [(score - lo) / (hi - lo) for score in scores]
    return [1.0 - x for x in normalized] if invert else normalized


RetrievalResult = tuple[str, str, float]


class BaseRetriever(ABC):
    """Base retriever with a shared evidence -> answering-LLM path."""

    def __init__(self, documents: list[dict], num_retrieved: int = NUM_RETRIEVED):
        self.documents = documents
        self.num_retrieved = num_retrieved
        self._llm = make_llm()

    @abstractmethod
    def getTopK(self, query: str, k: int) -> list[RetrievalResult]:
        ...

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, self.num_retrieved)
        return self._format_context(results, query)

    @staticmethod
    def _best_article_snippet(
        content: str,
        query: str,
        *,
        window_chars: int = 1200,
        overlap_chars: int = 200,
        windows_per_article: int = 2,
    ) -> str:
        """Select small query-relevant windows from a retrieved full article.

        The retrievers rank articles, but sending entire 20k-200k character
        Wikipedia pages to an 8K-context local model causes n_keep/n_ctx errors.
        This keeps retrieval article-level while making generation context-sized.
        """
        if len(content) <= window_chars:
            return content

        q_tokens = set(tokenize(query))
        step = max(1, window_chars - overlap_chars)
        windows = []
        for start in range(0, len(content), step):
            piece = content[start:start + window_chars]
            if not piece:
                continue
            p_tokens = tokenize(piece)
            overlap = sum(1 for token in p_tokens if token in q_tokens)
            distinct = len(set(p_tokens) & q_tokens)
            # Prefer early text on ties because Wikipedia leads usually contain
            # the defining facts needed by these checkpoint questions.
            score = (distinct * 10) + overlap - (start / max(len(content), 1))
            windows.append((score, start, piece))
            if start + window_chars >= len(content):
                break

        windows.sort(key=lambda x: x[0], reverse=True)
        selected = sorted(windows[:windows_per_article], key=lambda x: x[1])
        return "\n...\n".join(piece for _, _, piece in selected)

    @classmethod
    def _format_context(
        cls,
        results: list[RetrievalResult],
        query: str,
        max_total_chars: int = 16000,
    ) -> str:
        blocks = []
        used = 0
        for name, content, _ in results:
            snippet = cls._best_article_snippet(content, query)
            block = f"[{name}]\n{snippet}"
            remaining = max_total_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining]
            blocks.append(block)
            used += len(block) + 7
        return "\n\n---\n\n".join(blocks)

    def get_sources(self, query: str) -> list[str]:
        return [name for name, _, _ in self.getTopK(query, self.num_retrieved)]

    def query_with_trace(self, question: str) -> dict:
        """Retrieve once, then use compact evidence from those same results."""
        results = self.getTopK(question, self.num_retrieved)
        context = self._format_context(results, question)
        sources = [name for name, _, _ in results]

        if not results:
            return {
                "answer": "The retriever returned no Wikipedia evidence for this question.",
                "sources": [],
                "evidence": "",
            }

        # messages = [
        #     SystemMessage(content=ANSWER_SYSTEM),
        #     HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}"),
        # ]
        # response = self._llm.invoke(messages)
        messages = [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}"),
        ]

        print(
            f"\n  [answer generation] sending to LM Studio "
            f"| context chars={len(context):,}"
        )
        response = self._llm.invoke(messages)
        print("  [answer generation] LM Studio response received")

        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": str(answer), "sources": sources, "evidence": context}

    def query(self, question: str) -> str:
        return self.query_with_trace(question)["answer"]


class BM25Retriever(BaseRetriever):
    """BM25 keyword retrieval over the full Wikipedia corpus."""

    def __init__(self, documents: list[dict], num_retrieved: int = NUM_RETRIEVED):
        super().__init__(documents, num_retrieved)
        self._names = [d["id"] for d in documents]
        self._contents = [d["text"] for d in documents]
        self._bm25 = BM25Okapi([tokenize(text) for text in self._contents])
        print(f"BM25 retriever ready over {len(self._names)} Wikipedia articles.")

    def getTopK(self, query: str, k: int) -> list[RetrievalResult]:
        k = min(k, len(self._names))
        scores = self._bm25.get_scores(tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            (self._names[i], self._contents[i], float(scores[i]))
            for i in top_indices
        ]


class VectorRetriever(BaseRetriever):
    """Semantic vector-only retrieval using Chroma."""

    def __init__(
        self,
        db: Chroma,
        documents: list[dict],
        num_retrieved: int = NUM_RETRIEVED,
    ):
        super().__init__(documents, num_retrieved)
        self._db = db
        print("Vector retriever ready.")

    def getTopK(self, query: str, k: int) -> list[RetrievalResult]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [
            (
                doc.metadata.get("source", "unknown"),
                doc.page_content,
                float(distance),
            )
            for doc, distance in results
        ]


class HybridRetriever(BaseRetriever):
    """Weighted fusion of full-corpus BM25 and full-corpus vector retrieval."""

    def __init__(
        self,
        db: Chroma,
        documents: list[dict],
        num_retrieved: int = NUM_RETRIEVED,
    ):
        super().__init__(documents, num_retrieved)
        self._db = db
        self._names = [d["id"] for d in documents]
        self._contents = [d["text"] for d in documents]
        self._bm25 = BM25Okapi([tokenize(text) for text in self._contents])
        print(
            f"Hybrid retriever ready over {len(self._names)} Wikipedia articles "
            f"(BM25 weight={WEIGHT_BM25}, vector weight={WEIGHT_VECTOR})."
        )

    def _bm25_topk(self, query: str, k: int) -> list[RetrievalResult]:
        k = min(k, len(self._names))
        scores = self._bm25.get_scores(tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            (self._names[i], self._contents[i], float(scores[i]))
            for i in top_indices
        ]

    def _vector_topk(self, query: str, k: int) -> list[RetrievalResult]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [
            (
                doc.metadata.get("source", "unknown"),
                doc.page_content,
                float(distance),
            )
            for doc, distance in results
        ]

    @staticmethod
    def _fuse(
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        k: int,
    ) -> list[RetrievalResult]:
        content_by_name: dict[str, str] = {}
        bm25_norm: dict[str, float] = {}
        vector_norm: dict[str, float] = {}

        if bm25_results:
            values = [score for _, _, score in bm25_results]
            for (name, content, _), score in zip(bm25_results, _normalize(values)):
                content_by_name[name] = content
                bm25_norm[name] = score

        if vector_results:
            distances = [score for _, _, score in vector_results]
            for (name, content, _), score in zip(
                vector_results, _normalize(distances, invert=True)
            ):
                content_by_name[name] = content
                vector_norm[name] = score

        fused = [
            (
                name,
                content,
                WEIGHT_BM25 * bm25_norm.get(name, 0.0)
                + WEIGHT_VECTOR * vector_norm.get(name, 0.0),
            )
            for name, content in content_by_name.items()
        ]
        fused.sort(key=lambda item: item[2], reverse=True)
        return fused[:k]

    def getTopK(self, query: str, k: int) -> list[RetrievalResult]:
        pool_size = min(CANDIDATE_POOL, len(self._names))
        bm25_results = self._bm25_topk(query, pool_size)
        vector_results = self._vector_topk(query, pool_size)
        return self._fuse(bm25_results, vector_results, k)


class TaggedHybridRetriever(HybridRetriever):
    """
    Category-aware hybrid retrieval.

    Query
      -> infer categories
      -> filter candidate documents
      -> BM25 on filtered candidates
      -> vector search with category metadata filters
      -> weighted fusion
      -> top-k evidence

    If routing is uncertain or leaves too few candidates, the retriever falls
    back to ordinary full-corpus hybrid retrieval.
    """

    def __init__(
        self,
        db: Chroma,
        documents: list[dict],
        num_retrieved: int = NUM_RETRIEVED,
    ):
        super().__init__(db=db, documents=documents, num_retrieved=num_retrieved)
        self._doc_by_id = {d["id"]: d for d in documents}
        print("Tagged Hybrid category-aware routing enabled.")

    def _tagged_bm25_topk(
        self,
        query: str,
        candidate_docs: list[dict],
        k: int,
    ) -> list[RetrievalResult]:
        if not candidate_docs:
            return []
        names = [d["id"] for d in candidate_docs]
        contents = [d["text"] for d in candidate_docs]
        bm25 = BM25Okapi([tokenize(text) for text in contents])
        scores = bm25.get_scores(tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: min(k, len(names))]
        return [
            (names[i], contents[i], float(scores[i])) for i in top_indices
        ]

    def _tagged_vector_topk(
        self,
        query: str,
        categories: list[str],
        k: int,
    ) -> list[RetrievalResult]:
        if not categories:
            return self._vector_topk(query, k)

        merged: dict[str, RetrievalResult] = {}
        per_category_k = min(k, CANDIDATE_POOL)

        # Query each category separately. This avoids depending on a particular
        # Chroma $in/$or filter syntax across versions.
        for category in categories:
            try:
                results = self._db.similarity_search_with_score(
                    query,
                    k=per_category_k,
                    filter={"category": category},
                )
            except Exception as exc:
                print(
                    f"WARNING: Chroma category filter failed for {category!r}: {exc}. "
                    "Falling back to post-filtered vector candidates."
                )
                broad = self._db.similarity_search_with_score(
                    query,
                    k=min(CANDIDATE_POOL * 3, max(CANDIDATE_POOL, len(self.documents))),
                )
                results = [
                    (doc, distance)
                    for doc, distance in broad
                    if doc.metadata.get("category") == category
                ][:per_category_k]

            for doc, distance in results:
                name = doc.metadata.get("source", "unknown")
                item = (name, doc.page_content, float(distance))
                # Vector distances are lower-is-better, so keep the smallest.
                if name not in merged or item[2] < merged[name][2]:
                    merged[name] = item

        return sorted(merged.values(), key=lambda x: x[2])[:k]

    def getTopK(self, query: str, k: int) -> list[RetrievalResult]:
        categories = infer_query_categories(query, self.documents)
        candidate_docs = filter_documents_by_categories(self.documents, categories)

        if not categories:
            print("  tagged routing: no confident category -> full-corpus hybrid fallback")
            return super().getTopK(query, k)

        if len(candidate_docs) < MIN_TAGGED_CANDIDATES:
            print(
                f"  tagged routing: {categories} produced only {len(candidate_docs)} "
                "candidate(s) -> full-corpus hybrid fallback"
            )
            return super().getTopK(query, k)

        print(
            f"  tagged routing: categories={categories} | "
            f"candidate articles={len(candidate_docs)}/{len(self.documents)}"
        )

        pool_size = min(CANDIDATE_POOL, len(candidate_docs))
        bm25_results = self._tagged_bm25_topk(query, candidate_docs, pool_size)
        vector_results = self._tagged_vector_topk(query, categories, pool_size)
        return self._fuse(bm25_results, vector_results, k)


def sync_chroma_metadata_to_docs(docs: list[dict], vectorstore: Chroma) -> list[dict]:
    """Copy category/topics stored in Chroma back into the in-memory documents."""
    print("\n" + "=" * 80)
    print("SYNCING CHROMA METADATA -> SAMPLE_DOCS")
    print("=" * 80)

    data = vectorstore.get(include=["metadatas"])
    by_source = {
        str(m.get("source")): m
        for m in data.get("metadatas", [])
        if m and m.get("source")
    }

    updated = 0
    for doc in docs:
        metadata = by_source.get(doc["id"])
        if not metadata:
            doc.setdefault("category", "other")
            doc.setdefault("topics", [])
            doc.setdefault("links", [])
            print(f"WARNING: no Chroma metadata for {doc['id']}")
            continue

        category = str(metadata.get("category", "other")).strip() or "other"
        doc["category"] = category if category in CATEGORY_LABELS else "other"

        raw_topics = metadata.get("topics", "")
        if isinstance(raw_topics, str):
            topics = [x.strip() for x in raw_topics.split("|") if x.strip()]
        elif isinstance(raw_topics, (list, tuple, set)):
            topics = [str(x).strip() for x in raw_topics if str(x).strip()]
        else:
            topics = []

        doc["topics"] = topics
        doc.setdefault("links", [])
        updated += 1
        print(f"{doc['id']:<45} category={doc['category']:<22} topics={len(topics)}")

    print(f"Metadata synchronized for {updated}/{len(docs)} documents.")
    print("=" * 80)
    return docs


def initialize_retriever(mode: str = RETRIEVER_MODE):
        """Build/load retrieval resources once and select the active retriever."""

        global ACTIVE_RETRIEVER

        mode = mode.lower().strip()

        # ------------------------------------------------------------
        # BM25 does not need embeddings or Chroma.
        # ------------------------------------------------------------
        if mode == "bm25":
            ACTIVE_RETRIEVER = BM25Retriever(SAMPLE_DOCS)
            print(f"Active retriever: {mode}")
            return ACTIVE_RETRIEVER

        # ------------------------------------------------------------
        # Build or load ONE Chroma database.
        # ------------------------------------------------------------
        db = build_or_load_db(SAMPLE_DOCS)

        # ------------------------------------------------------------
        # Diagnostic: inspect metadata stored in Chroma.
        # ------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CHROMA METADATA CHECK")
        print("=" * 80)

        sample = db.get(
            limit=10,
            include=["metadatas"],
        )

        for doc_id, metadata in zip(
            sample.get("ids", []),
            sample.get("metadatas", []),
        ):
            print(doc_id, metadata)

        print("=" * 80)

        # ------------------------------------------------------------
        # Synchronize Chroma metadata back into SAMPLE_DOCS.
        # ------------------------------------------------------------
        sync_chroma_metadata_to_docs(
            SAMPLE_DOCS,
            db,
        )

        print("\nCATEGORY METADATA AFTER SYNC")

        for doc in SAMPLE_DOCS:
            print(
                f"{doc['id']:<45} "
                f"category={doc.get('category', 'other'):<22} "
                f"topics={len(doc.get('topics', []))}"
            )

        # ------------------------------------------------------------
        # Select retriever.
        # ------------------------------------------------------------
        if mode == "vector":
            ACTIVE_RETRIEVER = VectorRetriever(
                db,
                SAMPLE_DOCS,
            )

        elif mode == "hybrid":
            ACTIVE_RETRIEVER = HybridRetriever(
                db,
                SAMPLE_DOCS,
            )

        elif mode == "hybrid_tagged":
            ACTIVE_RETRIEVER = TaggedHybridRetriever(
                db,
                SAMPLE_DOCS,
            )

        else:
            raise ValueError(
                f"Unknown RETRIEVER_MODE={mode!r}. "
                "Choose bm25, vector, hybrid, or hybrid_tagged."
            )

        print(f"Active retriever: {mode}")

        return ACTIVE_RETRIEVER

# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def multistep_retrieve(
    sub_queries: list[str],
    k: int = 3
) -> list[str]:

    score_map: dict[str, float] = {}

    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, 3 * k):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score

    ranked = sorted(
        score_map.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    return [doc_id for doc_id, _ in ranked[:k]]



def build_graph(docs: list[dict]) -> nx.DiGraph:
    """
    Build a graph with article, category, topic,
    and optional direct-link edges.
    """
    G = nx.DiGraph()
    known_ids = {d["id"] for d in docs}
    for d in docs:
        # ----------------------------------------------------------
        # ARTICLE NODE
        # ----------------------------------------------------------
        doc_node = f"doc:{d['id']}"
        category = str(d.get("category", "other")).strip() or "other"
        G.add_node(doc_node, node_type="article", category=category,)
        # ----------------------------------------------------------
        # CATEGORY NODE + EDGES
        # ----------------------------------------------------------
        if category != "other":
            category_node = f"category:{category}"
            G.add_node(category_node, node_type="category", label=category,)
            G.add_edge(doc_node, category_node, edge_type="belongs_to_category",)
            G.add_edge(category_node, doc_node, edge_type="category_contains", )

        # ----------------------------------------------------------
        # TOPIC NODES + EDGES
        # ----------------------------------------------------------
        for topic in d.get("topics", []):
            topic = str(topic).strip()
            if not topic:
                continue
            topic_node = f"topic:{topic.lower()}"
            G.add_node(topic_node, node_type="topic", label=topic,)
            G.add_edge(doc_node, topic_node, edge_type="relates_to",)
        # ----------------------------------------------------------
        # OPTIONAL ARTICLE-TO-ARTICLE LINKS
        # ----------------------------------------------------------
        for other in d.get("links", []):

            if other in known_ids and other != d["id"]:

                G.add_edge(
                    doc_node,
                    f"doc:{other}",
                    edge_type="links_to",
                )
    # --------------------------------------------------------------
    # GRAPH DIAGNOSTICS
    # --------------------------------------------------------------
    from collections import Counter
    print("\n" + "=" * 80)
    print("GRAPH CREATED")
    print("=" * 80)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Node types:", Counter(data.get("node_type", "unknown") for _, data in G.nodes(data=True)),)
    print("Edge types:", Counter(data.get("edge_type", "unknown") for _, _, data in G.edges(data=True)),)
    print("=" * 80)
    return G


def graph_retrieve(graph: nx.DiGraph, query: str, k: int = 3) -> dict[str, str]:
    """Return {doc_id: source_label}: baseline seeds plus their graph neighbors."""
    union: dict[str, str] = {}
    for doc_id, _ in baseline_retrieve(query, k):
        union[doc_id] = "seed"
    for seed in list(union):
        node = f"doc:{seed}"
        if not graph.has_node(node):
            continue
        for _, target, ed in graph.out_edges(node, data=True):
            if ed.get("edge_type") == "links_to":
                union.setdefault(target.removeprefix("doc:"), "linked")
            elif ed.get("edge_type") == "relates_to":
                for sib, _, ed2 in graph.in_edges(target, data=True):  # Source docs on this topic
                    if ed2.get("edge_type") == "relates_to":
                        union.setdefault(sib.removeprefix("doc:"), "topic")
    return union

#def advanced_retrieve(llm: ChatOpenAI, graph: nx.DiGraph, query: str, k: int = 3) -> list[str]:
def advanced_retrieve(graph: nx.DiGraph, sub_queries: list[str], k: int = 3) -> list[str]:

    """Combine Lab 4.1 query decomposition with Lab 4.2 graph expansion.

    1. Decompose a complex question into focused sub-queries.
    2. Retrieve seed documents for every sub-query.
    3. Expand each seed through document links and shared-topic neighbors.
    4. Merge/deduplicate evidence and rank direct seeds ahead of graph-only context.

    NOTE: baseline_retrieve is a compatibility adapter over the active BM25/vector/
    hybrid/hybrid-tagged retriever selected by RETRIEVER_MODE.
    """
    #sub_queries = decompose_query(llm, query)
    #print(f"  advanced decomposition: {sub_queries}")

    scores: dict[str, float] = {}
    source_types: dict[str, set[str]] = {}

    for sq in sub_queries:
        seeds = baseline_retrieve(sq, 3 * k)
        for doc_id, score in seeds:
            scores[doc_id] = scores.get(doc_id, 0.0) + score
            source_types.setdefault(doc_id, set()).add("seed")

            node = f"doc:{doc_id}"
            if not graph.has_node(node):
                continue

            for _, target, edge_data in graph.out_edges(node, data=True):
                edge_type = edge_data.get("edge_type")
                if edge_type == "links_to" and target.startswith("doc:"):
                    linked_id = target.removeprefix("doc:")
                    if linked_id in DOC_BY_ID:
                        scores[linked_id] = scores.get(linked_id, 0.0) + 0.25 * max(score, 1.0)
                        source_types.setdefault(linked_id, set()).add("linked")

                elif edge_type == "relates_to" and target.startswith("topic:"):
                    for sibling, _, sibling_edge in graph.in_edges(target, data=True):
                        if sibling_edge.get("edge_type") != "relates_to" or not sibling.startswith("doc:"):
                            continue
                        sibling_id = sibling.removeprefix("doc:")
                        if sibling_id in DOC_BY_ID and sibling_id != doc_id:
                            scores[sibling_id] = scores.get(sibling_id, 0.0) + 0.10 * max(score, 1.0)
                            source_types.setdefault(sibling_id, set()).add("topic")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [doc_id for doc_id, _ in ranked[:k]]
    print("  advanced evidence:")
    for doc_id in selected:
        print(f"    {doc_id}: score={scores[doc_id]:.3f}, sources={sorted(source_types.get(doc_id, set()))}")
    return selected


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

def my_advanced_plan() -> dict[str, Any]:
    """
    Return the advanced-retrieval plan for the Wikipedia Retrieval Engine.

    The system combines two advanced retrieval techniques:

    1. Query decomposition:
       Complex or multi-part questions are divided into focused sub-queries.
       Each sub-query is retrieved independently using the existing
       hybrid/tagged-hybrid retriever.

    2. Graph-based retrieval:
       Retrieved Wikipedia articles are used as seed nodes in a graph.
       The graph represents relationships between articles, categories,
       and Wikipedia topics. Graph traversal can therefore add related
       evidence that may not appear among the highest-ranked results
       from a single retrieval query.
    """

    return {
        # Both techniques are implemented:
        # query decomposition + graph-based retrieval.
        "technique": "both",

        # These correspond to the node types actually created
        # by build_graph().
        "node_types": [
            "article",
            "category",
            "topic",
        ],

        # These correspond to the relationships actually created
        # by build_graph().
        "edge_types": [
            "belongs_to_category",
            "category_contains",
            "relates_to",
        ],

        # These questions require evidence from multiple facts,
        # entities, or parts of an article and are therefore useful
        # tests for advanced retrieval.
        "test_queries": [
            (
                "Compare the leadership roles and historical significance "
                "of Margaret Thatcher and Adolf Hitler as described in their "
                "respective Wikipedia articles. Support the comparison with "
                "a direct quote from each article."
            ),
            (
                "According to the Wikipedia article on Emirates, when was "
                "the airline founded, where is it based, and who owns it?"
            ),
            (
                "According to the Wikipedia article on Margaret Thatcher, "
                "where did she study, what degree did she receive, and when "
                "did she graduate?"
            ),
        ],

        # Why these techniques are appropriate for this corpus.
        "rationale": (
            "The Wikipedia corpus contains complex multi-part questions and "
            "relationships among articles, categories, and topics. Query "
            "decomposition separates complex questions into focused retrieval "
            "needs, while graph-based retrieval can expand the initial hybrid "
            "retrieval results through shared Wikipedia metadata. Together, "
            "these techniques are intended to improve evidence coverage for "
            "questions that are difficult to answer with a single retrieval pass."
        ),
    }


def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: list[str], chunks_per_doc: int = 3, chunk_size: int = 1800, max_context_chars: int = 18000,) -> str:
    """
    Build a bounded context from the most query-relevant chunks of the retrieved Wikipedia articles.This prevents full Wikipedia articles from exceeding the LLM context window.
    """
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    selected_chunks: list[tuple[float, str, str]] = []
    for doc_id in doc_ids:
        if doc_id not in DOC_BY_ID:
            continue
        text = DOC_BY_ID[doc_id]["text"]
        # Split article into manageable chunks.
        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]
        scored_chunks = []
        for chunk in chunks:
            chunk_terms = set(
                re.findall(r"[a-z0-9]+", chunk.lower())
            )
            score = float(len(query_terms & chunk_terms))
            if score > 0:
                scored_chunks.append((score, chunk))
        # Highest-scoring chunks first.
        scored_chunks.sort(
            key=lambda x: x[0],
            reverse=True
        )

        for score, chunk in scored_chunks[:chunks_per_doc]:
            selected_chunks.append(
                (score, doc_id, chunk)
            )

    # Rank evidence across all retrieved articles.
    selected_chunks.sort(key=lambda x: x[0],reverse=True)
    context_parts = []
    current_chars = 0
    for score, doc_id, chunk in selected_chunks:
        block = (
            f"\n[SOURCE: {doc_id} | relevance={score:.1f}]\n"
            f"{chunk.strip()}\n"
        )
        if current_chars + len(block) > max_context_chars:
            break
        context_parts.append(block)
        current_chars += len(block)
    context = "\n".join(context_parts)
    if not context.strip():
        return (
            "No sufficiently relevant evidence was retrieved "
            "from the selected Wikipedia documents."
        )
    print(
        f"  Answer context: {len(context):,} characters "
        f"from {len(context_parts)} chunks"
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=(f"Documents:\n{context}\n\n" f"Question: {query}")),]
    return llm.invoke(messages).content

TEST_QUERIES = [
    "According to the Wikipedia article on the Empire State Building, when was the building completed and when did it officially open?",
    "According to the Wikipedia article on Hallstatt, where is Hallstatt located and what is it particularly known for?",
    "What role did Steve Jobs play in the development of Apple, according to his Wikipedia article?",
    "Compare the leadership roles and historical significance of Margaret Thatcher and Adolf Hitler as described in their respective Wikipedia articles.",
    "How were Margaret Thatcher and Adolf Hitler connected through the major European political and military events that shaped the periods in which they rose to prominence?",
    "According to the Wikipedia article on Emirates airline, when was Emirates founded, where is it based, and who owns it? Quote the relevant sentence or sentences.",
    "Which Austrian settlement in the corpus is strongly associated with prehistoric salt mining, and where is it located?",
    "What did Margaret Thatcher study at the University of Oxford, and what academic qualification did she receive?",
    "Compare the roles of Queen Victoria and Margaret Thatcher in British history using evidence from their respective Wikipedia articles.",
    "Who was Victoria, what position did she hold, and which country or kingdom did she rule?",
    #"What is Mytilidae, and what type of organisms belong to this biological family?",
    # "According to the Wikipedia article on birds, what major characteristics distinguish birds from other animals?",
    # "Which landmark in the corpus is an Art Deco skyscraper in Manhattan that was completed in the early 1930s?",
    # "Which airline in the corpus was established in Dubai, and what organization owns it?",
    # "What was Steve Jobs's favorite restaurant, according to the Wikipedia evidence available in the corpus?",
    # "According to the corpus, when did Margaret Thatcher become President of the United States?",
    # "Compare the historical significance of the Empire State Building and Emirates airline within their respective domains.",
    # "How did Margaret Thatcher's education and early professional career precede her entry into national politics?",
    # "What were the 85th Academy Awards, and what major event or achievement did the ceremony recognize?",
    # "Compare birds and members of the Mytilidae family in terms of their biological classification and major physical characteristics.",
]

def run_demo() -> None:

    check_lm_studio()

    # ==========================================================
    # LLM INITIALIZATION
    # ==========================================================

    decomposition_llm = make_llm(
        DECOMPOSITION_MODEL
    )

    answer_llm = make_llm(
        ANSWER_MODEL
    )

    print("\nLLM CONFIGURATION")
    print(f"Decomposition model: {DECOMPOSITION_MODEL}")
    print(f"Answer model:        {ANSWER_MODEL}")

    # ==========================================================
    # RETRIEVAL INITIALIZATION
    # ==========================================================

    initialize_retriever()

    # IMPORTANT:
    # Build graph only after initialize_retriever(), because
    # initialize_retriever() synchronizes Chroma metadata into
    # SAMPLE_DOCS.
    graph = build_graph(SAMPLE_DOCS)


    print("\nGRAPH DIAGNOSTICS")
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())

    from collections import Counter

    print(
        "Edge types:",
        Counter(
            data.get("edge_type", "unknown")
            for _, _, data in graph.edges(data=True)
        )
    )

    print("\nDOCUMENT GRAPH METADATA")

    for doc in SAMPLE_DOCS:
        print(
            doc["id"],
            "links=", len(doc.get("links", [])),
            "topics=", len(doc.get("topics", []))
        )

    print("=" * 90)
    print(
        f"Checkpoint 4.1 — Advanced Retrieval Evaluation | "
        f"scenario: {SCENARIO}"
    )
    print(f"Running {len(TEST_QUERIES)} Wikipedia test queries")
    print("=" * 90)


#     DEBUG_QUERIES = [
#     TEST_QUERIES[2],  # Q3 - Steve Jobs
#     TEST_QUERIES[4],  # Q5 - Thatcher / Hitler
# ]

    # for q_num, query in enumerate(DEBUG_QUERIES, start=1):
    for q_num, query in enumerate(TEST_QUERIES, start=1):
        print("\n" + "=" * 90)
        print(f"QUESTION {q_num}")
        print(query)
        print("=" * 90)
        sub_queries = decompose_query(decomposition_llm,query)
        print("\nDECOMPOSITION:")
        for i, sq in enumerate(sub_queries, start=1):
            print(f"  {i}. {sq}")
        # ======================================================
        # 1. BASELINE
        # ======================================================
        base = [
            doc_id
            for doc_id, _ in baseline_retrieve(
                query,
                3
            )
        ]

        print("\nBASELINE:")
        print(base)
        # ======================================================
        # 2. MULTI-STEP
        # ======================================================
        multi = multistep_retrieve(
            sub_queries,
            k=3
        )
        print("\nMULTI-STEP:")
        print(multi)
        # ======================================================
        # 3. GRAPH
        # ======================================================
        graph_hits = graph_retrieve(
            graph,
            query,
            3
        )
        print("\nGRAPH:")
        print(graph_hits)
        # ======================================================
        # 4. COMBINED ADVANCED RETRIEVAL
        # ======================================================

        advanced = advanced_retrieve(
            graph,
            sub_queries,
            k=3
        )

        print("\nCOMBINED:")
        print(advanced)


        # ======================================================
        # 5. FINAL ANSWER
        #
        # Llama 3.1 handles this part instead of DeepSeek-Coder
        # ======================================================

        answer = answer_from_docs(
            answer_llm,
            query,
            advanced
        )

        print("\nADVANCED ANSWER:")
        print(answer)


        # ======================================================
        # 6. LOG RESULT
        # ======================================================

        log_response(
            "COMBINED_ADVANCED",
            query,
            answer
        )
    # ==========================================================
    # ADVANCED RETRIEVAL PLAN
    # ==========================================================
    print("\n" + "=" * 90)
    print("Your advanced-retrieval plan:")
    print(
        json.dumps(
            my_advanced_plan(),
            indent=2
        )
    )
    print("=" * 90)
    print(
        "Done. Completed all Wikipedia test queries."
    )
if __name__ == "__main__":
    run_demo()

