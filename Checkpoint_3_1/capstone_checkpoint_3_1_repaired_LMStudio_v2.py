"""
Capstone Checkpoint 3.1 — Local LM Studio + HuggingFace Wikipedia Retrieval Engine

Flow implemented in this file
=============================

WIKIPEDIA RETRIEVAL ENGINE
Wikipedia HTML Corpus
    -> Category Metadata
    -> Category-aware Routing / Filtering (Tagged Hybrid only)
    -> BM25 / Vector / Hybrid / Tagged Hybrid
    -> Retrieved Wikipedia Evidence
    -> Answering LLM
    -> Generated Answer

EVALUATION LAYER
Question + Generated Answer + grading_notes + Retrieved Sources
    -> RAGAS Experiment
    -> RAGAS experiment
    -> LM Studio LLM Judge

LOCAL MODEL STACK
LM Studio -> answer generation + category classification + RAGAS judge
HuggingFace SentenceTransformers -> Chroma embeddings
    -> PASS / FAIL
    -> Experiment CSV

Retriever choices:
    bm25
    vector
    hybrid
    hybrid_tagged

Examples:
    python capstone_checkpoint_3_1_repaired_LMStudio.py bm25
    python capstone_checkpoint_3_1_repaired_LMStudio.py vector
    python capstone_checkpoint_3_1_repaired_LMStudio.py hybrid
    python capstone_checkpoint_3_1_repaired_LMStudio.py hybrid_tagged
"""

# Local dependencies (install once if needed):
#   pip install langchain-openai langchain-huggingface sentence-transformers \
#       langchain-chroma chromadb rank-bm25 beautifulsoup4 python-dotenv ragas openai
#
# LM Studio:
#   1. Load deepseek-coder-6.7b-instruct (or change LM_STUDIO_MODEL).
#   2. Start the Local Server on http://127.0.0.1:1234.
#   3. The HuggingFace embedding model downloads once, then runs locally.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from rank_bm25 import BM25Okapi
from ragas import Dataset, experiment

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================================
# STEP 1 — CONFIGURATION
# ============================================================================

SCENARIO = "wikipedia"

# ------------------------------------------------------------------
# LOCAL MODEL CONFIGURATION
# ------------------------------------------------------------------
LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL",
    "http://127.0.0.1:1234/v1",
)

# Use the exact model identifier shown by LM Studio's local server.
LLM_MODEL = os.getenv(
    "LM_STUDIO_MODEL",
    "deepseek-coder-6.7b-instruct",
)
JUDGE_MODEL = os.getenv(
    "LM_STUDIO_JUDGE_MODEL",
    LLM_MODEL,
)

# HuggingFace / SentenceTransformers embeddings run locally.
# On the first use, the model is downloaded from Hugging Face and cached.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# IMPORTANT: use a NEW Chroma directory because a vector DB built with
# OpenAI/OpenRouter embeddings is incompatible with HuggingFace embeddings.
WIKIPEDIA_DIR = Path(os.getenv("WIKIPEDIA_HTML_DIR", "Wikipedia_10"))
HTML_DIR = WIKIPEDIA_DIR
CHROMA_DIR = os.getenv("WIKIPEDIA_CHROMA_DIR", "Wikipedia_chroma_10")
CATEGORY_CACHE_PATH = Path(
    os.getenv("WIKIPEDIA_CATEGORY_CACHE", "wikipedia_category_metadata_1.json")
)
LOG_PATH = Path.cwd() / "checkpoint_3_1_retrieval_LMStudio_20_1.log"

NUM_RETRIEVED = int(os.getenv("NUM_RETRIEVED", "6"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.4"))
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "0.6"))
TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.0"))

# If category routing finds too few candidate documents, Tagged Hybrid safely
# falls back to the full corpus rather than returning a tiny/empty retrieval set.
MIN_TAGGED_CANDIDATES = int(os.getenv("MIN_TAGGED_CANDIDATES", "3"))


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

# Deterministic query-routing hints. Exact Wikipedia-title matches are checked
# first, so these keywords are primarily a fallback for queries that do not name
# a corpus article explicitly.
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


ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "Wikipedia evidence. Quote from the evidence when the question requests a quote. "
    "If the evidence does not contain the answer, say so rather than guessing."
)


# Evaluation uses the RAGAS experiment runner, but the PASS/FAIL metric itself
# is a plain LM Studio judge. This avoids structured-output / Instructor
# response_format incompatibilities with some local OpenAI-compatible models.
JUDGE_SYSTEM = (
    "You are a strict evaluator. Compare the RESPONSE with the GRADING NOTES. "
    "Judge factual correctness and whether all central required facts are present. "
    "Minor wording differences are acceptable. End with exactly one line: "
    "FINAL_VERDICT=PASS or FINAL_VERDICT=FAIL."
)


# ============================================================================
# STEP 2 — API / LLM HELPERS
# ============================================================================


def make_llm(
    model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> ChatOpenAI:
    """Create the local LM Studio answering/classification LLM.

    LM Studio exposes an OpenAI-compatible API, so ChatOpenAI can talk to it
    without an OpenRouter key. The api_key value is only a local placeholder.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key="lm-studio",
        base_url=LM_STUDIO_BASE_URL,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create local HuggingFace sentence-transformer embeddings for Chroma.

    The embedding model is downloaded and cached by Hugging Face /
    SentenceTransformers on first use, then runs locally on subsequent runs.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )

def check_lm_studio() -> None:
    """Fail early if the local LM Studio OpenAI-compatible server is unavailable."""
    try:
        client = OpenAI(api_key="lm-studio", base_url=LM_STUDIO_BASE_URL)
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"LM Studio reachable at {LM_STUDIO_BASE_URL}")
        if available:
            print("Available LM Studio model(s): " + ", ".join(available[:8]))
            if JUDGE_MODEL not in available:
                print(
                    f"WARNING: LM_STUDIO_MODEL={JUDGE_MODEL!r} was not listed. "
                    "If LM Studio rejects the model name, set LM_STUDIO_MODEL to "
                    "one of the available model IDs shown above."
                )
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to LM Studio. Start the LM Studio local server "
            f"at {LM_STUDIO_BASE_URL} before running evaluation. Original error: {exc}"
        ) from exc


def make_ragas_judge() -> ChatOpenAI:
    """Create a plain LM Studio judge used inside the RAGAS experiment runner.

    We intentionally do NOT use RAGAS llm_factory/Instructor here because some
    LM Studio models reject the structured response_format generated by that
    path. The experiment is still managed and saved by RAGAS.
    """
    return make_llm(model=JUDGE_MODEL, temperature=0.0)


def local_correctness_score(
    llm: ChatOpenAI,
    response: str,
    grading_notes: str,
    *,
    question: str = "",
    debug: bool = False,
) -> str:
    """Return 'pass' or 'fail' using a plain-text LM Studio judge."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=(
                f"QUESTION:\n{question}\n\n"
                f"RESPONSE:\n{response}\n\n"
                f"GRADING NOTES:\n{grading_notes}\n\n"
                "Finish with FINAL_VERDICT=PASS or FINAL_VERDICT=FAIL."
            )
        ),
    ]
    raw = str(llm.invoke(messages).content).strip()
    normalized = raw.upper()

    matches = re.findall(r"FINAL_VERDICT\s*=\s*(PASS|FAIL)", normalized)
    if matches:
        verdict = matches[-1].lower()
    else:
        matches = re.findall(r"\bVERDICT\s*[:=-]\s*(PASS|FAIL)\b", normalized)
        if matches:
            verdict = matches[-1].lower()
        else:
            has_pass = bool(re.search(r"\bPASS\b", normalized))
            has_fail = bool(re.search(r"\bFAIL\b", normalized))
            verdict = "pass" if has_pass and not has_fail else "fail"

    if debug:
        print("  [judge raw output]")
        for line in raw.splitlines():
            print(f"    {line}")
        print(f"  [parsed verdict] {verdict.upper()}")

    return verdict


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# ============================================================================
# STEP 3 — LOAD WIKIPEDIA HTML CORPUS
# ============================================================================


def extract_text_from_html(file_path: Path) -> str:
    """Load one HTML file and convert it to readable text."""
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def load_wikipedia_corpus(html_dir: Path) -> list[dict]:
    """Load all .html files as {'id', 'text'} records."""
    html_files = sorted(html_dir.glob("*.html"))
    print(f"Found {len(html_files)} Wikipedia HTML files.")

    documents: list[dict] = []
    for file_path in html_files:
        try:
            text = extract_text_from_html(file_path)
            if not text.strip():
                print(f"WARNING: Empty document: {file_path.name}")
                continue
            documents.append({"id": file_path.name, "text": text})
        except Exception as exc:
            print(f"WARNING: Could not load {file_path.name}: {exc}")

    print(f"Successfully loaded {len(documents)} Wikipedia documents.")
    for doc in documents[:10]:
        print(f"{doc['id']:<45} {len(doc['text']):>8} characters")
    if len(documents) > 10:
        print(f"... plus {len(documents) - 10} additional documents.")

    if not documents:
        raise RuntimeError(
            f"No Wikipedia HTML files were loaded from {html_dir.resolve()}. "
            "Check WIKIPEDIA_HTML_DIR / HTML_DIR."
        )

    return documents


# ============================================================================
# STEP 4 — CATEGORY METADATA
# ============================================================================


def classify_wikipedia_category(filename: str, text: str, llm: ChatOpenAI) -> str:
    """Classify one Wikipedia article into exactly one fixed category."""
    title = filename.removesuffix(".html").replace("_", " ")
    lead_text = text[:4000]
    category_description = "\n".join(
        f"- {key}: {description}" for key, description in CATEGORY_LABELS.items()
    )

    prompt = f"""
Classify the following Wikipedia article into exactly ONE category.

ARTICLE TITLE:
{title}

ARTICLE OPENING:
{lead_text}

AVAILABLE CATEGORIES:
{category_description}

Rules:
1. Return ONLY the category key.
2. Do not explain your answer.
3. Choose the most specific appropriate category.
4. If no category clearly applies, return "other".

CATEGORY:
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You classify Wikipedia articles into a fixed taxonomy. "
                    "Return only the requested category key."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    raw = str(response.content).strip().lower().replace("`", "")

    # Local models sometimes answer with a sentence such as
    # "The category is arts_entertainment" even when asked for only the key.
    # Search the complete response for a valid taxonomy key instead of taking
    # the first word (which previously produced the erroneous category "the").
    category = None
    for key in CATEGORY_LABELS:
        if re.search(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])", raw):
            category = key
            break

    if category is None:
        print(
            f"WARNING: Could not identify a valid category in {raw!r} "
            f"for {filename}; using 'other'."
        )
        category = "other"

    return category


def create_category_metadata(
    documents: list[dict],
    cache_path: Path = CATEGORY_CACHE_PATH,
) -> dict[str, str]:
    """Create/load category metadata, classifying only uncached articles."""
    category_by_file: dict[str, str] = {}

    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                category_by_file = {
                    str(k): str(v)
                    for k, v in loaded.items()
                    if str(v) in CATEGORY_LABELS
                }
            print(
                f"Loaded category metadata for {len(category_by_file)} Wikipedia articles."
            )
        except Exception as exc:
            print(f"WARNING: Could not load category cache: {exc}")

    uncategorized = [d for d in documents if d["id"] not in category_by_file]
    if not uncategorized:
        print("All Wikipedia articles already have category metadata.")
        return category_by_file

    print(f"Need to classify {len(uncategorized)} new Wikipedia articles.")
    llm = make_llm(temperature=0)

    for index, doc in enumerate(uncategorized, start=1):
        try:
            category = classify_wikipedia_category(doc["id"], doc["text"], llm)
        except Exception as exc:
            print(f"WARNING: category classification failed for {doc['id']}: {exc}")
            category = "other"

        category_by_file[doc["id"]] = category
        print(f"[{index}/{len(uncategorized)}] {doc['id']:<45} -> {category}")

        # Save after every article so a long 2,400-document classification run can resume.
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(category_by_file, f, indent=2, ensure_ascii=False)

    print(f"Category metadata saved to: {cache_path.resolve()}")
    return category_by_file


def attach_category_metadata(documents: list[dict]) -> dict[str, str]:
    """Create/load metadata and attach a category field to every corpus document."""
    category_by_file = create_category_metadata(documents)
    for doc in documents:
        doc["category"] = category_by_file.get(doc["id"], "other")
    return category_by_file


# ============================================================================
# STEP 5 — QUERY CATEGORY ROUTING
# ============================================================================


def _readable_title(filename: str) -> str:
    return filename.removesuffix(".html").replace("_", " ").strip()


def infer_query_categories(question: str, documents: list[dict]) -> list[str]:
    """
    Infer relevant metadata categories for Tagged Hybrid retrieval.

    Routing order:
      1. Exact/near-exact article-title mentions in the question.
      2. Deterministic keyword hints.
      3. If nothing is confidently inferred, return [] and Tagged Hybrid falls
         back to the full corpus.

    The router itself uses no LLM call, keeping tagged retrieval cheap and stable.
    """
    q = " " + re.sub(r"\s+", " ", question.lower()).strip() + " "
    categories: list[str] = []

    # Strongest signal: the query explicitly names one or more corpus articles.
    for doc in documents:
        title = _readable_title(doc["id"]).lower()
        if len(title) >= 4 and title in q:
            category = doc.get("category", "other")
            if category in CATEGORY_LABELS and category != "other":
                categories.append(category)

    # Secondary deterministic routing signals.
    for category, hints in CATEGORY_QUERY_HINTS.items():
        if any(hint in q for hint in hints):
            categories.append(category)

    # Stable de-duplication while preserving priority/order.
    return list(dict.fromkeys(categories))


def filter_documents_by_categories(
    documents: list[dict],
    categories: Iterable[str],
) -> list[dict]:
    wanted = set(categories)
    if not wanted:
        return documents
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

        messages = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
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


# ============================================================================
# STEP 8 — RETRIEVER FACTORY
# ============================================================================


def make_retriever(
    kind: str,
    documents: list[dict],
    db: Chroma | None = None,
) -> BaseRetriever:
    if kind == "bm25":
        return BM25Retriever(documents=documents)

    if kind == "vector":
        if db is None:
            raise ValueError("Vector retrieval requires a Chroma database.")
        return VectorRetriever(db=db, documents=documents)

    if kind == "hybrid":
        if db is None:
            raise ValueError("Hybrid retrieval requires a Chroma database.")
        return HybridRetriever(db=db, documents=documents)

    if kind == "hybrid_tagged":
        if db is None:
            raise ValueError("Tagged Hybrid retrieval requires a Chroma database.")
        return TaggedHybridRetriever(db=db, documents=documents)

    raise ValueError(f"Unknown retriever type: {kind}")


# ============================================================================
# STEP 9 — FIXED CHECKPOINT 3.1 EVALUATION SET
# ============================================================================


def my_eval_set() -> list[dict]:
    """Evaluation questions for the Wikipedia Retrieval Engine."""

    return [

        {
            "question": (
                "According to the Wikipedia article on the Empire State Building, "
                "when was the building completed and when did it officially open?"
            ),
            "grading_notes": (
                "States that the Empire State Building was completed on April 11, 1931 "
                "and officially opened on May 1, 1931."
            ),
        },

        {
            "question": (
                "According to the Wikipedia article on Hallstatt, where is Hallstatt "
                "located and what is it particularly known for?"
            ),
            "grading_notes": (
                "States that Hallstatt is in Austria and identifies its historical "
                "significance, especially its association with salt mining and/or the "
                "prehistoric Hallstatt culture."
            ),
        },

        {
            "question": (
                "What role did Steve Jobs play in the development of Apple, according "
                "to his Wikipedia article?"
            ),
            "grading_notes": (
                "Identifies Steve Jobs as a co-founder and major leader of Apple and "
                "describes his important role in the company's development and products."
            ),
        },

        {
            "question": (
                "Compare the leadership roles and historical significance of Margaret "
                "Thatcher and Adolf Hitler as described in their respective Wikipedia articles."
            ),
            "grading_notes": (
                "Identifies Margaret Thatcher as a British prime minister and Adolf Hitler "
                "as the dictator of Nazi Germany, and clearly distinguishes their different "
                "political roles, historical periods, and historical significance."
            ),
        },

        {
            "question": (
                "How were Margaret Thatcher and Adolf Hitler connected through the major "
                "European political and military events that shaped the periods in which "
                "they rose to prominence?"
            ),
            "grading_notes": (
                "Recognizes that they belonged to different historical periods and does "
                "not invent a direct relationship. A correct answer should distinguish "
                "their historical contexts and explain that broader connections require "
                "synthesizing evidence across multiple historical events or articles."
            ),
        },

        {
            "question": (
                "According to the Wikipedia article on Emirates airline, when was Emirates "
                "founded, where is it based, and who owns it? Quote the relevant sentence "
                "or sentences."
            ),
            "grading_notes": (
                "States that Emirates was founded in 1985, is based in Dubai, United Arab "
                "Emirates, and is owned by the Emirates Group, and includes a relevant "
                "supporting quote or quotes from the article."
            ),
        },

        {
            "question": (
                "Which Austrian settlement in the corpus is strongly associated with "
                "prehistoric salt mining, and where is it located?"
            ),
            "grading_notes": (
                "Identifies Hallstatt as the settlement and correctly states that it is "
                "located in Austria. The answer should connect Hallstatt with its long "
                "history of salt mining."
            ),
        },

        {
            "question": (
                "What did Margaret Thatcher study at the University of Oxford, and what "
                "academic qualification did she receive?"
            ),
            "grading_notes": (
                "Correctly identifies Thatcher's field of study at Oxford and accurately "
                "describes the academic qualification reported in the article."
            ),
        },

        {
            "question": (
                "Compare the roles of Queen Victoria and Margaret Thatcher in British "
                "history using evidence from their respective Wikipedia articles."
            ),
            "grading_notes": (
                "Identifies Queen Victoria as a British monarch and Margaret Thatcher as "
                "a British prime minister, and clearly distinguishes the constitutional, "
                "political, and historical roles described in the two articles."
            ),
        },

        {
            "question": (
                "Who was Victoria, what position did she hold, and which country or "
                "kingdom did she rule?"
            ),
            "grading_notes": (
                "Correctly resolves Victoria to Queen Victoria and identifies her as "
                "monarch of the United Kingdom, while accurately describing her role "
                "from the retrieved article."
            ),
        },

        {
            "question": (
                "What is Mytilidae, and what type of organisms belong to this biological family?"
            ),
            "grading_notes": (
                "Identifies Mytilidae as a family of marine bivalve mollusks commonly "
                "associated with mussels and accurately describes the biological group."
            ),
        },

        {
            "question": (
                "According to the Wikipedia article on birds, what major characteristics "
                "distinguish birds from other animals?"
            ),
            "grading_notes": (
                "Identifies defining characteristics described in the article, such as "
                "feathers, beaks, egg laying, wings, and other major biological traits. "
                "The answer should be grounded in the Bird article."
            ),
        },

        {
            "question": (
                "Which landmark in the corpus is an Art Deco skyscraper in Manhattan "
                "that was completed in the early 1930s?"
            ),
            "grading_notes": (
                "Identifies the Empire State Building and correctly connects it with "
                "Manhattan, Art Deco architecture, and completion in the early 1930s."
            ),
        },

        {
            "question": (
                "Which airline in the corpus was established in Dubai, and what organization "
                "owns it?"
            ),
            "grading_notes": (
                "Identifies Emirates airline and correctly states that it is based in Dubai "
                "and owned by the Emirates Group."
            ),
        },

        {
            "question": (
                "What was Steve Jobs's favorite restaurant, according to the Wikipedia "
                "evidence available in the corpus?"
            ),
            "grading_notes": (
                "PASS if the response states that the retrieved Wikipedia evidence does "
                "not provide sufficient information to identify Steve Jobs's favorite "
                "restaurant. FAIL if the system invents or supplies an unsupported answer."
            ),
        },

        {
            "question": (
                "According to the corpus, when did Margaret Thatcher become President "
                "of the United States?"
            ),
            "grading_notes": (
                "PASS if the response rejects the false premise and explains that "
                "Margaret Thatcher did not become President of the United States. "
                "FAIL if the system accepts the premise or invents a date."
            ),
        },

        {
            "question": (
                "Compare the historical significance of the Empire State Building and "
                "Emirates airline within their respective domains."
            ),
            "grading_notes": (
                "Uses evidence from both relevant articles and explains the significance "
                "of the Empire State Building in architecture or New York history and "
                "Emirates in commercial aviation, without confusing the two domains."
            ),
        },

        {
            "question": (
                "How did Margaret Thatcher's education and early professional career "
                "precede her entry into national politics?"
            ),
            "grading_notes": (
                "Connects evidence about Thatcher's education, early professional career, "
                "and subsequent political career using information from the Margaret "
                "Thatcher article without adding unsupported claims."
            ),
        },

        {
            "question": (
                "What were the 85th Academy Awards, and what major event or achievement "
                "did the ceremony recognize?"
            ),
            "grading_notes": (
                "Correctly identifies the 85th Academy Awards as an Academy Awards ceremony "
                "honoring achievements in film and accurately summarizes its purpose based "
                "on the article."
            ),
        },

        {
            "question": (
                "Compare birds and members of the Mytilidae family in terms of their "
                "biological classification and major physical characteristics."
            ),
            "grading_notes": (
                "Uses evidence from both the Bird and Mytilidae articles, correctly "
                "distinguishes birds from marine bivalve mollusks, and summarizes major "
                "characteristics of each group."
            ),
        },

    ]


def load_eval_inputs(inputs_path: str | None = None) -> list[dict]:
    """
    Load evaluation questions.

    The external variants JSON may contain columns from a previous
    experiment such as response, score, retrieved_sources, or sources.
    Only question and grading_notes are required for a fresh evaluation.
    """

    if inputs_path is None:
        print("Using built-in 6-question baseline evaluation set.")
        return my_eval_set()

    path = Path(inputs_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation input file not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8-sig") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(
            "Evaluation input JSON must contain a list of records."
        )

    cleaned_data = []

    for i, row in enumerate(raw_data, start=1):

        if not isinstance(row, dict):
            raise ValueError(
                f"Evaluation row {i} is not a JSON object."
            )

        question = str(row.get("question", "")).strip()
        grading_notes = str(row.get("grading_notes", "")).strip()

        if not question:
            raise ValueError(
                f"Evaluation row {i} is missing 'question'."
            )

        if not grading_notes:
            raise ValueError(
                f"Evaluation row {i} is missing 'grading_notes'."
            )

        # IMPORTANT:
        # Give every RAGAS row exactly the same schema.
        cleaned_data.append({
            "question": question,
            "grading_notes": grading_notes,
        })

    print(
        f"Loaded {len(cleaned_data)} evaluation questions "
        f"from {path.resolve()}"
    )

    print(
        "Normalized evaluation schema to: "
        "question, grading_notes"
    )

    return cleaned_data

# ============================================================================
# STEP 10 — RAGAS DATASET + PLAIN LM STUDIO JUDGE EXPERIMENT
# ============================================================================


# def load_ragas_dataset() -> Dataset:
#     dataset = Dataset(
#         name="wikipedia_capstone_3_1_eval",
#         backend="local/csv",
#         root_dir="ragas_experiments",
#     )

#     # Recreating the file each run avoids accidental duplicate rows if the same
#     # script is run repeatedly during retriever comparison.
#     dataset_path = Path("ragas_experiments") / "datasets" / "wikipedia_capstone_3_1_eval.csv"
#     if dataset_path.exists():
#         dataset_path.unlink()
#         dataset = Dataset(
#             name="wikipedia_capstone_3_1_eval",
#             backend="local/csv",
#             root_dir="ragas_experiments",
#         )

#     for sample in my_eval_set():
#         dataset.append(sample)
#     dataset.save()
#     return dataset

def load_ragas_dataset(
    eval_inputs: list[dict],
    dataset_name: str = "wikipedia_capstone_3_1_eval",
) -> Dataset:

    dataset = Dataset(
        name=dataset_name,
        backend="local/csv",
        root_dir="ragas_experiments",
    )

    dataset_path = (
        Path("ragas_experiments")
        / "datasets"
        / f"{dataset_name}.csv"
    )

    if dataset_path.exists():
        dataset_path.unlink()

        dataset = Dataset(
            name=dataset_name,
            backend="local/csv",
            root_dir="ragas_experiments",
        )

    for sample in eval_inputs:
        dataset.append(sample)

    dataset.save()

    return dataset


def build_ragas_experiment(
    retriever: BaseRetriever,
    kind: str,
    ragas_judge,
):
    """Build one RAGAS experiment for the selected retriever."""

    @experiment()
    async def run_experiment(row):
        question = row["question"]
        categories = infer_query_categories(question, retriever.documents)

        try:
            trace = retriever.query_with_trace(question)
            response = trace["answer"]
            sources = trace["sources"]
        except Exception as exc:
            return {
                **row,
                "retriever": kind,
                "query_categories": "; ".join(categories) or "all",
                "retrieved_sources": "",
                "response": f"[GENERATION ERROR] {exc}",
                "score": "fail",
                "score_reason": "generation error",
            }

        try:
            score_value = local_correctness_score(
                llm=ragas_judge,
                response=response,
                grading_notes=row["grading_notes"],
                question=question,
            )
            score_reason = "plain-text LM Studio judge inside RAGAS experiment"
        except Exception as exc:
            score_value = "fail"
            score_reason = f"LM Studio judge error: {exc}"

        return {
            **row,
            "retriever": kind,
            "query_categories": "; ".join(categories) or "all",
            "retrieved_sources": "; ".join(sources),
            "response": response,
            "score": score_value,
            "score_reason": score_reason,
        }

    return run_experiment


# async def run_ragas_evaluation(retriever: BaseRetriever, kind: str):
async def run_ragas_evaluation(
    retriever: BaseRetriever,
    kind: str,
    eval_inputs: list[dict],
    evaluation_name: str = "baseline",
):
    ragas_judge = make_ragas_judge()
    # dataset = load_ragas_dataset()
    dataset_name = f"wikipedia_capstone_3_1_{evaluation_name}"
    dataset = load_ragas_dataset(eval_inputs, dataset_name,)

    print(
        f"\nCheckpoint 3.1 — RAGAS experiment + plain LM Studio judge "
        f"| scenario: {SCENARIO} | retriever: {kind}\n"
    )
    print(f"Loaded {len(dataset)} fixed evaluation questions.")

    # experiment_name = f"checkpoint_3_1_{kind}"
    experiment_name = (f"checkpoint_3_1_{kind}_{evaluation_name}")
    results = await build_ragas_experiment(
        retriever, kind, ragas_judge
    ).arun(dataset, name=experiment_name)

    if not results:
        print("RAGAS produced 0 result rows. Check errors above.")
        return results

    passes = sum(1 for r in results if str(r.get("score", "")).lower() == "pass")
    print("=" * 72)
    print(f"RAGAS pass rate: {passes}/{len(results)} ({passes / len(results):.0%})")

    for i, row in enumerate(results, 1):
        print("-" * 72)
        print(f"Q{i}: {row['question']}")
        print(f"  categories={row.get('query_categories', 'all')}")
        print(f"  retrieved={row.get('retrieved_sources', '')}")
        print(f"  verdict={str(row.get('score', 'fail')).upper()}")
        answer_preview = str(row.get("response", "")).replace("\n", " ").strip()
        if len(answer_preview) > 500:
            answer_preview = answer_preview[:500] + " ..."
        print(f"  answer={answer_preview}")

        log(
            f"Q{i}: {row['question']}",
            (
                f"retriever={kind}\n"
                f"categories={row.get('query_categories', '')}\n"
                f"retrieved={row.get('retrieved_sources', '')}\n"
                f"verdict={row.get('score', '')}\n"
                f"answer={row.get('response', '')}\n"
                f"reason={row.get('score_reason', '')}"
            ),
        )

    csv_path = (
        Path("ragas_experiments")
        / "experiments"
        / f"{experiment_name}.csv"
    )
    print(f"Experiment CSV: {csv_path.resolve()}")
    return results


# ============================================================================
# STEP 11 — VALIDATE THAT THE EVALUATION FRAMEWORK CATCHES A BAD ANSWER
# ============================================================================


async def validate_framework_ragas() -> None:
    """Validate the plain LM Studio PASS/FAIL judge with known answers."""
    ragas_judge = make_ragas_judge()
    question = (
        "According to the Wikipedia article on Emirates airline, when was Emirates "
        "founded, where is it based, and who owns it?"
    )
    notes = (
        "States that Emirates was founded in 1985, is based in Dubai, United Arab "
        "Emirates, and is owned by the Emirates Group."
    )

    good = (
        "Emirates was founded in 1985. It is based in Dubai, United Arab Emirates, "
        "and is owned by The Emirates Group."
    )
    manipulated = (
        "Emirates was founded in 1995, is based in Abu Dhabi, and is owned by Qatar Airways."
    )

    print("\n--- LM Studio judge framework validation ---")

    good_score = local_correctness_score(
        llm=ragas_judge,
        question=question,
        response=good,
        grading_notes=notes,
        debug=True,
    )
    print(f"Known-correct answer     -> {good_score.upper()} (expected PASS)")

    bad_score = local_correctness_score(
        llm=ragas_judge,
        question=question,
        response=manipulated,
        grading_notes=notes,
        debug=True,
    )
    print(f"Known-manipulated answer -> {bad_score.upper()} (expected FAIL)")

    log(
        "RAGAS FRAMEWORK VALIDATION",
        f"good={good_score} manipulated={bad_score}",
    )


# ============================================================================
# STEP 12 — MAIN
# ============================================================================


async def main() -> None:

    # ---------------------------------------------------------------
    # COMMAND-LINE ARGUMENTS
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Evaluate the Wikipedia capstone with RAGAS."
    )

    parser.add_argument(
        "retriever",
        nargs="?",
        choices=["bm25", "vector", "hybrid", "hybrid_tagged"],
        default="hybrid",
        help=(
            "Retriever to evaluate. hybrid is the ordinary full-corpus baseline; "
            "hybrid_tagged adds category-aware routing/filtering before hybrid fusion."
        ),
    )

    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help=(
            "Optional JSON evaluation file containing questions and grading_notes. "
            "If omitted, the built-in 6-question baseline is used."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # LOAD EVALUATION QUESTIONS
    # ---------------------------------------------------------------
    eval_inputs = load_eval_inputs(args.inputs)

    evaluation_name = "variants" if args.inputs else "baseline"

    # Validate LM Studio before expensive corpus work.
    check_lm_studio()

    # ---------------------------------------------------------------
    # LOAD WIKIPEDIA CORPUS + CATEGORY METADATA
    # ---------------------------------------------------------------
    documents = load_wikipedia_corpus(HTML_DIR)
    attach_category_metadata(documents)

    print("\nCategory distribution:")
    counts: dict[str, int] = {}

    for doc in documents:
        category = doc.get("category", "other")
        counts[category] = counts.get(category, 0) + 1

    for category, count in sorted(counts.items()):
        print(f"  {category:<24} {count}")

    # ---------------------------------------------------------------
    # BUILD VECTOR DATABASE IF REQUIRED
    # ---------------------------------------------------------------
    db = None

    if args.retriever in {"vector", "hybrid", "hybrid_tagged"}:
        db = build_or_load_db(documents)

    # ---------------------------------------------------------------
    # CREATE RETRIEVER
    # ---------------------------------------------------------------
    retriever = make_retriever(
        kind=args.retriever,
        documents=documents,
        db=db,
    )

    # ---------------------------------------------------------------
    # RUN RAGAS EVALUATION
    # ---------------------------------------------------------------
    await run_ragas_evaluation(
        retriever,
        args.retriever,
        eval_inputs,
        evaluation_name,
    )

    # Validate that the judge catches a deliberately incorrect answer.
    await validate_framework_ragas()

if __name__ == "__main__":
    asyncio.run(main())
