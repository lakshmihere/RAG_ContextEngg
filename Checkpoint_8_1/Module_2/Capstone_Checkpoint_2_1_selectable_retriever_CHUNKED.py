r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# #LLM_MODEL = "openai/gpt-oss-120b"
# LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
# EMBEDDING_MODEL = "openai/text-embedding-3-small"
# CHROMA_DIR = "wikipedia_chroma_leader_lens_chunks"

# NUM_RETRIEVED = 6         # Emails sent to the LLM as context.
# CANDIDATE_POOL = 20        # Candidates pulled from EACH retriever before fusion.
# TOP_K = 6
# CHUNK_SIZE = 1200
# CHUNK_OVERLAP = 150

# WEIGHT_BM25 = 0.4
# WEIGHT_VECTOR = 0.6
# TEMPERATURE = 0.2


# LLM_MODEL = "openai/gpt-5.4-mini"
# JUDGE_MODEL = "openai/gpt-5.4"
# EMBEDDING_MODEL = "openai/text-embedding-3-small"
# LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval_2.log"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
JUDGE_MODEL = "openai/gpt-5.4"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# IMPORTANT: new DB name because this index contains chunks,
# not the previous 65 whole Wikipedia chunks.
CHROMA_DIR = "wikipedia_chroma_full"

NUM_RETRIEVED = 6
CANDIDATE_POOL = 20
TOP_K = 6

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

WEIGHT_BM25 = 0.4
WEIGHT_VECTOR = 0.6

TEMPERATURE = 0.2

LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval_2.log"


# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "wikipedia"   # "research_papers" or "wikipedia"
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def extract_text_from_html(file_path: Path) -> str:
    """Load one HTML file and convert it to clean readable text."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Extract visible text
    return soup.get_text(separator=" ", strip=True)

def load_wikipedia_corpus(html_dir: Path) -> list[dict]:
    """
    Load every .html file in the Wikipedia corpus directory.
    Returns:
        A list of dictionaries with:
        {
            "id": filename,
            "text": extracted article text
        }
    """
    # Find every HTML file in the directory
    html_files = sorted(html_dir.glob("*.html"))
    print(f"Found {len(html_files)} Wikipedia HTML files.")
    documents = []
    for file_path in html_files:
        try:
            text = extract_text_from_html(file_path)
            # Skip empty documents
            if not text.strip():
                print(f"WARNING: Empty document: {file_path.name}")
                continue
            documents.append(
                {
                    "id": file_path.name,
                    "text": text,
                }
            )
        except Exception as e:
            print(f"WARNING: Could not load {file_path.name}: {e}")
    print(f"Successfully loaded {len(documents)} Wikipedia documents.")
    # Show only a small sample
    for doc in documents[:10]:
        print(
            f"{doc['id']:<40} "
            f"{len(doc['text']):>8} characters"
        )
    if len(documents) > 10:
        print(
            f"... plus {len(documents) - 10} additional documents."
        )
    return documents

def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split Wikipedia chunks into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    step = chunk_size - overlap
    chunks = []

    for document in documents:
        text = document["text"]
        source = document["id"]

        chunk_number = 0

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if not chunk_text.strip():
                continue

            chunks.append(
                {
                    "id": f"{source}::chunk_{chunk_number}",
                    "source": source,
                    "chunk_id": chunk_number,
                    "text": chunk_text,
                }
            )

            chunk_number += 1

            if start + chunk_size >= len(text):
                break

    return chunks


HTML_DIR = Path("Leader_lens")
SAMPLE_DOCS = load_wikipedia_corpus(HTML_DIR)
SAMPLE_CHUNKS = chunk_documents(SAMPLE_DOCS)
DOC_BY_ID = {
    d["id"]: d
    for d in SAMPLE_DOCS
}

print()
print(f"Articles loaded : {len(SAMPLE_DOCS)}")
print(f"Chunks created  : {len(SAMPLE_CHUNKS)}")
print(f"Chunk size      : {CHUNK_SIZE}")
print(f"Chunk overlap   : {CHUNK_OVERLAP}")

TEXT_DIR = Path("Wikipedia_text")
SAMPLE_DOCS = load_wikipedia_corpus(TEXT_DIR)
SAMPLE_CHUNKS = chunk_documents(SAMPLE_DOCS)
DOC_BY_ID = {
    d["id"]: d
    for d in SAMPLE_DOCS
}

print()
print(f"Articles loaded : {len(SAMPLE_DOCS)}")
print(f"Chunks created  : {len(SAMPLE_CHUNKS)}")
print(f"Chunk size      : {CHUNK_SIZE}")
print(f"Chunk overlap   : {CHUNK_OVERLAP}")


# ─── Keyword side (BM25) ──────────────────────────────────────────────
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
    """Tokenize text for BM25 and remove common stopwords."""
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOPWORDS
    ]


# ─── Vector side (Chroma + OpenAI embeddings) ─────────────────────────
def get_embeddings() -> OpenAIEmbeddings:
    """Create the embedding model used by Chroma."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(chroma_dir: str = CHROMA_DIR) -> Chroma:
    """Build/load a persistent Chroma DB over Wikipedia text chunks."""
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing chunk vector DB from {chroma_dir}/")
        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(),
        )

    print("Building vector DB (first run — embedding Wikipedia chunks)...")

    docs = [
        Document(
            page_content=d["text"],
            metadata={
                "source": d["source"],
                "chunk_id": d["chunk_id"],
                "chunk_key": d["id"],
            },
        )
        for d in SAMPLE_CHUNKS
    ]

    if not docs:
        raise RuntimeError(
            f"No Wikipedia chunks were created. Check HTML_DIR={HTML_DIR!s}."
        )

    db = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=chroma_dir,
    )
    print(
        f"  Indexed {len(docs)} Wikipedia chunks from "
        f"{len(SAMPLE_DOCS)} articles into {chroma_dir}/"
    )
    return db


class BaseRetriever(ABC):
    """Base class that sends retrieved context to the grounded-answer LLM."""

    def __init__(self, llm_model: str = LLM_MODEL):
        super().__init__()
        self._llm = ChatOpenAI(
            model=llm_model,
            temperature=TEMPERATURE,
            api_key=check_api_key(),
            base_url=OPENROUTER_BASE_URL,
        )

    @abstractmethod
    def retrievedContext(self, query: str) -> str:
        ...

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale scores to [0, 1].

    BM25 uses higher-is-better scores. Chroma returns vector DISTANCES, where
    lower is better, so vector scores are normalized with invert=True.
    """
    if not scores:
        return []

    lo, hi = min(scores), max(scores)

    if hi == lo:
        return [0.5] * len(scores)

    normalized = [(score - lo) / (hi - lo) for score in scores]

    if invert:
        normalized = [1.0 - score for score in normalized]

    return normalized


class BM25Retriever(BaseRetriever):
    """BM25-only retrieval over the Wikipedia corpus."""

    def __init__(
        self,
        documents: list[dict] = SAMPLE_CHUNKS,
        num_retrieved: int = NUM_RETRIEVED,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_retrieved = num_retrieved
        self._names = [d["id"] for d in documents]
        self._contents = [d["text"] for d in documents]

        if not self._contents:
            raise RuntimeError(
                "Cannot build BM25Retriever: no Wikipedia documents loaded."
            )

        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])

        print(
            f"BM25 retriever ready over {len(self._names)} Wikipedia chunks."
        )

    def getTopK(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Return the top-k Wikipedia chunks ranked by BM25 score."""
        k = min(k, len(self._names))
        scores = self._bm25.get_scores(tokenize(query))

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (self._names[i], self._contents[i], float(scores[i]))
            for i in top_indices
        ]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, self._num_retrieved)
        return "\n\n---\n\n".join(
            f"[{name}]\n{content}"
            for name, content, _ in results
        )


class VectorRetriever(BaseRetriever):
    """Semantic vector-only retrieval using Chroma."""

    def __init__(
        self,
        db: Chroma,
        num_retrieved: int = NUM_RETRIEVED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._db = db
        self._num_retrieved = num_retrieved
        print("Vector retriever ready.")

    def getTopK(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Return the top-k Wikipedia chunks ranked by vector distance."""
        results = self._db.similarity_search_with_score(query, k=k)
        return [
            (
                f"{doc.metadata.get('source', 'unknown')}::chunk_{doc.metadata.get('chunk_id', '?')}",
                doc.page_content,
                float(distance),
            )
            for doc, distance in results
        ]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, self._num_retrieved)
        return "\n\n---\n\n".join(
            f"[{name}]\n{content}"
            for name, content, _ in results
        )


class HybridRetriever(BaseRetriever):
    """Fuse BM25 keyword retrieval with Chroma semantic vector retrieval."""

    def __init__(
        self,
        db: Chroma,
        documents: list[dict] = SAMPLE_CHUNKS,
        num_retrieved: int = NUM_RETRIEVED,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._db = db
        self._num_retrieved = num_retrieved

        # Build an in-memory BM25 index over the same Wikipedia articles
        # used by Chroma.
        self._names = [d["id"] for d in documents]
        self._contents = [d["text"] for d in documents]

        if not self._contents:
            raise RuntimeError("Cannot build HybridRetriever: no Wikipedia documents loaded.")

        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])

        print(
            f"Hybrid retriever ready over {len(self._names)} Wikipedia chunks "
            f"(BM25 weight={WEIGHT_BM25}, vector weight={WEIGHT_VECTOR})."
        )

    def _bm25_topk(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Return the top-k Wikipedia chunks ranked by BM25."""
        k = min(k, len(self._names))
        scores = self._bm25.get_scores(tokenize(query))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (self._names[i], self._contents[i], float(scores[i]))
            for i in top_indices
        ]

    def _vector_topk(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Return the top-k Wikipedia chunks ranked by vector distance."""
        k = min(k, len(self._names))
        results = self._db.similarity_search_with_score(query, k=k)

        return [
            (
                f"{doc.metadata.get('source', 'unknown')}::chunk_{doc.metadata.get('chunk_id', '?')}",
                doc.page_content,
                float(distance),
            )
            for doc, distance in results
        ]

    def getTopK(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Fuse BM25 and vector rankings using a weighted normalized score.

        BM25 scores are higher-is-better.
        Chroma vector values are distances, so lower-is-better.
        After normalization, both contributions are higher-is-better and are
        combined using WEIGHT_BM25 and WEIGHT_VECTOR.
        """
        pool_size = min(CANDIDATE_POOL, len(self._names))

        bm25_results = self._bm25_topk(query, pool_size)
        vector_results = self._vector_topk(query, pool_size)

        content_by_name: dict[str, str] = {}
        bm25_norm: dict[str, float] = {}
        vector_norm: dict[str, float] = {}

        if bm25_results:
            bm25_values = [score for _, _, score in bm25_results]
            for (name, content, _), normalized_score in zip(
                bm25_results,
                _normalize(bm25_values),
            ):
                content_by_name[name] = content
                bm25_norm[name] = normalized_score

        if vector_results:
            vector_distances = [distance for _, _, distance in vector_results]
            for (name, content, _), normalized_score in zip(
                vector_results,
                _normalize(vector_distances, invert=True),
            ):
                content_by_name[name] = content
                vector_norm[name] = normalized_score

        fused = [
            (
                name,
                content,
                WEIGHT_BM25 * bm25_norm.get(name, 0.0)
                + WEIGHT_VECTOR * vector_norm.get(name, 0.0),
            )
            for name, content in content_by_name.items()
        ]

        # Fused score is higher-is-better.
        fused.sort(key=lambda item: item[2], reverse=True)
        return fused[:k]

    def retrievedContext(self, query: str) -> str:
        """Return the top hybrid-ranked documents as LLM context."""
        results = self.getTopK(query, self._num_retrieved)

        return "\n\n---\n\n".join(
            f"[{name}]\n{content}"
            for name, content, _ in results
        )


def make_retriever(
    kind: str,
    db: Chroma | None = None,
) -> BaseRetriever:
    """Create the retriever selected on the command line."""

    if kind == "bm25":
        return BM25Retriever(
            documents=SAMPLE_CHUNKS,
            num_retrieved=NUM_RETRIEVED,
        )

    if kind == "vector":
        if db is None:
            raise ValueError("Vector retrieval requires a Chroma database.")
        return VectorRetriever(
            db=db,
            num_retrieved=NUM_RETRIEVED,
        )

    if kind == "hybrid":
        if db is None:
            raise ValueError("Hybrid retrieval requires a Chroma database.")
        return HybridRetriever(
            db=db,
            documents=SAMPLE_CHUNKS,
            num_retrieved=NUM_RETRIEVED,
        )

    raise ValueError(f"Unknown retriever type: {kind}")


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.


# # %%
# def my_representative_queries() -> list[str]:
#     """Return representative queries for the Wikipedia Retrieval Engine scenario.

#     These queries test:
#     - single-document factual retrieval,
#     - exact/quoted information retrieval,
#     - multi-document comparison,
#     - and retrieval from different Wikipedia article topics.
#     """

#     return [
#         "According to the Wikipedia article on the Empire State Building, "
#         "when was the building completed and when did it officially open? "
#         "Quote the relevant sentence from the article.",

#         "Quote, word for word, the opening sentence of the Wikipedia article "
#         "on Steve Jobs.",

#         "According to the Wikipedia article on Margaret Thatcher, what was "
#         "her educational background before entering politics? Quote the "
#         "relevant passage.",

#         "Compare the leadership roles and historical significance of "
#         "Margaret Thatcher and Adolf Hitler as described in their respective "
#         "Wikipedia chunks. Support the comparison with a direct quote from "
#         "each article.",

#         "According to the Wikipedia article on Emirates airline, when was "
#         "Emirates founded, where is it based, and who owns it? Quote the "
#         "relevant sentence or sentences.",

#         "According to the Hallstatt article, how many levels does the "
#         "Hallstatt salt mine comprise, and what is the elevation range of "
#         "its shafts?"
#     ]

def my_representative_queries() -> list[str]:
    """
    Return 15 representative queries for the 65-article
    Wikipedia Leader Lens corpus.

    These are intentionally aligned with the Checkpoint 1.1
    baseline probe prompts so the same evaluation set can be
    compared across:
        - no-retrieval baseline,
        - BM25 retrieval,
        - vector retrieval,
        - hybrid retrieval.

    Query coverage:
        1-5   Single-document factual retrieval
        6-8   Multi-fact retrieval
        9-11  Cross-document / multi-hop retrieval
        12-13 Exact-evidence / quotation retrieval
        14    Temporal / leadership comparison
        15    Political-period article retrieval
    """

    return [

        # =========================================================
        # 1-5. SINGLE-DOCUMENT FACTUAL RETRIEVAL
        # =========================================================

        # 1. President
        "According to the Wikipedia article on Nelson Mandela, "
        "what leadership office did he hold in South Africa and during "
        "what years did he serve? Quote the relevant sentence.",

        # 2. Prime Minister
        "According to the Wikipedia article on Margaret Thatcher, "
        "what position did she hold in the United Kingdom and during "
        "what years did she serve? Quote the relevant sentence.",

        # 3. Monarch
        "According to the Wikipedia article on Queen Victoria, "
        "when did her reign begin and when did it end? "
        "Quote the relevant evidence from the article.",

        # 4. U.S. President
        "According to the Wikipedia article on Abraham Lincoln, "
        "which numbered president of the United States was he and "
        "when did he serve? Quote the relevant sentence.",

        # 5. International leader
        "According to the Wikipedia article on Emmanuel Macron, "
        "what national leadership office does the article identify him "
        "as holding? Quote the relevant sentence.",


        # =========================================================
        # 6-8. MULTI-FACT RETRIEVAL
        # =========================================================

        # 6. Justin Trudeau
        "According to the Wikipedia article on Justin Trudeau, "
        "identify his nationality, the leadership office he held, "
        "and his term of service. Support all parts of the answer "
        "with evidence from the article.",

        # 7. Suharto
        "According to the Wikipedia article on Suharto, what does "
        "the article state about his political role, his presidency "
        "of Indonesia, and the length or period of his leadership? "
        "Quote the relevant evidence.",

        # 8. Adolf Hitler
        "According to the Wikipedia article on Adolf Hitler, "
        "identify the major national leadership positions attributed "
        "to him and the periods associated with those positions. "
        "Use only information stated in the article and quote the "
        "supporting evidence.",


        # =========================================================
        # 9-11. CROSS-DOCUMENT / MULTI-HOP RETRIEVAL
        # =========================================================

        # 9. Mandela vs. Suharto
        "Compare Nelson Mandela and Suharto using their respective "
        "Wikipedia chunks. What country did each lead, what office "
        "did each hold, and how did their periods of leadership differ? "
        "Quote evidence from both articles.",

        # 10. Thatcher vs. May
        "Compare Margaret Thatcher and Theresa May using their "
        "respective Wikipedia chunks. What leadership office did "
        "they share, and when did each serve? Cite evidence from "
        "both articles.",

        # 11. Three U.S. presidents
        "Using the Wikipedia chunks on George Washington, "
        "Abraham Lincoln, and Barack Obama, identify which numbered "
        "U.S. president each was and the years each served. "
        "Provide supporting evidence from all three articles.",


        # =========================================================
        # 12-13. EXACT-EVIDENCE / QUOTATION RETRIEVAL
        # =========================================================

        # 12. George Washington
        "Quote, word for word, the opening sentence of the "
        "Wikipedia article on George Washington. Do not paraphrase.",

        # 13. Cleopatra
        "Quote, word for word, the opening sentence of the "
        "Wikipedia article on Cleopatra. Do not supplement the "
        "quotation with outside knowledge.",


        # =========================================================
        # 14. TEMPORAL / LEADERSHIP COMPARISON
        # =========================================================

        # 14. UK prime ministers
        "Using the Wikipedia chunks on Boris Johnson, Liz Truss, "
        "and Rishi Sunak, identify the prime-ministerial term stated "
        "for each person and place the three leaders in chronological "
        "order. Quote supporting evidence from each article.",


        # =========================================================
        # 15. POLITICAL-PERIOD ARTICLE RETRIEVAL
        # =========================================================

        # 15. Presidency of Joe Biden
        "According to the Wikipedia article titled "
        "'Presidency of Joe Biden', what event marked the beginning "
        "of Biden's presidential tenure and on what date did it occur? "
        "Quote the relevant sentence from that article. "
        "Do not answer from a general Joe Biden biography."
    ]





# %%
def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Wikipedia Retrieval Engine using BM25, vector, or hybrid retrieval."
        )
    )
    parser.add_argument(
        "retriever",
        choices=["bm25", "vector", "hybrid"],
        help="Retrieval method to use.",
    )
    args = parser.parse_args()

    # Chroma is needed only for vector and hybrid retrieval.
    db = None
    if args.retriever in ("vector", "hybrid"):
        db = build_or_load_db(CHROMA_DIR)

    retriever = make_retriever(args.retriever, db)
    queries = my_representative_queries()

    print(
        f"\nCheckpoint 2.1 — {args.retriever.upper()} retrieval "
        f"| scenario: {SCENARIO}\n"
    )

    for i, query in enumerate(queries, 1):
        hits = retriever.getTopK(query, TOP_K)

        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print("\nRetrieved results:")

        for rank, (source, content, score) in enumerate(hits, start=1):
            display_source = source.replace("::chunk_", " [chunk ") + ("]" if "::chunk_" in source else "")
            if args.retriever == "vector":
                print(
                    f"    {rank}. {display_source} | "
                    f"vector distance = {score:.4f}"
                )
            elif args.retriever == "bm25":
                print(
                    f"    {rank}. {display_source} | "
                    f"BM25 score = {score:.4f}"
                )
            else:
                print(
                    f"    {rank}. {display_source} | "
                    f"hybrid score = {score:.4f}"
                )

        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue

        ans = retriever.query(query)
        print(f"\nAnswer:\n{ans}\n")

        hit_summary = [
            (source, score)
            for source, _, score in hits
        ]

        log(
            f"{args.retriever.upper()} | QUERY {i}: {query}",
            f"retrieved={hit_summary}\nanswer={ans}",
        )

    print("=" * 72)
    print(f"Done. Retrieval method used: {args.retriever.upper()}")


if __name__ == "__main__":
    run()

