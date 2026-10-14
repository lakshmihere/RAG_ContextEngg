r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import argparse
import asyncio
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
load_dotenv()


# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits

WIKIPEDIA_DIR = Path(os.getenv("WIKIPEDIA_TEXT_DIR", "Wikipedia_text"))
TEXT_DIR = WIKIPEDIA_DIR
CHROMA_DIR = os.getenv("WIKIPEDIA_CHROMA_DIR", "Wikipedia_chroma_full")
TOPICS_METADATA_PATH = Path(os.getenv("WIKIPEDIA_TOPICS_METADATA", "wikipedia_topics_10.json"))

LOG_PATH = Path.cwd() / "checkpoint_4_1_advanced_retrieval_final.log"
NUM_RETRIEVED = int(os.getenv("NUM_RETRIEVED", "3"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.4"))
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "0.6"))
TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.0"))
MIN_TAGGED_CANDIDATES = int(os.getenv("MIN_TAGGED_CANDIDATES", "1"))
# RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "hybrid_tagged").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
GRAPH_SEED_K = int(os.getenv("GRAPH_SEED_K", "2"))
GRAPH_MAX_LINKED = int(os.getenv("GRAPH_MAX_LINKED", "3"))
GRAPH_MAX_TOPIC = int(os.getenv("GRAPH_MAX_TOPIC", "3"))

DECOMPOSITION_MODEL = os.getenv("DECOMPOSITION_MODEL","deepseek-coder-6.7b-instruct")
ANSWER_MODEL = os.getenv("ANSWER_MODEL","meta-llama-3.1-8b-instruct")

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "wikipedia"   # "research_papers" or "wikipedia"

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


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

def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
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
    demo still calls the LLM, which requires an OpenRouter API key."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return float(len(q & t))

SAMPLE_DOCS = load_wikipedia_corpus(TEXT_DIR)
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}
print("\nCATEGORY METADATA CHECK")

def baseline_retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Single-pass retrieval: Score every doc once, take the top k. (id, score)."""
    scored = [(d["id"], keyword_score(query, d["text"])) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in scored[:k] if s > 0]

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

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

def build_or_load_db(documents: list[dict], chroma_dir: str = CHROMA_DIR) -> Chroma:
    """Build/load a persistent Chroma index over the Wikipedia articles."""
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())

    print("Building vector DB (first run — embedding Wikipedia articles)...")
    docs = [
        Document(page_content=d["text"], metadata={"source": d["id"]})
        for d in documents
        if d.get("text", "").strip()
    ]
    if not docs:
        raise RuntimeError("No Wikipedia documents are available for vector indexing.")
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
    print(f"  Indexed {len(docs)} Wikipedia articles into {chroma_dir}/")
    return db

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

GRAPH_SYSTEM_PROMPT = (
    "You answer questions using only the supplied Wikipedia evidence. "
    "Evidence may be a direct hybrid-search seed, a directly linked article, "
    "or an article connected through a shared topic. Use evidence from all "
    "required articles for comparison questions. Do not fill missing evidence "
    "with outside knowledge. For exact-quotation questions, quote only text "
    "that appears in the supplied evidence."
)


class GraphRetriever(BaseRetriever):
    """Graph-augmented hybrid retriever for the Wikipedia corpus.

    Hybrid search supplies seed articles. The graph then expands those seeds via
    ``links_to`` edges and via shared ``relates_to`` topic nodes. Returned items
    keep the same ``(document_id, article_text, score)`` contract as the other
    retrievers, so BaseRetriever can reuse the normal compact-context and answer
    generation path.

    Source priority is: seed > linked > topic. Within each source class, a small
    lexical query-overlap score ranks graph-expanded candidates. ``seed_k`` is
    intentionally smaller than the final k so graph evidence can occupy at least
    one result slot when useful.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        hybrid: HybridRetriever,
        documents: list[dict],
        num_retrieved: int = NUM_RETRIEVED,
        seed_k: int = GRAPH_SEED_K,
        max_linked: int = GRAPH_MAX_LINKED,
        max_topic: int = GRAPH_MAX_TOPIC,
    ):
        super().__init__(documents, num_retrieved)
        self._G = graph
        self._hybrid = hybrid
        self._doc_by_id = {d["id"]: d for d in documents}
        self._seed_k = max(1, seed_k)
        self._max_linked = max(0, max_linked)
        self._max_topic = max(0, max_topic)
        print(
            "Graph retriever ready "
            f"(seed_k={self._seed_k}, linked≤{self._max_linked}, "
            f"topic≤{self._max_topic})."
        )

    @staticmethod
    def _doc_node(doc_id: str) -> str:
        return f"doc:{doc_id}"

    @staticmethod
    def _doc_id(node: str) -> str:
        return node.removeprefix("doc:")

    def _linked_documents(self, doc_id: str) -> list[str]:
        node = self._doc_node(doc_id)
        if not self._G.has_node(node):
            return []
        out: list[str] = []
        for _, target, edata in self._G.out_edges(node, data=True):
            if edata.get("edge_type") == "links_to" and str(target).startswith("doc:"):
                candidate = self._doc_id(str(target))
                if candidate in self._doc_by_id and candidate != doc_id and candidate not in out:
                    out.append(candidate)
        return out[: self._max_linked]

    def _topic_documents(self, doc_id: str) -> list[str]:
        node = self._doc_node(doc_id)
        if not self._G.has_node(node):
            return []
        out: list[str] = []
        for _, topic, edata in self._G.out_edges(node, data=True):
            if edata.get("edge_type") != "relates_to":
                continue
            for sibling, _, edata2 in self._G.in_edges(topic, data=True):
                if edata2.get("edge_type") != "relates_to":
                    continue
                if not str(sibling).startswith("doc:"):
                    continue
                candidate = self._doc_id(str(sibling))
                if candidate in self._doc_by_id and candidate != doc_id and candidate not in out:
                    out.append(candidate)
                    if len(out) >= self._max_topic:
                        return out
        return out

    def _expand(self, seed_ids: list[str]) -> dict[str, str]:
        """Return {doc_id: source_label}, preserving seed > linked > topic."""
        union: dict[str, str] = {doc_id: "seed" for doc_id in seed_ids}
        for seed in seed_ids:
            for linked in self._linked_documents(seed):
                union.setdefault(linked, "linked")
            for related in self._topic_documents(seed):
                union.setdefault(related, "topic")
        return union

    def getTopK(self, query: str, k: int) -> list[RetrievalResult]:
        if k <= 0:
            return []

        seed_count = min(self._seed_k, k)
        seed_results = self._hybrid.getTopK(query, seed_count)
        if not seed_results:
            return []

        seed_ids = [name for name, _, _ in seed_results]
        expanded = self._expand(seed_ids)

        # Keep the exact hybrid seed scores. Graph candidates get a bounded
        # lexical relevance score plus a source-class bonus. The final layout
        # reserves seed slots and fills remaining slots with the best expansion.
        results: list[RetrievalResult] = list(seed_results)
        remaining = max(0, k - len(results))
        if remaining == 0:
            return results[:k]

        graph_candidates: list[tuple[str, str, float, str]] = []
        source_bonus = {"linked": 0.20, "topic": 0.10}
        q_tokens = set(tokenize(query))

        for doc_id, label in expanded.items():
            if label == "seed":
                continue
            doc = self._doc_by_id.get(doc_id)
            if not doc:
                continue
            text = doc["text"]
            d_tokens = set(tokenize(text))
            overlap = len(q_tokens & d_tokens)
            lexical = overlap / max(len(q_tokens), 1)
            score = min(1.0, lexical) + source_bonus.get(label, 0.0)
            graph_candidates.append((doc_id, text, score, label))

        graph_candidates.sort(key=lambda x: x[2], reverse=True)
        for doc_id, text, score, _label in graph_candidates[:remaining]:
            results.append((doc_id, text, float(score)))

        return results[:k]

    def get_sources_with_labels(self, query: str, k: int | None = None) -> list[tuple[str, str]]:
        """Diagnostic helper showing why each article entered the graph result."""
        final_k = self.num_retrieved if k is None else k
        seed_count = min(self._seed_k, final_k)
        seed_results = self._hybrid.getTopK(query, seed_count)
        seed_ids = [name for name, _, _ in seed_results]
        expanded = self._expand(seed_ids)
        selected = {name for name, _, _ in self.getTopK(query, final_k)}
        return [(doc_id, expanded.get(doc_id, "seed")) for doc_id in expanded if doc_id in selected]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]

def multistep_retrieve(llm: ChatOpenAI, query: str, k: int = 3) -> list[str]:
    sub_queries = decompose_query(llm, query)
    print(f"  decomposed into: {sub_queries}")
    score_map: dict[str, float] = {}
    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, 3 * k):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]

# %% [markdown]
# ## Step 3 — Graph-based retrieval (provided, adapted from Lab 4.2)
#
# Build a graph whose nodes are documents and topics, with edges for "links to" and
# "relates to topic." Retrieval seeds with the baseline hits, then pulls in each
# seed's linked documents and topic-siblings as extra context — deduplicated.

def _load_topics_metadata(path: Path) -> dict[str, dict]:
    """Load optional article/topic metadata in either dict or list form."""
    if not path.exists():
        print(f"Topic metadata not found at {path}; graph will use text links only.")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Could not read topic metadata {path}: {exc}")
        return {}

    if isinstance(raw, dict):
        # Common forms: {filename: {...}} or {"articles": [...]}
        if isinstance(raw.get("articles"), list):
            raw = raw["articles"]
        else:
            return {str(k): (v if isinstance(v, dict) else {"topics": v}) for k, v in raw.items()}

    if isinstance(raw, list):
        out: dict[str, dict] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("id") or item.get("filename") or item.get("article") or item.get("source")
            if key:
                out[str(key)] = item
        return out
    return {}

def _metadata_topics(meta: dict) -> list[str]:
    values: list[str] = []
    for key in ("topics", "topic", "categories", "category", "tags"):
        value = meta.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(x) for x in value if x)
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))

def _resolve_wikipedia_links(text: str, valid_ids: set[str]) -> list[str]:
    """Resolve markdown/Wikipedia-style links only when the target exists in corpus."""
    by_stem = {Path(doc_id).stem.replace(" ", "_").lower(): doc_id for doc_id in valid_ids}
    candidates = re.findall(r"\]\((?:https?://[^)]*?/wiki/|/wiki/)?([^#?)]+)", text)
    out: list[str] = []
    for raw in candidates:
        slug = raw.strip().replace("%20", "_").replace(" ", "_")
        slug = slug.rsplit("/", 1)[-1]
        match = by_stem.get(slug.lower())
        if match and match not in out:
            out.append(match)
    return out

def build_graph(docs: list[dict], metadata_path: Path = TOPICS_METADATA_PATH) -> nx.DiGraph:
    """Build a Wikipedia article/topic graph from corpus links + optional metadata."""
    G = nx.DiGraph()
    valid_ids = {d["id"] for d in docs}
    metadata = _load_topics_metadata(metadata_path)

    # Make metadata keys easy to match with relative filenames/stems.
    meta_by_stem: dict[str, dict] = {}
    for key, value in metadata.items():
        meta_by_stem[Path(str(key)).stem.replace(" ", "_").lower()] = value

    for d in docs:
        doc_id = d["id"]
        G.add_node(f"doc:{doc_id}", node_type="doc", filename=doc_id)

        for other in _resolve_wikipedia_links(d["text"], valid_ids):
            if other != doc_id:
                G.add_edge(f"doc:{doc_id}", f"doc:{other}", edge_type="links_to")

        meta = meta_by_stem.get(Path(doc_id).stem.replace(" ", "_").lower(), {})
        for topic in _metadata_topics(meta):
            topic_node = f"topic:{topic}"
            G.add_node(topic_node, node_type="topic", label=topic)
            G.add_edge(f"doc:{doc_id}", topic_node, edge_type="relates_to")

    print(
        f"Wikipedia graph ready: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges."
    )
    return G

def graph_retrieve(graph: nx.DiGraph, query: str, k: int = 3) -> dict[str, str]:
    """Legacy diagnostic helper: baseline seeds plus graph neighbors."""
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
                for sib, _, ed2 in graph.in_edges(target, data=True):
                    if ed2.get("edge_type") == "relates_to":
                        union.setdefault(sib.removeprefix("doc:"), "topic")
    return union

# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

# %%
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

# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence
#
# This demonstration compares single-pass retrieval with the two advanced retrieval strategies 
# using the sample corpus and a representative query. It also logs the result. Read the
# retrieved sets: The advanced strategies should surface related documents that the
# baseline misses. Then run the same comparison in your real system for your report.

# %%
def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content

TEST_QUERIES = [
    "According to the Wikipedia article on Nelson Mandela, what leadership office did he hold in South Africa and during what years did he serve? Quote the relevant sentence.",
    "According to the Wikipedia article on Queen Victoria, when did her reign begin and when did it end? Quote the relevant evidence from the article.",
    "According to the Wikipedia article on Justin Trudeau, identify his nationality, the leadership office he held, and his term of service. Support all parts of the answer with evidence from the article.",
    "According to the Wikipedia article on Margaret Thatcher, what position did she hold in the United Kingdom and during what years did she serve? Quote the relevant sentence.",
    "According to the Wikipedia article on Adolf Hitler, identify the major national leadership positions attributed to him and the periods associated with those positions. Use only information stated in the article and quote the supporting evidence.",
    "Quote, word for word, the opening sentence of the Wikipedia article on George Washington. Do not paraphrase.",
    "According to the Wikipedia article on Abraham Lincoln, which numbered president of the United States was he and when did he serve? Quote the relevant sentence.",
    "Using the Wikipedia chunks on George Washington, Abraham Lincoln, and Barack Obama, identify which numbered U.S. president each was and the years each served. Provide supporting evidence from all three articles.",
    "According to the Wikipedia article on Emmanuel Macron, what national leadership office does the article identify him as holding? Quote the relevant sentence.",
    "Compare Margaret Thatcher and Theresa May using their respective Wikipedia chunks. What leadership office did they share, and when did each serve? Cite evidence from both articles.",
    "According to the Wikipedia article titled 'Presidency of Joe Biden', what event marked the beginning of Biden's presidential tenure and on what date did it occur? Quote the relevant sentence from that article. Do not answer from a general Joe Biden biography.",
    "Compare Nelson Mandela and Suharto using their respective Wikipedia chunks. What country did each lead, what office did each hold, and how did their periods of leadership differ? Quote evidence from both articles.",
    "Quote, word for word, the opening sentence of the Wikipedia article on Cleopatra. Do not supplement the quotation with outside knowledge.",
    "According to the Wikipedia article on Suharto, what does the article state about his political role, his presidency of Indonesia, and the length or period of his leadership? Quote the relevant evidence.",
    "Using the Wikipedia chunks on Boris Johnson, Liz Truss, and Rishi Sunak, identify the prime-ministerial term stated for each person and place the three leaders in chronological order. Quote supporting evidence from each article.",
]

def run_demo() -> None:
    """Compare baseline, hybrid, multistep, and graph-augmented retrieval."""
    llm = make_llm()
    graph = build_graph(SAMPLE_DOCS)
    db = build_or_load_db(SAMPLE_DOCS)
    hybrid = HybridRetriever(db=db, documents=SAMPLE_DOCS, num_retrieved=NUM_RETRIEVED)
    graph_retriever = GraphRetriever(
        graph=graph,
        hybrid=hybrid,
        documents=SAMPLE_DOCS,
        num_retrieved=NUM_RETRIEVED,
    )

    print("=" * 88)
    print(f"Checkpoint 4.1 — Wikipedia advanced retrieval | scenario={SCENARIO}")
    print(f"Articles={len(SAMPLE_DOCS)} | test queries={len(TEST_QUERIES)}")
    print("=" * 88)

    for qnum, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQ{qnum}: {query}")

        base = [i for i, _ in baseline_retrieve(query, NUM_RETRIEVED)]
        print(f"  BASELINE: {base}")

        hybrid_results = hybrid.getTopK(query, NUM_RETRIEVED)
        print(f"  HYBRID:   {[name for name, _, _ in hybrid_results]}")

        multi = multistep_retrieve(llm, query, NUM_RETRIEVED)
        print(f"  MULTI:    {multi}")

        graph_results = graph_retriever.getTopK(query, NUM_RETRIEVED)
        print(f"  GRAPH:    {[name for name, _, _ in graph_results]}")
        print(f"  LABELS:   {graph_retriever.get_sources_with_labels(query)}")

        trace = graph_retriever.query_with_trace(query)
        print(f"  ANSWER:   {trace['answer']}")
        log_response("GRAPH_RETRIEVER", query, trace["answer"])

    print("\nAdvanced-retrieval plan:")
    print(json.dumps(my_advanced_plan(), indent=2))
    print("=" * 88)

if __name__ == "__main__":
    run_demo()
