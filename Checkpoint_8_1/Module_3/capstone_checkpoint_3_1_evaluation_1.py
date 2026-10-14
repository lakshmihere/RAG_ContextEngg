r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import json
import os
import re

import argparse
import asyncio
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from rank_bm25 import BM25Okapi
from ragas import Dataset, experiment

load_dotenv()

SCENARIO = "wikipedia"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
JUDGE_MODEL = "openai/gpt-5.4"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

CHROMA_DIR = "wikipedia_chroma_full"

# NUM_RETRIEVED = 6
# CANDIDATE_POOL = 20
# TOP_K = 3

NUM_RETRIEVED = 10
CANDIDATE_POOL = 40
TOP_K = 6

WEIGHT_BM25 = 0.40
WEIGHT_VECTOR = 0.60

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


TEMPERATURE = 0.0
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation_1_MOD.log"
VARIANTS_PATH = Path("nifty_minsky_variants.json")
# VARIANTS_PATH = Path("checkpoint_3_1_hybrid.json")

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)

# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
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

def load_wikipedia_corpus(text_dir: Path) -> list[dict]:
    """
    Load every .txt file in the Wikipedia corpus directory.

    Returns:
        A list of dictionaries:
        {
            "id": filename,
            "text": article text
        }
    """

    text_files = sorted(text_dir.glob("*.txt"))

    print(f"Found {len(text_files)} Wikipedia text files.")

    documents = []

    for file_path in text_files:
        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            # Skip empty documents
            if not text.strip():
                print(f"WARNING: Empty document: {file_path.name}")
                continue

            documents.append(
                {
                    "id": file_path.name,
                    "text": text.strip(),
                }
            )

        except Exception as e:
            print(
                f"WARNING: Could not load {file_path.name}: {e}"
            )

    print(
        f"Successfully loaded {len(documents)} "
        f"Wikipedia documents."
    )

    # Show only a small sample
    for doc in documents[:10]:
        print(
            f"{doc['id']:<40} "
            f"{len(doc['text']):>8} characters"
        )

    if len(documents) > 10:
        print(
            f"... plus {len(documents) - 10} "
            f"additional documents."
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

    # ---------------------------------------------------------
    # Load existing Chroma DB if already built
    # ---------------------------------------------------------
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing chunk vector DB from {chroma_dir}/")

        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(),
        )

    # ---------------------------------------------------------
    # Otherwise build the vector DB
    # ---------------------------------------------------------
    print(
        "Building vector DB "
        "(first run — embedding Wikipedia chunks)..."
    )

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

    # ---------------------------------------------------------
    # Safety check — ONLY raise error if docs is actually empty
    # ---------------------------------------------------------
    if not docs:
        raise RuntimeError(
            f"No Wikipedia chunks were created. "
            f"Check TEXT_DIR={TEXT_DIR!s}."
        )

    print(
        f"Preparing to embed {len(docs)} chunks "
        f"from {len(SAMPLE_DOCS)} Wikipedia articles..."
    )

    # ---------------------------------------------------------
    # Build persistent Chroma database
    # ---------------------------------------------------------
    db = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=chroma_dir,
    )

    print(
        f"Indexed {len(docs)} Wikipedia chunks from "
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

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    q = _tokens(query)
    scored = [(d["id"], float(len(q & _tokens(d["text"])))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]

def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content

def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"

# def my_eval_set() -> list[dict]:
#     """
#     Evaluation set for the 65-article Wikipedia Leader Lens corpus.

#     The questions are intentionally identical to the 15 prompts in
#     my_representative_queries() so the same evaluation set can be used
#     across:
#         - no-retrieval baseline,
#         - BM25 retrieval,
#         - vector retrieval,
#         - hybrid retrieval.

#     Each item contains:
#         - question: the evaluation prompt
#         - grading_notes: expected evidence/content for judging the answer
#     """

#     return [

#         # =========================================================
#         # 1-5. SINGLE-DOCUMENT FACTUAL RETRIEVAL
#         # =========================================================

#         # 1. Nelson Mandela
#         {
#             "question": (
#                 "According to the Wikipedia article on Nelson Mandela, "
#                 "what leadership office did he hold in South Africa and during "
#                 "what years did he serve? Quote the relevant sentence."
#             ),
#             "grading_notes": (
#                 "Nelson Mandela served as President of "
#                 "South Africa from 1994 to 1999. The response should include "
#                 "supporting evidence or a relevant quotation from the "
#                 "Nelson Mandela article."
#             ),
#         },

#         # 2. Margaret Thatcher
#         {
#             "question": (
#                 "According to the Wikipedia article on Margaret Thatcher, "
#                 "what position did she hold in the United Kingdom and during "
#                 "what years did she serve? Quote the relevant sentence."
#             ),
#             "grading_notes": (
#                 "Margaret Thatcher served as Prime Minister of the United "
#                 "Kingdom from 1979 to 1990. The response should include "
#                 "supporting evidence or a relevant quotation from the "
#                 "Margaret Thatcher article."
#             ),
#         },

#         # 3. Queen Victoria
#         {
#             "question": (
#                 "According to the Wikipedia article on Queen Victoria, "
#                 "when did her reign begin and when did it end? "
#                 "Quote the relevant evidence from the article."
#             ),
#             "grading_notes": (
#                 "Queen Victoria's reign began on 20 June 1837 and ended "
#                 "on 22 January 1901. The response should include supporting "
#                 "evidence or a relevant quotation from the Queen Victoria "
#                 "article."
#             ),
#         },

#         # 4. Abraham Lincoln
#         {
#             "question": (
#                 "According to the Wikipedia article on Abraham Lincoln, "
#                 "which numbered president of the United States was he and "
#                 "when did he serve? Quote the relevant sentence."
#             ),
#             "grading_notes": (
#                 "Abraham Lincoln was the 16th president of the United States "
#                 "and served from 1861 until his assassination in 1865. "
#                 "The response should include supporting evidence or a relevant "
#                 "quotation from the Abraham Lincoln article."
#             ),
#         },

#         # 5. Emmanuel Macron
#         {
#             "question": (
#                 "According to the Wikipedia article on Emmanuel Macron, "
#                 "what national leadership office does the article identify him "
#                 "as holding? Quote the relevant sentence."
#             ),
#             "grading_notes": (
#                 "Emmanuel Macron is identified as President of France. "
#                 "The response should include supporting evidence or a relevant "
#                 "quotation from the Emmanuel Macron article."
#             ),
#         },


#         # =========================================================
#         # 6-8. MULTI-FACT RETRIEVAL
#         # =========================================================

#         # 6. Justin Trudeau
#         {
#             "question": (
#                 "According to the Wikipedia article on Justin Trudeau, "
#                 "identify his nationality, the leadership office he held, "
#                 "and his term of service. Support all parts of the answer "
#                 "with evidence from the article."
#             ),
#             "grading_notes": (
#                 "Justin Trudeau is Canadian and served as the 23rd Prime "
#                 "Minister of Canada from 2015 to 2025. The response should "
#                 "support his nationality, leadership office, and term of "
#                 "service with evidence from the Justin Trudeau article."
#             ),
#         },

#         # 7. Suharto
#         {
#             "question": (
#                 "According to the Wikipedia article on Suharto, what does "
#                 "the article state about his political role, his presidency "
#                 "of Indonesia, and the length or period of his leadership? "
#                 "Quote the relevant evidence."
#             ),
#             "grading_notes": (
#                 "Suharto was an Indonesian military officer and politician "
#                 "who served as the second and longest-serving President of "
#                 "Indonesia from 1967 to 1998. The response should accurately "
#                 "describe his political role, presidency, and leadership "
#                 "period using evidence from the Suharto article."
#             ),
#         },

#         # 8. Adolf Hitler
#         {
#             "question": (
#                 "According to the Wikipedia article on Adolf Hitler, "
#                 "identify the major national leadership positions attributed "
#                 "to him and the periods associated with those positions. "
#                 "Use only information stated in the article and quote the "
#                 "supporting evidence."
#             ),
#             "grading_notes": (
#                 "The Adolf Hitler article identifies major national leadership "
#                 "positions including Chancellor of Germany and Führer of "
#                 "Germany, together with their corresponding periods. "
#                 "The response should accurately describe these positions "
#                 "and periods using only evidence retrieved from the "
#                 "Adolf Hitler article."
#             ),
#         },


#         # =========================================================
#         # 9-11. CROSS-DOCUMENT / MULTI-HOP RETRIEVAL
#         # =========================================================

#         # 9. Mandela vs. Suharto
#         {
#             "question": (
#                 "Compare Nelson Mandela and Suharto using their respective "
#                 "Wikipedia chunks. What country did each lead, what office "
#                 "did each hold, and how did their periods of leadership differ? "
#                 "Quote evidence from both articles."
#             ),
#             "grading_notes": (
#                 "Nelson Mandela served as President of South Africa from "
#                 "1994 to 1999, while Suharto served as President of Indonesia "
#                 "from 1967 to 1998. The response should clearly compare their "
#                 "countries, offices, and leadership periods using evidence "
#                 "from both the Nelson Mandela and Suharto articles. If evidence "
#                 "from one required article is unavailable, the response should "
#                 "not invent the missing information."
#             ),
#         },

#         # 10. Thatcher vs. May
#         {
#             "question": (
#                 "Compare Margaret Thatcher and Theresa May using their "
#                 "respective Wikipedia chunks. What leadership office did "
#                 "they share, and when did each serve? Cite evidence from "
#                 "both articles."
#             ),
#             "grading_notes": (
#                 "Margaret Thatcher and Theresa May both served as Prime "
#                 "Minister of the United Kingdom. Thatcher served from "
#                 "1979 to 1990 and May served from 2016 to 2019. "
#                 "The response should compare their terms and support the "
#                 "information with evidence from both articles."
#             ),
#         },

#         # 11. Three U.S. presidents
#         {
#             "question": (
#                 "Using the Wikipedia chunks on George Washington, "
#                 "Abraham Lincoln, and Barack Obama, identify which numbered "
#                 "U.S. president each was and the years each served. "
#                 "Provide supporting evidence from all three articles."
#             ),
#             "grading_notes": (
#                 "George Washington was the 1st U.S. president and served "
#                 "from 1789 to 1797, Abraham Lincoln was the 16th and served "
#                 "from 1861 to 1865, and Barack Obama was the 44th and served "
#                 "from 2009 to 2017. The response should support these facts "
#                 "with evidence from all three requested articles rather than "
#                 "unrelated presidential timeline chunks."
#             ),
#         },


#         # =========================================================
#         # 12-13. EXACT-EVIDENCE / QUOTATION RETRIEVAL
#         # =========================================================

#         # 12. George Washington
#         {
#             "question": (
#                 "Quote, word for word, the opening sentence of the "
#                 "Wikipedia article on George Washington. Do not paraphrase."
#             ),
#             "grading_notes": (
#                 "The response should retrieve the opening passage of the "
#                 "George Washington article and reproduce the opening sentence "
#                 "word for word. The sentence should not be paraphrased or "
#                 "supplemented with unsupported material."
#             ),
#         },

#         # 13. Cleopatra
#         {
#             "question": (
#                 "Quote, word for word, the opening sentence of the "
#                 "Wikipedia article on Cleopatra. Do not supplement the "
#                 "quotation with outside knowledge."
#             ),
#             "grading_notes": (
#                 "The response should retrieve the opening passage of the "
#                 "Cleopatra article and reproduce the opening sentence exactly. "
#                 "If the retrieved evidence does not contain the opening "
#                 "sentence, the response should explicitly state that the "
#                 "requested evidence is unavailable rather than inventing or "
#                 "reconstructing it from outside knowledge."
#             ),
#         },


#         # =========================================================
#         # 14. TEMPORAL / LEADERSHIP COMPARISON
#         # =========================================================

#         # 14. UK prime ministers
#         {
#             "question": (
#                 "Using the Wikipedia chunks on Boris Johnson, Liz Truss, "
#                 "and Rishi Sunak, identify the prime-ministerial term stated "
#                 "for each person and place the three leaders in chronological "
#                 "order. Quote supporting evidence from each article."
#             ),
#             "grading_notes": (
#                 "Boris Johnson served from 24 July 2019 to 6 September 2022, "
#                 "Liz Truss served from 6 September 2022 to 25 October 2022, "
#                 "and Rishi Sunak served from 25 October 2022 to 5 July 2024. "
#                 "The chronological order is Boris Johnson -> Liz Truss -> "
#                 "Rishi Sunak. The response should support the terms and order "
#                 "with evidence from all three articles."
#             ),
#         },


#         # =========================================================
#         # 15. POLITICAL-PERIOD ARTICLE RETRIEVAL
#         # =========================================================

#         # 15. Presidency of Joe Biden
#         {
#             "question": (
#                 "According to the Wikipedia article titled "
#                 "'Presidency of Joe Biden', what event marked the beginning "
#                 "of Biden's presidential tenure and on what date did it occur? "
#                 "Quote the relevant sentence from that article. "
#                 "Do not answer from a general Joe Biden biography."
#             ),
#             "grading_notes": (
#                 "The 'Presidency of Joe Biden' article states that Biden's "
#                 "tenure as the 46th president began with his inauguration "
#                 "on January 20, 2021. The response should quote or clearly "
#                 "cite supporting evidence specifically from the political-period "
#                 "article rather than relying on a general Joe Biden biography."
#             ),
#         },
#     ]

def my_eval_set() -> list[dict]:
    """
    Advanced evaluation set for the 65-article Wikipedia Leader Lens corpus.

    These 15 questions are aligned with the advanced agentic probes in
    new_queries.txt.

    Evaluation goals:
        - context / graph expansion
        - multi-fact retrieval and query refinement
        - cross-article query decomposition
        - temporal multi-article reasoning
        - targeted retrieval + neighboring context
        - article-type / biography cross-reference
        - high-complexity multi-hop reasoning

    Each item contains:
        - question
        - grading_notes
    """

    return [

        # =========================================================
        # 1-3. CONTEXT EXPANSION / GRAPH EXPANSION
        # =========================================================

        # 1. Nelson Mandela
        {
            "question": (
                "Using the Wikipedia article on Nelson Mandela, identify when "
                "he became President of South Africa and explain what the article "
                "says about the political transition surrounding his presidency. "
                "Support both the office/date and the surrounding political context "
                "with passage evidence."
            ),
            "grading_notes": (
                "The response should identify Nelson Mandela as President of "
                "South Africa beginning in 1994 and describe the political "
                "transition surrounding his presidency using evidence from the "
                "Nelson Mandela article. Both the leadership fact and surrounding "
                "political context must be supported by retrieved evidence."
            ),
        },

        # 2. Queen Victoria
        {
            "question": (
                "According to the Wikipedia article on Queen Victoria, identify "
                "when her reign began and explain the circumstances described "
                "around her accession to the throne. Use evidence for both the "
                "date and the surrounding context."
            ),
            "grading_notes": (
                "The response should identify the beginning of Queen Victoria's "
                "reign as 20 June 1837 and explain the circumstances surrounding "
                "her accession as described in the Queen Victoria article. "
                "Evidence should support both the accession date and context."
            ),
        },

        # 3. Abraham Lincoln
        {
            "question": (
                "Using the Wikipedia article on Abraham Lincoln, identify when "
                "his presidency began and explain the major national crisis "
                "described in the article around the beginning of his presidency. "
                "Support the answer with evidence from the relevant and surrounding "
                "passages."
            ),
            "grading_notes": (
                "The response should identify the beginning of Abraham Lincoln's "
                "presidency in 1861 and explain the major national crisis associated "
                "with the beginning of his presidency, particularly the secession "
                "crisis and/or beginning of the American Civil War, using evidence "
                "from the Abraham Lincoln article."
            ),
        },

        # =========================================================
        # 4-6. MULTI-FACT / QUERY-REFINEMENT TESTS
        # =========================================================

        # 4. Justin Trudeau
        {
            "question": (
                "Using only the Justin Trudeau article, determine his numbered "
                "position among Canadian prime ministers, the years he served, "
                "the political party he led, and what the article says about the "
                "end of his premiership. Retrieve additional evidence if the "
                "initial passages do not support every part of the answer."
            ),
            "grading_notes": (
                "The response should identify Justin Trudeau as Canada's 23rd "
                "prime minister, give his term as 2015 to 2025, identify the "
                "Liberal Party, and accurately describe the end of his premiership "
                "using evidence from the Justin Trudeau article. Every requested "
                "fact should be grounded in retrieved evidence."
            ),
        },

        # 5. Adolf Hitler
        {
            "question": (
                "From the Adolf Hitler article, identify when he became Chancellor "
                "of Germany, explain how his national leadership position changed "
                "after Paul von Hindenburg's death, and identify the title associated "
                "with that later position. Cite evidence supporting each stage of "
                "the leadership transition."
            ),
            "grading_notes": (
                "The response should identify Hitler becoming Chancellor of Germany "
                "in 1933 and explain the change in his leadership position following "
                "Paul von Hindenburg's death, including the Führer title or merger "
                "of presidential and chancellorial authority as supported by the "
                "Adolf Hitler article. Each stage must be grounded in evidence."
            ),
        },

        # 6. Suharto
        {
            "question": (
                "Using the Suharto article, identify how Suharto came to national "
                "leadership in Indonesia, the period during which he served as "
                "president, and how the article describes the end of his rule. "
                "Use multiple passages if these facts are not located together."
            ),
            "grading_notes": (
                "The response should explain Suharto's rise to national leadership, "
                "identify his presidency of Indonesia from 1967 to 1998, and "
                "describe the end of his rule using evidence from the Suharto "
                "article. The response should not invent details absent from the "
                "retrieved evidence."
            ),
        },

        # =========================================================
        # 7-9. CROSS-ARTICLE / QUERY-DECOMPOSITION TESTS
        # =========================================================

        # 7. Mandela vs. Suharto
        {
            "question": (
                "Compare Nelson Mandela and Suharto as national leaders. For each "
                "person, determine the country led, leadership office, period in "
                "office, and the political circumstances associated with the "
                "beginning of that leadership. Support the comparison with evidence "
                "from both Wikipedia articles."
            ),
            "grading_notes": (
                "The response should identify Mandela as President of South Africa "
                "from 1994 to 1999 and Suharto as President of Indonesia from "
                "1967 to 1998. It should also explain the political circumstances "
                "associated with each leader's rise using evidence from both "
                "articles. Evidence from both required articles is necessary."
            ),
        },

        # 8. Thatcher vs. May
        {
            "question": (
                "Compare Margaret Thatcher and Theresa May as British prime "
                "ministers. Determine when each entered office, when each left "
                "office, and what political event or circumstance described in "
                "their respective articles was associated with the end of each "
                "premiership. Cite evidence from both articles."
            ),
            "grading_notes": (
                "The response should identify Margaret Thatcher's premiership as "
                "1979 to 1990 and Theresa May's as 2016 to 2019, and explain the "
                "political circumstances associated with the end of each premiership "
                "using evidence from their respective articles. Evidence from both "
                "articles must be present."
            ),
        },

        # 9. Washington, Lincoln, Obama
        {
            "question": (
                "Using the George Washington, Abraham Lincoln, and Barack Obama "
                "articles, determine which numbered president each was, the years "
                "each served, and one major national circumstance associated with "
                "the beginning of each presidency as described in the respective "
                "article. Evidence must be retrieved from all three articles."
            ),
            "grading_notes": (
                "The response should identify George Washington as the 1st U.S. "
                "president (1789-1797), Abraham Lincoln as the 16th (1861-1865), "
                "and Barack Obama as the 44th (2009-2017). It should also provide "
                "one relevant national circumstance associated with the beginning "
                "of each presidency using evidence from each of the three articles."
            ),
        },

        # =========================================================
        # 10-11. TEMPORAL MULTI-ARTICLE REASONING
        # =========================================================

        # 10. Johnson -> Truss -> Sunak
        {
            "question": (
                "Using the Wikipedia articles on Boris Johnson, Liz Truss, and "
                "Rishi Sunak, reconstruct the chronological transition of UK prime "
                "ministerial leadership across the three figures. Identify each "
                "person's term and use evidence from the articles to explain how "
                "one premiership gave way to the next."
            ),
            "grading_notes": (
                "The response should establish the chronological order Boris "
                "Johnson -> Liz Truss -> Rishi Sunak and accurately identify their "
                "prime-ministerial terms. It should explain the transitions using "
                "evidence retrieved from all three articles rather than relying "
                "only on general chronology."
            ),
        },

        # 11. Five UK prime ministers
        {
            "question": (
                "Using the Margaret Thatcher, Theresa May, Boris Johnson, "
                "Liz Truss, and Rishi Sunak articles, arrange the five leaders "
                "chronologically by their periods as UK prime minister. Then "
                "identify any gaps between their premierships that show that "
                "another prime minister must have served between members of this "
                "set. Base the dates only on evidence retrieved from the corpus."
            ),
            "grading_notes": (
                "The response should order Margaret Thatcher, Theresa May, Boris "
                "Johnson, Liz Truss, and Rishi Sunak correctly by their periods "
                "as UK prime minister. It should recognize gaps between the listed "
                "terms where the supplied leaders are not consecutive. Claims "
                "about dates and gaps must be supported by corpus evidence."
            ),
        },

        # =========================================================
        # 12-13. TARGETED RETRIEVAL + LOCAL CONTEXT EXPANSION
        # =========================================================

        # 12. George Washington
        {
            "question": (
                "Locate the passage in the George Washington article that "
                "identifies him as the first president of the United States. "
                "Then inspect the surrounding article context and explain what "
                "additional leadership or nation-building role the nearby passages "
                "attribute to Washington. Cite the retrieved and neighboring "
                "passages separately."
            ),
            "grading_notes": (
                "The response should retrieve evidence identifying George "
                "Washington as the first president of the United States and use "
                "neighboring passage evidence to describe an additional leadership "
                "or nation-building role attributed to him. Both the initial "
                "passage and expanded local context should be represented."
            ),
        },

        # 13. Emmanuel Macron
        {
            "question": (
                "Locate evidence in the Emmanuel Macron article identifying his "
                "presidency of France. Then examine nearby passages to determine "
                "what immediately preceding political role or political background "
                "the article associates with his rise to the presidency. Support "
                "both parts using passage evidence."
            ),
            "grading_notes": (
                "The response should retrieve evidence identifying Emmanuel Macron "
                "as President of France and use nearby context to identify the "
                "political role or background preceding his presidency as stated "
                "in the article. Both claims must be supported by passage evidence."
            ),
        },

        # =========================================================
        # 14. ARTICLE-TYPE + BIOGRAPHICAL CROSS-REFERENCE
        # =========================================================

        # 14. Biden / Obama
        {
            "question": (
                "Use the 'Presidency of Joe Biden' article to identify the event "
                "and date that began Biden's presidential tenure. Then compare that "
                "leadership transition with the presidential term information in "
                "the Barack Obama article. Explain the chronological relationship "
                "between the two presidencies using evidence from both articles."
            ),
            "grading_notes": (
                "The response should identify Biden's inauguration on January 20, "
                "2021 as the beginning of his presidential tenure using the "
                "'Presidency of Joe Biden' article. It should also use evidence "
                "from the Barack Obama article for Obama's presidential term and "
                "accurately explain the chronological relationship. Evidence from "
                "both articles is required."
            ),
        },

        # =========================================================
        # 15. HIGH-COMPLEXITY MULTI-HOP AGENT TEST
        # =========================================================

        # 15. Victoria / Mandela / Thatcher / Lincoln
        {
            "question": (
                "Compare Queen Victoria, Nelson Mandela, Margaret Thatcher, and "
                "Abraham Lincoln as national leaders from different political "
                "systems. For each figure, identify the country, leadership role, "
                "and period of leadership stated in the corpus. Then classify each "
                "role as monarch, president, or prime minister and arrange the four "
                "leadership periods chronologically by starting year. Retrieve and "
                "cite evidence from every required article."
            ),
            "grading_notes": (
                "The response should identify Queen Victoria as monarch of the "
                "United Kingdom (reign beginning 1837 and ending 1901), Abraham "
                "Lincoln as President of the United States (1861-1865), Margaret "
                "Thatcher as Prime Minister of the United Kingdom (1979-1990), "
                "and Nelson Mandela as President of South Africa (1994-1999). "
                "The roles should be classified correctly and ordered by starting "
                "year as Victoria -> Lincoln -> Thatcher -> Mandela. Evidence "
                "from all four required articles must be retrieved."
            ),
        },
    ]

def load_variant_eval_set(path: Path = VARIANTS_PATH,) -> list[dict]:
    """
    Load the expanded Checkpoint 3.1 evaluation set containing
    original questions and paraphrase variants.

    Expected fields:
        - question
        - grading_notes
        - question_type
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Variant evaluation file not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Variant evaluation JSON must contain a list of records."
        )

    required_fields = {
        "question",
        "grading_notes",
    }

    for i, item in enumerate(data, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Record {i} is not a dictionary."
            )

        missing = required_fields - set(item)

        if missing:
            raise ValueError(
                f"Record {i} is missing required field(s): "
                + ", ".join(sorted(missing))
            )

        # Preserve original/variant distinction.
        # Older files without this column still work.
        item.setdefault(
            "question_type",
            "unknown",
        )

    print(
        f"Loaded {len(data)} evaluation questions "
        f"from {path.name}"
    )

    return data

def load_dataset(
    use_variants: bool = True,
) -> Dataset:
    """
    Build the local RAGAS dataset.

    use_variants=True:
        Load original + paraphrased questions from
        checkpoint_3_1_hybrid_variants.json.

    use_variants=False:
        Use the original 15 questions from my_eval_set().
    """

    if use_variants:

        samples = load_variant_eval_set()

        dataset_name = (
            "wikipedia_leader_lens_variants_eval"
        )

    else:

        samples = my_eval_set()

        dataset_name = (
            "wikipedia_leader_lens_eval"
        )

    dataset = Dataset(
        name=dataset_name,
        backend="local/csv",
        root_dir="ragas_experiments",
    )

    for sample in samples:

        dataset.append(
            {
                "question": sample["question"],
                "grading_notes": sample["grading_notes"],
                "question_type": sample.get(
                    "question_type",
                    "original",
                ),
            }
        )

    dataset.save()

    return dataset

def build_experiment(retriever: BaseRetriever, kind: str):

    @experiment()
    async def run_experiment(row):

        question = row["question"]

        # ---------------------------------------------------------
        # Retrieve chunks once
        # ---------------------------------------------------------
        results = retriever.getTopK(
            question,
            NUM_RETRIEVED,
        )

        # ---------------------------------------------------------
        # Column 1: source/chunk IDs only
        # Example:
        # Nelson_Mandela.txt::chunk_0;
        # Nelson_Mandela.txt::chunk_15
        # ---------------------------------------------------------
        retrieved_sources = "; ".join(
            name
            for name, content, score in results
        )

        # ---------------------------------------------------------
        # Column 2: actual retrieved text/context
        # ---------------------------------------------------------
        retrieved_context = "\n\n---\n\n".join(
            f"[{name}]\n{content}"
            for name, content, score in results
        )

        # ---------------------------------------------------------
        # Generate grounded answer using the retrieved context
        # ---------------------------------------------------------
        messages = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"Documents:\n{retrieved_context}\n\n"
                    f"Question: {question}"
                )
            ),
        ]

        llm_response = retriever._llm.invoke(messages)

        response = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )

        # ---------------------------------------------------------
        # Judge generated answer
        # ---------------------------------------------------------
        verdict = judge(
            judge_llm,
            response,
            row["grading_notes"],
        )

        # ---------------------------------------------------------
        # RAGAS experiment output row
        # ---------------------------------------------------------
        return {
            **row,
            "retriever": kind,
            "retrieved_sources": retrieved_sources,
            "retrieved_context": retrieved_context,
            "response": response,
            "score": verdict,
        }

    return run_experiment

def run_evaluation(retriever: BaseRetriever, retriever_name: str) -> None:
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

    eval_set = my_eval_set()
    passes = 0

    print(
        f"Checkpoint 3.1 — baseline evaluation  |  "
        f"scenario: {SCENARIO}  |  retriever: {retriever_name}\n"
    )

    for i, item in enumerate(eval_set, 1):

        question = item["question"]

        # Retrieve CHUNKS and generate the grounded answer.
        ans = retriever.query(question)

        verdict = judge(
            judge_llm,
            ans,
            item["grading_notes"],
        )

        passes += verdict == "pass"

        print("=" * 72)
        print(f"Q{i}: {question}")
        print(f"  retriever={retriever_name}")
        print(f"  verdict={verdict.upper()}")
        print(f"  answer: {ans}")

        log(
            f"Q{i}: {question}",
            (
                f"retriever={retriever_name}\n"
                f"verdict={verdict}\n"
                f"answer={ans}"
            ),
        )

    print("=" * 72)
    print(
        f"{retriever_name.upper()} pass rate: "
        f"{passes}/{len(eval_set)}"
    )

def validate_framework() -> None:
    llm = make_llm()
    q = "What is BM25?"
    notes = "States that BM25 is a keyword / term-frequency ranking function for documents."
    good = answer(llm, q, [doc_id for doc_id, _ in retrieve(q)])
    manipulated = "BM25 is a deep neural network that generates images from text prompts."
    good_verdict = judge(llm, good, notes)
    manip_verdict = judge(llm, manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")

async def main() -> None:

    global judge_llm

    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint 3.1 Wikipedia Leader Lens "
            "retrieval and RAGAS evaluation"
        )
    )

    parser.add_argument(
        "retriever",
        choices=["bm25", "vector", "hybrid"],
        help="Retriever to evaluate.",
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("CHECKPOINT 3.1 — WIKIPEDIA LEADER LENS EVALUATION")
    print("=" * 72)
    print(f"Retriever selected : {args.retriever}")
    print(f"Articles           : {len(SAMPLE_DOCS)}")
    print(f"Chunks             : {len(SAMPLE_CHUNKS)}")
    print()

    # ---------------------------------------------------------
    # Judge LLM
    # ---------------------------------------------------------

    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

    # ---------------------------------------------------------
    # Vector DB required only for vector/hybrid
    # ---------------------------------------------------------

    db = None

    if args.retriever in {"vector", "hybrid"}:
        db = build_or_load_db()

    # ---------------------------------------------------------
    # Build selected retriever
    # ---------------------------------------------------------

    retriever = make_retriever(
        kind=args.retriever,
        db=db,
    )

    # ---------------------------------------------------------
    # Standard Checkpoint evaluation
    # ---------------------------------------------------------

    run_evaluation(
        retriever=retriever,
        retriever_name=args.retriever,
    )

    # ---------------------------------------------------------
    # RAGAS experiment
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("RAGAS EXPERIMENT")
    print("=" * 72)

    # dataset = load_dataset()
    dataset = load_dataset(use_variants=True,)

    print(
        f"Loaded {len(dataset)} questions. "
        f"Evaluating with '{args.retriever}' retriever..."
    )

    ragas_experiment = build_experiment(
        retriever,
        args.retriever,
    )

    results = await ragas_experiment.arun(dataset)

    passes = sum(
        1
        for row in results
        if row["score"] == "pass"
    )

    print()
    print(
        f"RAGAS experiment complete: "
        f"{passes}/{len(results)} passed."
    )

    results.save()

    csv_path = (
        Path("ragas_experiments")
        / "experiments"
        / f"{results.name}.csv"
    )

    print(f"RAGAS results saved to: {csv_path.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%

# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# ============================================================
# RAGAS EXPERIMENT
# ============================================================

# def load_dataset() -> Dataset:
#     """
#     Build a local RAGAS dataset from the 15-question
#     Wikipedia Leader Lens evaluation set.
#     """

#     dataset = Dataset(
#         name="wikipedia_leader_lens_eval",
#         backend="local/csv",
#         root_dir="ragas_experiments",
#     )

#     samples = my_eval_set()

#     for sample in samples:
#         dataset.append(
#             {
#                 "question": sample["question"],
#                 "grading_notes": sample["grading_notes"],
#             }
#         )

#     dataset.save()
#     return dataset

# %%
# run_evaluation()
# validate_framework()


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%