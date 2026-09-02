r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
#LLM_MODEL = "openai/gpt-oss-120b"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 4          # Emails sent to the LLM as context.

CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion.
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval_1.log"

# SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
# You answer questions by drawing information exclusively from the company e-mails \
# provided to you as context in each message.

# Rules:
# - If the answer can be found in the provided e-mails, answer clearly and concisely.
# - If the provided e-mails do not contain enough information to answer the question, \
# say so explicitly and do not speculate or use outside knowledge.
# - Do not answer questions that are unrelated to the content of the provided e-mails."""


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


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
# There are six short "documents" on distinct topics so a baseline retriever has something to
# discriminate between. Your real corpus is the PDFs/articles in `Labs/CapstoneDatasets/`,
# which came with the course in Module 1. Point your code at your local copy of that
# folder, and update the path if your checkout puts it elsewhere.

# %%

HTML_DIR = Path("Wikipedia")
HTML_FILES = [
    "Adolf_Hitler.html",
    "Emirates_(airline).html",
    "Empire_State_Building.html",
    "Margaret_Thatcher.html",
    "Steve_Jobs.html",
    "Hallstatt.html",
]

def extract_text_from_html(file_path: Path) -> str:
    """Load one HTML file and convert it to clean readable text."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Extract visible text
    text = soup.get_text(separator=" ", strip=True)
    return text

SAMPLE_DOCS = []
for filename in HTML_FILES:
    file_path = HTML_DIR / filename
    if not file_path.exists():
        print(f"WARNING: File not found: {file_path}")
        continue
    SAMPLE_DOCS.append(
        {
            "id": filename,
            "text": extract_text_from_html(file_path),
        }
    )
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


# Check that the files loaded correctly
print(f"Loaded {len(SAMPLE_DOCS)} documents")
for doc in SAMPLE_DOCS:
    print(doc["id"], len(doc["text"]))


# %% [markdown]
# ## Step 2 — Hybrid Retriever (BM25 + Semantic Vector Search)
# The hybrid retriever combines BM25 keyword retrieval with Chroma semantic vector
# retrieval. BM25 favors exact terms such as Wikipedia article/entity names, while
# vector retrieval handles paraphrases and conceptually related wording. Scores from
# both retrievers are normalized and combined using weighted fusion before the top-k
# documents are supplied to the grounded-answer LLM.

# %%
# def _tokens(text: str) -> set[str]:
#     return set(re.findall(r"[a-z0-9]+", text.lower()))


# def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
#     """Baseline keyword retrieval: Score each doc by shared-word count, return top-k."""
#     q = _tokens(query)
#     scored = [(d["id"], float(len(q & _tokens(d["text"])))) for d in SAMPLE_DOCS]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]


# def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
#     context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
#     messages = [
#         SystemMessage(content=ANSWER_SYSTEM),
#         HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
#     ]
#     return llm.invoke(messages).content

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
    """Build a persistent Chroma DB from the loaded Wikipedia HTML articles,
    or load it on later runs.
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
            metadata={"source": d["id"]},
        )
        for d in SAMPLE_DOCS
    ]

    if not docs:
        raise RuntimeError(
            f"No Wikipedia documents were loaded. Check HTML_DIR={HTML_DIR!s} "
            "and make sure the HTML files are present."
        )

    db = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=chroma_dir,
    )
    print(f"  Indexed {len(docs)} Wikipedia articles into {chroma_dir}/")
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


class HybridRetriever(BaseRetriever):
    """Fuse BM25 keyword retrieval with Chroma semantic vector retrieval."""

    def __init__(
        self,
        db: Chroma,
        documents: list[dict] = SAMPLE_DOCS,
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
            f"Hybrid retriever ready over {len(self._names)} Wikipedia articles "
            f"(BM25 weight={WEIGHT_BM25}, vector weight={WEIGHT_VECTOR})."
        )

    def _bm25_topk(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Return the top-k Wikipedia articles ranked by BM25."""
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
        """Return the top-k Wikipedia articles ranked by vector distance."""
        k = min(k, len(self._names))
        results = self._db.similarity_search_with_score(query, k=k)

        return [
            (
                doc.metadata.get("source", "unknown"),
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


# %%
def my_representative_queries() -> list[str]:
    """Return representative queries for the Wikipedia Retrieval Engine scenario.

    These queries test:
    - single-document factual retrieval,
    - exact/quoted information retrieval,
    - multi-document comparison,
    - and retrieval from different Wikipedia article topics.
    """

    return [
        "According to the Wikipedia article on the Empire State Building, "
        "when was the building completed and when did it officially open? "
        "Quote the relevant sentence from the article.",

        "Quote, word for word, the opening sentence of the Wikipedia article "
        "on Steve Jobs.",

        "According to the Wikipedia article on Margaret Thatcher, what was "
        "her educational background before entering politics? Quote the "
        "relevant passage.",

        "Compare the leadership roles and historical significance of "
        "Margaret Thatcher and Adolf Hitler as described in their respective "
        "Wikipedia articles. Support the comparison with a direct quote from "
        "each article.",

        "According to the Wikipedia article on Emirates airline, when was "
        "Emirates founded, where is it based, and who owns it? Quote the "
        "relevant sentence or sentences.",

        "According to the Hallstatt article, how many levels does the "
        "Hallstatt salt mine comprise, and what is the elevation range of "
        "its shafts?"
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
def run() -> None:
    # Build or load the Chroma vector database used by the vector side.
    db = build_or_load_db(CHROMA_DIR)

    # Create the hybrid retriever: BM25 + semantic vector retrieval.
    retriever = HybridRetriever(
        db=db,
        documents=SAMPLE_DOCS,
        num_retrieved=NUM_RETRIEVED,
    )

    queries = my_representative_queries()

    print(
        f"Checkpoint 2.1 — hybrid retrieval (BM25 + vector)  |  "
        f"scenario: {SCENARIO}\n"
    )

    for i, query in enumerate(queries, 1):
        hits = retriever.getTopK(query, TOP_K)

        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print("  retrieved:")

        # Hybrid/fused scores are HIGHER = BETTER.
        for rank, (source, content, score) in enumerate(hits, start=1):
            print(
                f"    {rank}. {source} | "
                f"hybrid score = {score:.4f}"
            )

        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue

        ans = retriever.query(query)
        print(f"\n  answer: {ans}\n")

        hit_summary = [
            (source, score)
            for source, _, score in hits
        ]

        log(
            f"QUERY {i}: {query}",
            f"retrieved={hit_summary}\nanswer={ans}",
        )

    print("=" * 72)
    print(
        "Done. Use the hybrid retrieved document results above as evidence "
        "in your writeup."
    )


if __name__ == "__main__":
    run()


# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
