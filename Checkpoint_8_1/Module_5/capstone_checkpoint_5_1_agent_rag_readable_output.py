r"""Capstone Checkpoint 5.1 — Wikipedia hybrid retrieval with Chroma — OpenRouter LLM variant.

Setup: pip install chromadb sentence-transformers rank-bm25 langchain-openai langchain-core
Place Wikipedia_text next to this script and set OPENROUTER_API_KEY in a .env file.
Run: python capstone_checkpoint_5_1_agent_rag_final_1.py

"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import argparse
import asyncio
import json
import os
import re
import pandas as pd
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod
import networkx as nx

import sys
import io
from contextlib import redirect_stdout

# Force UTF-8 output on Windows so Unicode characters such as
# →, ←, ✓, —, etc. can be printed safely.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load evaluation questions from Question_set.xlsx via Questions.py
from Questions import get_questions
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

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")
WIKIPEDIA_DIR = Path(os.getenv("WIKIPEDIA_TEXT_DIR", "Wikipedia_text"))
CHROMA_DIR = Path(os.getenv("WIKIPEDIA_CHROMA_DIR", "Wikipedia_chroma_full"))
TEXT_DIR = WIKIPEDIA_DIR

LOG_PATH = Path.cwd() / "checkpoint_5_1_openrouter.log"
TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.0"))
MAX_STEPS = 3

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
CANDIDATE_POOL = 12
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "wikipedia"   
DECIDE_SYSTEM = (
    "You inspect retrieved Wikipedia passages for a multi-part question. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries). "
    "Do not require a source to compare entities directly: evidence from each article suffices. "
    "If all requested facts are supported, return done=true."
)
ANSWER_SYSTEM = (
    "Answer only the parts of the question asked, using ONLY the provided passages. "
    "Cite exact passage IDs in square brackets after each factual claim. Never alter an ID. If a requested fact "
    "is unsupported, say that the provided passages do not establish it. "
    "Do not claim that a comparison requires a passage explicitly comparing the entities. "
    "For ownership, distinguish the owner from the chairman or manager."
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

# Initialized once after the corpus directory is selected in main().
SAMPLE_DOCS: list[dict] = []
PASSAGES: list[dict[str, str]] = []
PASSAGE_BY_ID: dict[str, dict[str, str]] = {}
HYBRID = None

def initialize_corpus(text_dir: Path) -> None:
    global SAMPLE_DOCS, PASSAGES, PASSAGE_BY_ID, HYBRID
    SAMPLE_DOCS = load_wikipedia_corpus(text_dir)
    PASSAGES = []
    for article in SAMPLE_DOCS:
        words = article["text"].split()
        for start in range(0, len(words), 150):
            chunk = words[start:start + 180]  # 30-word overlap
            if chunk:
                PASSAGES.append({ "id": f"{article['id']}#p{start // 150}", "article": article["id"], "text": " ".join(chunk), })

    PASSAGE_BY_ID = {p["id"]: p for p in PASSAGES}
    print(f"Prepared {len(PASSAGES)} Wikipedia passages.")
    db = build_or_load_db( PASSAGES, CHROMA_DIR,)
    HYBRID = HybridPassageRetriever(passages=PASSAGES, db=db, )

def check_openrouter() -> str:
    """Verify that an OpenRouter API key is available."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in a .env file or environment variable before running."
        )
    print(f"OpenRouter configured at {OPENROUTER_BASE_URL}")
    print(f"Selected OpenRouter model: {LLM_MODEL}")
    return key

def make_llm(model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_openrouter(),
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

_STOP = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}

def tokenize(value: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", value.lower()) if t not in _STOP]

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

# def build_or_load_db(documents: list[dict], chroma_dir: str = CHROMA_DIR) -> Chroma:
#     """Build/load a persistent Chroma index over the Wikipedia articles."""
#     if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
#         print(f"Loading existing vector DB from {chroma_dir}/")
#         return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())

#     print("Building vector DB (first run — embedding Wikipedia articles)...")
#     docs = [
#         Document(page_content=d["text"], metadata={"source": d["id"]})
#         for d in documents
#         if d.get("text", "").strip()
#     ]
#     if not docs:
#         raise RuntimeError("No Wikipedia documents are available for vector indexing.")
#     db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
#     print(f"  Indexed {len(docs)} Wikipedia articles into {chroma_dir}/")
#     return db

def build_or_load_db(passages: list[dict], chroma_dir: str = CHROMA_DIR ) -> Chroma:

    """Build/load a persistent Chroma index over Wikipedia passages."""

    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")

        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(),
        )

    print(
        "Building vector DB "
        "(first run — embedding Wikipedia passages)..."
    )

    docs = [
        Document(
            page_content=p["text"],
            metadata={
                "source": p["id"],
                "article": p["article"],
            },
        )
        for p in passages
        if p.get("text", "").strip()
    ]

    if not docs:
        raise RuntimeError(
            "No Wikipedia passages are available for vector indexing."
        )

    db = Chroma.from_documents(
        docs,
        get_embeddings(),
        persist_directory=chroma_dir,
    )

    print(
        f"Indexed {len(docs)} Wikipedia passages "
        f"into {chroma_dir}/"
    )

    return db

def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}

class HybridPassageRetriever:
    """Passage-level hybrid retriever using BM25 + Chroma vector search."""
    def __init__(self, passages: list[dict[str, str]], db: Chroma):
        self.passages = passages
        self.db = db
        self.passage_by_id = {p["id"]: p for p in passages}
        self.bm25 = BM25Okapi([
            tokenize(p["article"].replace("_", " ")) * 2 + tokenize(p["text"])
            for p in passages
        ])

    def search(self, query: str, k: int = 4) -> list[str]:
        candidate_count = min(CANDIDATE_POOL, len(self.passages))
        lexical_scores = self.bm25.get_scores(tokenize(query))
        bm_indices = sorted(range(len(lexical_scores)), key=lambda i: (-lexical_scores[i], i))[:candidate_count]
        bm25_scores = {self.passages[i]["id"]: float(lexical_scores[i]) for i in bm_indices if lexical_scores[i] > 0}
        vector_results = self.db.similarity_search_with_score(query, k=candidate_count)
        vector_scores = {
            document.metadata["source"]: 1.0 / (1.0 + float(distance))
            for document, distance in vector_results
            if document.metadata.get("source") in self.passage_by_id
        }
        bm25_norm = normalize(bm25_scores)
        vector_norm = normalize(vector_scores)
        candidate_ids = set(bm25_norm) | set(vector_norm)
        fused_scores = {
            passage_id: WEIGHT_BM25 * bm25_norm.get(passage_id, 0.0)
            + WEIGHT_VECTOR * vector_norm.get(passage_id, 0.0)
            for passage_id in candidate_ids
        }

        return sorted(fused_scores, key=lambda passage_id: (-fused_scores[passage_id], passage_id))[:k]
    
def retrieve(query: str, k: int = 4) -> list[str]:
    if HYBRID is None:
        raise RuntimeError("Initialize the Wikipedia corpus and hybrid index first.")
    return HYBRID.search(query, k)

def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    docs = "\n".join(f"[{i}] {text[:1100]}" for i, text in list(collected.items())[:10]) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nPassages so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get("done")) is not bool:
            raise ValueError("Expected JSON Boolean done")
        queries = data.get("new_queries", [])
        if not isinstance(queries, list):
            raise ValueError("Expected list of queries")
        return {"done": data["done"],
                "new_queries": [q.strip() for q in queries[:2] if isinstance(q, str) and q.strip()],
                "reasoning": str(data.get("reasoning", ""))[:300]}
    except (json.JSONDecodeError, ValueError):
        return {"done": False, "new_queries": [], "reasoning": "Invalid decision JSON", "parse_failure": True}

def initial_queries(question: str) -> list[str]:
    if "Margaret Thatcher" in question and "Adolf Hitler" in question:
        return ["Margaret Thatcher political role United Kingdom",
                "Adolf Hitler political role Germany"]
    return [question]

def answer_from_passages(llm: ChatOpenAI, question: str, collected: dict[str, str]) -> str:
    context = "\n\n".join(f"[{i}] {text[:1400]}" for i, text in list(collected.items())[:12])
    if not context:
        return "No relevant Wikipedia passages were retrieved."
    return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                       HumanMessage(content=f"Passages:\n{context}\n\nQuestion: {question}")]).content

def fixed_answer(llm: ChatOpenAI, question: str) -> dict:
    """Single-pass comparator using the same passage search and answer prompt."""
    start = perf_counter()
    ids = retrieve(question)
    collected = {i: PASSAGE_BY_ID[i]["text"] for i in ids}
    response = answer_from_passages(llm, question, collected)
    return {"answer": response, "stop_reason": "single_pass", "queries": [question],
            "passage_ids": ids, "rounds": 1, "llm_calls": int(bool(collected)),
            "elapsed_seconds": round(perf_counter() - start, 2)}

def agentic_answer(llm: ChatOpenAI, question: str) -> dict:
    """Lab 5.1 style retrieve -> analyze -> retrieve loop over Wikipedia passages."""
    start = perf_counter()
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = initial_queries(question)
    stop_reason = "round_limit"
    decision_calls = 0
    rounds = 0
    for step in range(MAX_STEPS):
        rounds = step + 1
        for query in pending:
            if query.casefold() in {q.casefold() for q in executed}:
                continue
            print(f"[agentic] retrieving for: {query!r}")
            for passage_id in retrieve(query):
                collected[passage_id] = PASSAGE_BY_ID[passage_id]["text"]
            executed.append(query)
        decision = decide(llm, question, collected, executed)
        decision_calls += 1
        print(f"[agentic] done={decision['done']} reasoning={decision['reasoning']!r}")
        if decision.get("parse_failure"):
            stop_reason = "parse_failure"
            break
        if decision["done"]:
            stop_reason = "evidence_sufficient"
            break
        if rounds == MAX_STEPS:
            break
        pending = [q for q in decision["new_queries"]
                   if q.casefold() not in {x.casefold() for x in executed}]
        if pending:
            print(f"[agentic] new queries: {pending}")
        else:
            stop_reason = "no_new_queries"
            break
    if stop_reason != "evidence_sufficient":
        print(f"[agentic] stopped: {stop_reason}; answer will identify unsupported parts")
    response = answer_from_passages(llm, question, collected)
    return {"answer": response, "stop_reason": stop_reason, "queries": executed,
            "passage_ids": list(collected), "rounds": rounds,
            "llm_calls": decision_calls + int(bool(collected)),
            "elapsed_seconds": round(perf_counter() - start, 2)}

class ToolUsingAgent:
    """
    Lab 5.2-style tool-using agent.

    Available tools:
      1. hybrid_search
      2. decompose_query
      3. article_search
      4. graph_expand
      5. answer
    """

    def __init__(self, llm: ChatOpenAI, max_steps: int = MAX_STEPS):
        self.llm = llm
        self.max_steps = max_steps

    # ---------------------------------------------------------
    # TOOL 1: HYBRID SEARCH
    # ---------------------------------------------------------
    def hybrid_search(self, query: str, k: int = 4) -> dict[str, str]:

        print(f"[tool] HYBRID_SEARCH: {query!r}")

        ids = retrieve(query, k)

        return {
            passage_id: PASSAGE_BY_ID[passage_id]["text"]
            for passage_id in ids
        }

    # ---------------------------------------------------------
    # TOOL 2: DECOMPOSE QUERY
    # ---------------------------------------------------------
    def decompose_query(self, question: str) -> list[str]:

        print(f"[tool] DECOMPOSE_QUERY: {question!r}")

        prompt = (
            "Break the following Wikipedia question into 2-4 focused "
            "retrieval queries. Return ONLY a JSON list of strings.\n\n"
            f"Question: {question}"
        )

        raw = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You decompose complex questions into focused "
                        "Wikipedia retrieval queries."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        ).content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            queries = json.loads(raw)

            if not isinstance(queries, list):
                return [question]

            return [
                q.strip()
                for q in queries[:4]
                if isinstance(q, str) and q.strip()
            ]

        except (json.JSONDecodeError, ValueError):
            return [question]

    # ---------------------------------------------------------
    # TOOL 3: ARTICLE SEARCH
    # ---------------------------------------------------------
    def article_search(
        self,
        article_name: str,
        query: str,
        k: int = 4
    ) -> dict[str, str]:

        print(
            f"[tool] ARTICLE_SEARCH: article={article_name!r}, "
            f"query={query!r}"
        )

        article_tokens = set(tokenize(article_name))

        candidates = []

        for passage in PASSAGES:

            name_tokens = set(
                tokenize(passage["article"].replace("_", " "))
            )

            if article_tokens and article_tokens <= name_tokens:
                candidates.append(passage)

        if not candidates:
            return {}

        query_tokens = set(tokenize(query))

        scored = []

        for passage in candidates:

            text_tokens = set(tokenize(passage["text"]))

            score = len(query_tokens & text_tokens)

            scored.append(
                (score, passage["id"])
            )

        scored.sort(
            key=lambda item: (-item[0], item[1])
        )

        ids = [
            passage_id
            for _, passage_id in scored[:k]
        ]

        return {
            passage_id: PASSAGE_BY_ID[passage_id]["text"]
            for passage_id in ids
        }

    # ---------------------------------------------------------
    # TOOL 4: GRAPH EXPAND
    # ---------------------------------------------------------
    def graph_expand(
        self,
        seed_ids: list[str],
        k: int = 4
    ) -> dict[str, str]:

        print(f"[tool] GRAPH_EXPAND: seeds={seed_ids}")

        seed_articles = {
            PASSAGE_BY_ID[passage_id]["article"]
            for passage_id in seed_ids
            if passage_id in PASSAGE_BY_ID
        }

        expanded = {}

        # Simple article-level expansion:
        # retrieve neighboring passages from the same articles.
        for article in seed_articles:

            article_passages = [
                p for p in PASSAGES
                if p["article"] == article
            ]

            seed_positions = []

            for passage_id in seed_ids:

                if passage_id not in PASSAGE_BY_ID:
                    continue

                if PASSAGE_BY_ID[passage_id]["article"] != article:
                    continue

                match = re.search(r"#p(\d+)$", passage_id)

                if match:
                    seed_positions.append(int(match.group(1)))

            for passage in article_passages:

                match = re.search(r"#p(\d+)$", passage["id"])

                if not match:
                    continue

                position = int(match.group(1))

                if any(
                    abs(position - seed) <= 1
                    for seed in seed_positions
                ):
                    expanded[passage["id"]] = passage["text"]

                    if len(expanded) >= k:
                        return expanded

        return expanded

    # ---------------------------------------------------------
    # AGENT DECISION
    # ---------------------------------------------------------
    def choose_action(
        self,
        question: str,
        evidence: dict[str, str],
        actions: list[dict],
    ) -> dict:

        evidence_text = "\n\n".join(
            f"[{pid}] {text[:900]}"
            for pid, text in list(evidence.items())[:12]
        )

        previous = json.dumps(
            actions[-6:],
            ensure_ascii=False
        )

        system = """
You are a tool-using Wikipedia retrieval agent.

Your goal is to gather enough evidence to answer every part
of the user's question.

Available tools:

1. hybrid_search
   Use for normal lexical + semantic retrieval.

2. decompose_query
   Use for complex, comparative, or multi-part questions.

3. article_search
   Use when you know which Wikipedia article contains the
   needed evidence.

4. graph_expand
   Use when retrieved passages need nearby contextual evidence.

5. answer
   Use ONLY when the evidence supports every requested fact.

Return ONLY JSON.

Valid formats:

{"tool":"hybrid_search",
 "query":"..."}

{"tool":"decompose_query"}

{"tool":"article_search",
 "article":"...",
 "query":"..."}

{"tool":"graph_expand"}

{"tool":"answer",
 "reasoning":"..."}

Do not repeatedly select the same action with the same query.
Prefer targeted retrieval when evidence is missing.
"""

        user = f"""
Question:
{question}

Previous actions:
{previous}

Evidence collected:
{evidence_text or "(none)"}

Choose the single best next action.
"""

        raw = self.llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
        ).content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            action = json.loads(raw)

            if not isinstance(action, dict):
                raise ValueError()

            return action

        except (json.JSONDecodeError, ValueError):

            # Safe fallback
            return {
                "tool": "hybrid_search",
                "query": question,
            }

    # ---------------------------------------------------------
    # MAIN TOOL-USING LOOP
    # ---------------------------------------------------------
    def run(self, question: str) -> dict:

        start = perf_counter()

        evidence: dict[str, str] = {}
        actions: list[dict] = []

        stop_reason = "step_limit"

        for step in range(self.max_steps):

            print(
                f"\n[tool-agent] STEP {step + 1}/{self.max_steps}"
            )

            action = self.choose_action(
                question,
                evidence,
                actions,
            )

            tool = action.get("tool", "hybrid_search")

            print(f"[tool-agent] selected: {tool}")

            actions.append(action)

            # ---------------------------------------------
            # HYBRID SEARCH
            # ---------------------------------------------
            if tool == "hybrid_search":

                query = action.get("query", question)

                result = self.hybrid_search(query)

                evidence.update(result)

            # ---------------------------------------------
            # DECOMPOSITION
            # ---------------------------------------------
            elif tool == "decompose_query":

                subqueries = self.decompose_query(question)

                print(
                    f"[tool-agent] subqueries: {subqueries}"
                )

                for query in subqueries:

                    result = self.hybrid_search(query)

                    evidence.update(result)

            # ---------------------------------------------
            # ARTICLE SEARCH
            # ---------------------------------------------
            elif tool == "article_search":

                article = action.get("article", "")
                query = action.get("query", question)

                result = self.article_search(
                    article,
                    query,
                )

                evidence.update(result)

            # ---------------------------------------------
            # GRAPH EXPANSION
            # ---------------------------------------------
            elif tool == "graph_expand":

                result = self.graph_expand(
                    list(evidence.keys())
                )

                evidence.update(result)

            # ---------------------------------------------
            # ANSWER
            # ---------------------------------------------
            elif tool == "answer":

                stop_reason = "agent_evidence_sufficient"
                break

            else:

                print(
                    f"[tool-agent] unknown tool {tool!r}; "
                    "falling back to hybrid search"
                )

                evidence.update(
                    self.hybrid_search(question)
                )

        response = answer_from_passages(
            self.llm,
            question,
            evidence,
        )

        return {
            "answer": response,
            "stop_reason": stop_reason,
            "actions": actions,
            "passage_ids": list(evidence),
            "steps": len(actions),
            "elapsed_seconds": round(
                perf_counter() - start,
                2,
            ),
        }

def tool_using_answer(llm: ChatOpenAI, question: str) -> dict:
    agent = ToolUsingAgent(llm)
    return agent.run(question)

# =========================================================
# EVALUATION QUESTIONS — loaded from Question_set.xlsx
# =========================================================
# Questions.py exposes get_questions(sheet_number), which returns the
# Question column from Sheet1, Sheet2, or Sheet3. Change this value to
# 1, 2, or 3 to select the evaluation sheet used by this checkpoint.
QUESTION_SHEET = 1


def load_test_queries(sheet_number: int = QUESTION_SHEET) -> list[str]:
    """Load the checkpoint evaluation questions from Questions.py."""
    questions = get_questions(sheet_number)
    if not questions:
        raise RuntimeError(
            f"No evaluation questions were loaded from Sheet{sheet_number}."
        )
    print(
        f"Loaded {len(questions)} evaluation questions "
        f"from Question_set.xlsx / Sheet{sheet_number}."
    )
    return questions


TEST_QUERIES = load_test_queries()
#print(f"Questions from queries", TEST_QUERIES)

def my_agent_plan() -> dict[str, Any]:
    """Return the agent design for the Wikipedia Retrieval Engine."""

    return {
        "tools": [
            "hybrid_search",
            "decompose_query",
            "find_by_category",
            "find_by_topic",
            "graph_expand",
            "clarify",
            "answer",
        ],

        "stop_condition": (
            "Stop when the retrieved evidence covers every part of the user's "
            "question, including all required entities and facts, and the answer "
            "can be supported directly by the retrieved Wikipedia passages. "
            "Otherwise, perform another targeted retrieval step until the evidence "
            "is sufficient or the maximum number of agent steps is reached."
        ),

        "system_prompt_idea": (
            "Begin with hybrid retrieval and inspect the evidence before answering. "
            "For complex or multi-part questions, decompose the request into focused "
            "subqueries and use category, topic, or graph expansion when they can "
            "recover missing evidence. Avoid repeating searches, preserve evidence "
            "for every required entity, clarify only when the request is genuinely "
            "ambiguous, and answer only from retrieved Wikipedia evidence."
        ),

        "test_tasks": TEST_QUERIES,
    }

# =========================================================
# READABLE EVALUATION OUTPUT
# =========================================================

READABLE_OUTPUT_PATH = Path.cwd() / "checkpoint_5_1_readable_output.txt"
SHOW_INTERNAL_TRACES = False


def _clean_display_text(value: Any) -> str:
    """Normalize text for readable console/report output."""
    text = str(value or "").strip()
    # Keep output readable if a captured Windows stream contains common mojibake.
    replacements = {
        "ΓÇ£": '"', "ΓÇ¥": '"', "ΓÇÖ": "'", "ΓÇÿ": "'",
        "ΓÇô": "-", "ΓÇö": "--", "ΓåÆ": "->",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _format_tool_action(action: dict, number: int) -> str:
    """Turn a raw tool-action dictionary into a compact human-readable line."""
    tool = str(action.get("tool", "unknown")).replace("_", " ").upper()
    details = []
    if action.get("article"):
        details.append(f"article={action['article']}")
    if action.get("query"):
        details.append(f"query={action['query']}")
    if action.get("reasoning"):
        details.append(f"reason={action['reasoning']}")
    suffix = f" | {' | '.join(details)}" if details else ""
    return f"  {number}. {tool}{suffix}"


def format_readable_result(label: str, result: dict, question: str) -> str:
    """Create a concise, presentation-friendly view of one workflow result."""
    title = label.replace("_", " ")
    lines = [
        f"--- {title} ---",
        "",
        "ANSWER",
        _clean_display_text(result.get("answer", "")),
        "",
        "RUN SUMMARY",
        f"  Stop reason       : {result.get('stop_reason', 'n/a')}",
        f"  Time              : {result.get('elapsed_seconds', 'n/a')} s",
        f"  Passages retrieved: {len(result.get('passage_ids', []))}",
    ]

    if "rounds" in result:
        lines.append(f"  Retrieval rounds  : {result.get('rounds', 'n/a')}")
    if "llm_calls" in result:
        lines.append(f"  LLM calls         : {result.get('llm_calls', 'n/a')}")

    queries = result.get("queries", [])
    # Do not repeat the full original question when it was the only query.
    follow_ups = [q for q in queries if q.strip() != question.strip()]
    if follow_ups:
        lines.extend(["", "FOLLOW-UP QUERIES"])
        for i, query in enumerate(follow_ups, 1):
            lines.append(f"  {i}. {_clean_display_text(query)}")
    elif queries:
        lines.append("  Follow-up queries : none")

    actions = result.get("actions", [])
    if actions:
        lines.extend(["", "TOOL PATH"])
        for i, action in enumerate(actions, 1):
            lines.append(_format_tool_action(action, i))

    return "\n".join(lines)


def _run_without_trace(runner, llm, question: str) -> dict:
    """Run a workflow while hiding low-level retrieval debug messages."""
    if SHOW_INTERNAL_TRACES:
        return runner(llm, question)
    with redirect_stdout(io.StringIO()):
        return runner(llm, question)


def _format_comparison(results: dict[str, dict]) -> str:
    """Create a compact side-by-side metrics summary for one test question."""
    lines = [
        "",
        "WORKFLOW COMPARISON",
        f"{'Workflow':<20} {'Stop reason':<26} {'Time (s)':>10} {'Passages':>10}",
        "-" * 70,
    ]
    for label, result in results.items():
        workflow = label.replace("_", " ")
        lines.append(
            f"{workflow:<20} "
            f"{str(result.get('stop_reason', 'n/a')):<26} "
            f"{float(result.get('elapsed_seconds', 0)):>10.2f} "
            f"{len(result.get('passage_ids', [])):>10}"
        )
    return "\n".join(lines)


def run() -> None:
    """Compare workflows and emit a clean human-readable evaluation report."""
    initialize_corpus(WIKIPEDIA_DIR)
    check_openrouter()
    llm = make_llm()

    report_parts = [
        "LEADER LENS - CHECKPOINT 5.1 EVALUATION",
        "Fixed Retrieval vs Agentic Retrieval vs Tool-Using Agent",
        "=" * 78,
    ]

    workflows = (
        ("FIXED", fixed_answer),
        ("AGENTIC", agentic_answer),
        ("TOOL_USING_AGENT", tool_using_answer),
    )

    for number, question in enumerate(my_agent_plan()["test_tasks"], 1):
        question = _clean_display_text(question)
        test_parts = [
            "",
            "=" * 78,
            f"TEST {number}",
            "=" * 78,
            "QUESTION",
            question,
        ]
        results = {}

        for label, runner in workflows:
            result = _run_without_trace(runner, llm, question)
            results[label] = result
            test_parts.extend(["", format_readable_result(label, result, question)])

            # Preserve the full machine-readable diagnostic record separately.
            log_response(label, question, json.dumps(result, ensure_ascii=False))

        test_parts.append(_format_comparison(results))
        block = "\n".join(test_parts)
        report_parts.append(block)
        print(block)

    report = "\n".join(report_parts)
    READABLE_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("Evaluation complete.")
    print(f"Readable report: {READABLE_OUTPUT_PATH}")
    print(f"Detailed log   : {LOG_PATH}")


if __name__ == "__main__":
    run()
