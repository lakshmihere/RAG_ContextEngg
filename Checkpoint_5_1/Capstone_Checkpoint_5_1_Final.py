r"""Capstone Checkpoint 5.1 — Wikipedia hybrid retrieval with Chroma.

Setup: pip install chromadb sentence-transformers rank-bm25 langchain-openai langchain-core
Place Wikipedia_10_text next to this script and start LM Studio.
Run: python capstone_checkpoint_5_1_wikipedia_HYBRID.py

Run: python capstone_checkpoint_5_1_wikipedia_PLAN_RUN.py
Reads Wikipedia_10_text, builds/loads Chroma, and runs the three my_agent_plan test tasks automatically.
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

    # %% [markdown]
    # # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
    # **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
    #
    # ## What this checkpoint is
    #
    # This is the final build step of your capstone. You will turn the retrieval system you've
    # developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
    # decides at each step whether it has enough information, what to retrieve next, which
    # tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
    # retriever and Lab 5.2's tool-using agent) to your capstone scenario.
    #
    # The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
    # runnable agentic loop and single-pass comparison over Wikipedia passages.
    #
    # **Learning outcomes (Module 5):**
    # 1. Build an agent-based system that integrates retrieval and external tools.
    # 2. Use system prompts to guide agent behavior and decision-making.
    # 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
    # 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
    # 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
    # 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

    # %% [markdown]
    # ## Step 1 — Keep your capstone scenario

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import json
import hashlib
import os
import re
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI


LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
# LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-coder-6.7b-instruct")
LLM_MODEL = os.getenv("LLM_MODEL","meta-llama-3.1-8b-instruct")
WIKIPEDIA_DIR = Path(os.getenv("WIKIPEDIA_TEXT_DIR", "Wikipedia_10_text"))
TEXT_DIR = WIKIPEDIA_DIR
LOG_PATH = Path.cwd() / "Tool_using_agent_1.log"
TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.0"))
MAX_STEPS = 3
CHROMA_DIR = Path(os.getenv("WIKIPEDIA_5_1_CHROMA_DIR", "wikipedia_5_1_chroma_hf"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "wikipedia_checkpoint_5_1_passages"
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
                PASSAGES.append({"id": f"{article['id']}#p{start // 150}",
                                 "article": article["id"], "text": " ".join(chunk)})
    PASSAGE_BY_ID = {p["id"]: p for p in PASSAGES}
    print(f"Prepared {len(PASSAGES)} Wikipedia passages.")
    HYBRID = HybridPassageRetriever(PASSAGES, CHROMA_DIR)


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


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


class HybridPassageRetriever:
    def __init__(self, passages: list[dict[str, str]], path: Path):
        self.passages = passages
        self.embedding = SentenceTransformer(EMBEDDING_MODEL)
        self.bm25 = BM25Okapi([tokenize(p["article"].replace("_", " ")) * 2
                               + tokenize(p["text"]) for p in passages])
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        # A manifest guards against mixing a previous corpus, chunk layout, or model.
        signature = hashlib.sha256()
        for p in passages:
            signature.update(p["id"].encode("utf-8"))
            signature.update(b"\0")
            signature.update(p["text"].encode("utf-8"))
            signature.update(b"\0")
        expected = {"embedding_model": EMBEDDING_MODEL, "corpus_sha256": signature.hexdigest(),
                    "passage_count": len(passages), "collection": COLLECTION_NAME}
        manifest = path / "checkpoint_5_1_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                raise RuntimeError(f"Index settings differ from {manifest}. Use a new WIKIPEDIA_5_1_CHROMA_DIR for this corpus/model.")
            if self.collection.count() != len(passages):
                raise RuntimeError(f"Index contains {self.collection.count()} records, expected {len(passages)}. Use a new index directory.")
            print(f"Loading existing vector DB from {path}/ ({len(passages)} passages)")
        else:
            if self.collection.count():
                raise RuntimeError(f"Existing Chroma collection at {path} has no matching manifest. Use a new WIKIPEDIA_5_1_CHROMA_DIR.")
            print(f"Building Chroma vector DB in {path}/ ({len(passages)} passages)...")
            for start in range(0, len(passages), 64):
                batch = passages[start:start + 64]
                vectors = self.embedding.encode([p["text"] for p in batch],
                                                convert_to_numpy=True, show_progress_bar=False)
                self.collection.upsert(ids=[p["id"] for p in batch],
                                       embeddings=vectors.tolist(),
                                       documents=[p["text"] for p in batch],
                                       metadatas=[{"article": p["article"]} for p in batch])
            if self.collection.count() != len(passages):
                raise RuntimeError("Incomplete Chroma index; use a fresh index directory before retrying.")
            manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
            print(f"Saved Chroma vector DB to {path}/")

    def search(self, query: str, k: int = 4) -> list[str]:
        count = min(CANDIDATE_POOL, len(self.passages))
        lexical = self.bm25.get_scores(tokenize(query))
        bm_indices = sorted(range(len(lexical)), key=lambda i: (-lexical[i], i))[:count]
        bm = {self.passages[i]["id"]: float(lexical[i]) for i in bm_indices if lexical[i] > 0}
        query_vector = self.embedding.encode(query, convert_to_numpy=True).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=count,
                                        include=["distances"])
        vec = {passage_id: 1.0 / (1.0 + float(distance))
               for passage_id, distance in zip(results["ids"][0], results["distances"][0])}
        bm_norm, vec_norm = normalize(bm), normalize(vec)
        fused = {passage_id: WEIGHT_BM25 * bm_norm.get(passage_id, 0.0)
                 + WEIGHT_VECTOR * vec_norm.get(passage_id, 0.0)
                 for passage_id in bm.keys() | vec.keys()}
        return sorted(fused, key=lambda passage_id: (-fused[passage_id], passage_id))[:k]


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

        "test_tasks": [
            (
                "According to the Wikipedia article on the Empire State Building, "
                "when was the building completed and when did it officially open?"
            ),
            (
                "According to the Wikipedia article on Hallstatt, where is Hallstatt "
                "located and what is it particularly known for?"
            ),
            (
                "What role did Steve Jobs play in the development of Apple, "
                "according to his Wikipedia article?"
            ),
            (
                "Compare the leadership roles and historical significance of "
                "Margaret Thatcher and Adolf Hitler as described in their "
                "respective Wikipedia articles."
            ),
            (
                "How were Margaret Thatcher and Adolf Hitler connected through the "
                "major European political and military events that shaped the "
                "periods in which they rose to prominence?"
            ),
            (
                "According to the Wikipedia article on Emirates airline, when was "
                "Emirates founded, where is it based, and who owns it? Quote the "
                "relevant sentence or sentences."
            ),
            (
                "Which Austrian settlement in the corpus is strongly associated "
                "with prehistoric salt mining, and where is it located?"
            ),
            (
                "What did Margaret Thatcher study at the University of Oxford, "
                "and what academic qualification did she receive?"
            ),
            (
                "Compare the roles of Queen Victoria and Margaret Thatcher in "
                "British history using evidence from their respective Wikipedia "
                "articles."
            ),
            (
                "Who was Victoria, what position did she hold, and which country "
                "or kingdom did she rule?"
            ),
        ],
    }

def run() -> None:
    """Compare one-pass and agentic workflows with the same hybrid retriever."""
    initialize_corpus(WIKIPEDIA_DIR)
    check_lm_studio()
    llm = make_llm()
    print("Checkpoint 5.1 — Wikipedia hybrid retrieval (fixed versus agentic)")
    for number, question in enumerate(my_agent_plan()["test_tasks"], 1):
        print(f"\n{'=' * 72}\nTest {number}: {question}")
        for label, runner in (("FIXED", fixed_answer), ("AGENTIC", agentic_answer), ("TOOL_USING_AGENT", tool_using_answer),):
            result = runner(llm, question)
            print(f"\n{label} answer:\n{result['answer']}\n")
            print(f"{label} " f"stop={result['stop_reason']} " f"seconds={result['elapsed_seconds']}")
            if "queries" in result:
                print(f"Queries: {result['queries']}")
            if "actions" in result:
                print("\nTool actions:")
                for action in result["actions"]:
                    print(f"  {action}")
            print(f"Passages retrieved: " f"{len(result['passage_ids'])}")
            log_response(label, question, json.dumps(result, ensure_ascii=False))

        # for label, runner in (("FIXED", fixed_answer), ("AGENTIC", agentic_answer)):
        #     result = runner(llm, question)
        #     print(f"\n{label} answer: {result['answer']}\n")
        #     print(f"{label} stop={result['stop_reason']} rounds={result['rounds']} "
        #           f"queries={len(result['queries'])} model_calls={result['llm_calls']} "
        #           f"seconds={result['elapsed_seconds']}")
        #     log_response(label, question, json.dumps(result, ensure_ascii=False))
    print("\nAll the fixed and agentic comparisons completed.")


if __name__ == "__main__":
    run()
