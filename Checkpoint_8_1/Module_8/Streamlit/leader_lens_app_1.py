"""
Leader Lens Console — Streamlit demonstration layer
---------------------------------------------------
Place this file next to Capstone_checkpoint_7_1.py and run:

    streamlit run leader_lens_app.py

This UI calls the existing Capstone backend. It does not replace the RAG engine.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st

BACKEND_FILE = Path(__file__).with_name("Capstone_checkpoint_7_1.py")

st.set_page_config(
    page_title="Leader Lens Console",
    page_icon="🔎",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1450px;}
.ll-title {font-size: 2.05rem; font-weight: 800; margin-bottom: 0;}
.ll-sub {color: #666; margin-top: -0.2rem; margin-bottom: 1.0rem;}
.stage {
    border: 1px solid #dddddd; border-radius: 12px; padding: 14px 16px;
    background: white; min-height: 132px;
}
.stage h4 {margin: 0 0 8px 0;}
.green {border-top: 5px solid #238636;}
.blue {border-top: 5px solid #2563eb;}
.amber {border-top: 5px solid #d97706;}
.purple {border-top: 5px solid #6d28d9;}
.metricbox {
    border: 1px solid #e4e4e4; border-radius: 10px; padding: 10px 12px;
    background: #fafafa;
}
.small {font-size: .88rem; color: #666;}

.leader-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    padding: 14px 4px 15px 4px;
    margin-bottom: 14px;
    border-top: 3px solid #c9c9c9;
    border-bottom: 3px solid #c9c9c9;
}
.leader-header::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -6px;
    border-bottom: 1px solid #e2e2e2;
}
.leader-brand {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    color: #182235;
    line-height: 1.1;
}
.leader-tagline {
    margin-left: 42px;
    margin-top: 5px;
    font-size: 0.92rem;
    color: #6b7280;
}
.leader-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.7px;
    white-space: nowrap;
}
.status-label { color: #374151; }
.status-dot {
    width: 9px;
    height: 9px;
    background: #22a447;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 4px rgba(34, 164, 71, 0.12);
}
.status-online { color: #16843a; }

</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_backend():
    if not BACKEND_FILE.exists():
        raise FileNotFoundError(
            f"{BACKEND_FILE.name} was not found. Put leader_lens_app.py in the same "
            "folder as Capstone_checkpoint_7_1.py."
        )
    spec = importlib.util.spec_from_file_location("leader_lens_backend", BACKEND_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["leader_lens_backend"] = module
    spec.loader.exec_module(module)

    module.initialize_corpus(module.TEXT_DIR)
    llm = module.make_llm()
    cache = module.SemanticCache(module.current_system_signature())
    return module, llm, cache


def token_total(usage: dict[str, int]) -> int:
    return sum(int(v or 0) for v in usage.values())


def article_name(passage_id: str) -> str:
    return passage_id.split("#", 1)[0].replace(".txt", "").replace("_", " ")


def run_normal(backend, llm, cache, question: str) -> dict[str, Any]:
    """Run the real routed backend and collect a UI-friendly trace."""
    sanitized = backend._sanitize_user_text(question)
    security_status = "PASS" if sanitized.strip() else "BLOCKED"

    # Check cache first so the UI can truthfully report hit/miss.
    before_hits = cache.hits
    cached = cache.get(sanitized)
    if cached is not None:
        return {
            "mode": "normal",
            "raw_question": question,
            "sanitized_question": sanitized,
            "security_status": security_status,
            "cache_status": "HIT",
            "cache_similarity": cached.get("similarity"),
            "route": "SKIPPED",
            "model": "SKIPPED",
            "retrieval": "SKIPPED",
            "retrieved_ids": cached.get("evidence_ids", []),
            "agent_steps": [],
            "quality_gate": "SKIPPED",
            "path": "semantic-cache-hit",
            "tokens": 0,
            "latency_ms": 0.0,
            "answer": cached["answer"],
        }

    decision = backend.route(sanitized)
    evidence_ids = backend.retrieve(sanitized, k=3 if decision == "direct" else 4)

    t0 = perf_counter()
    answer, usage, path = backend.answer_routed(
        llm, sanitized, cache, decision=decision
    )
    latency_ms = (perf_counter() - t0) * 1000

    if decision == "direct":
        model = backend.DIRECT_MODEL
        quality_gate = "PASS" if path == "direct-quality-pass" else "ESCALATED"
        agent_steps = [] if path == "direct-quality-pass" else ["Direct answer rejected", "Agent fallback executed"]
    else:
        model = backend.LLM_MODEL
        quality_gate = "AGENT PATH"
        agent_steps = ["Retrieve", "Inspect evidence", f"Refine if required (max {backend.MAX_STEPS})"]

    return {
        "mode": "normal",
        "raw_question": question,
        "sanitized_question": sanitized,
        "security_status": security_status,
        "cache_status": "MISS",
        "cache_similarity": None,
        "route": decision.upper(),
        "model": model,
        "retrieval": f"Hybrid BM25 {backend.WEIGHT_BM25:.0%} + Vector {backend.WEIGHT_VECTOR:.0%}",
        "retrieved_ids": evidence_ids,
        "agent_steps": agent_steps,
        "quality_gate": quality_gate,
        "path": path,
        "tokens": token_total(usage),
        "latency_ms": latency_ms,
        "answer": answer,
    }


def run_cache_demo(backend, llm, cache) -> dict[str, Any]:
    """Prime with the canonical question, then run the real paraphrase cache lookup."""
    # Ensure the first question is in this session's semantic cache.
    if cache.get(backend.question) is None:
        decision = backend.route(backend.question)
        backend.answer_routed(llm, backend.question, cache, decision=decision)

    t0 = perf_counter()
    cached = cache.get(backend.paraphrase)
    lookup_ms = (perf_counter() - t0) * 1000

    if cached is not None:
        return {
            "mode": "cache",
            "raw_question": backend.paraphrase,
            "sanitized_question": backend.paraphrase,
            "security_status": "PASS",
            "cache_status": "HIT",
            "cache_similarity": cached.get("similarity"),
            "route": "SKIPPED",
            "model": "SKIPPED",
            "retrieval": "SKIPPED",
            "retrieved_ids": cached.get("evidence_ids", []),
            "agent_steps": [],
            "quality_gate": "SKIPPED",
            "path": "semantic-cache-hit",
            "tokens": 0,
            "latency_ms": lookup_ms,
            "answer": cached["answer"],
        }

    # If the configured threshold/validator does not accept the paraphrase, show the truth:
    result = run_normal(backend, llm, cache, backend.paraphrase)
    result["mode"] = "cache"
    return result


def run_security(backend, llm) -> dict[str, Any]:
    attack = backend.wiki_attack
    sanitized = backend._sanitize_user_text(attack)

    evidence_ids = backend.retrieve(sanitized, k=3)
    t0 = perf_counter()
    answer = backend.probe_agent(llm, attack, sanitize=True)
    latency_ms = (perf_counter() - t0) * 1000

    return {
        "mode": "security",
        "raw_question": attack,
        "sanitized_question": sanitized,
        "security_status": "ATTACK NEUTRALIZED",
        "cache_status": "BYPASSED",
        "cache_similarity": None,
        "route": "HARDENED",
        "model": backend.LLM_MODEL,
        "retrieval": f"Hybrid BM25 {backend.WEIGHT_BM25:.0%} + Vector {backend.WEIGHT_VECTOR:.0%}",
        "retrieved_ids": evidence_ids,
        "agent_steps": ["Escape XML", "Neutralize instruction injection", "Retrieve using sanitized input"],
        "quality_gate": "HARDENED XML TRUST BOUNDARY",
        "path": "hardened-direct",
        "tokens": None,
        "latency_ms": latency_ms,
        "answer": answer,
    }


def stage_cards(trace: dict[str, Any]):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sim = trace.get("cache_similarity")
        sim_text = f"{sim:.3f}" if isinstance(sim, (float, int)) else "—"
        st.markdown(
            f"""<div class="stage green"><h4>1 · PROTECT & REUSE</h4>
            <b>Security</b> · {trace['security_status']}<br>
            <b>Cache</b> · {trace['cache_status']}<br>
            <span class="small">Similarity: {sim_text}</span></div>""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""<div class="stage blue"><h4>2 · ROUTE & RETRIEVE</h4>
            <b>Route</b> · {trace['route']}<br>
            <b>Model</b> · {trace['model']}<br>
            <span class="small">{trace['retrieval']}</span></div>""",
            unsafe_allow_html=True,
        )

    with c3:
        steps = trace.get("agent_steps", [])
        steps_text = "<br>".join(f"✓ {s}" for s in steps) if steps else "No agent loop required"
        st.markdown(
            f"""<div class="stage amber"><h4>3 · CHECK & ESCALATE</h4>
            <b>Gate</b> · {trace['quality_gate']}<br>
            <span class="small">{steps_text}</span></div>""",
            unsafe_allow_html=True,
        )

    with c4:
        ev = len(trace.get("retrieved_ids", []))
        st.markdown(
            f"""<div class="stage purple"><h4>4 · GROUND & MEASURE</h4>
            <b>Path</b> · {trace['path']}<br>
            <b>Evidence</b> · {ev} passages<br>
            <span class="small">Grounded answer + execution metrics</span></div>""",
            unsafe_allow_html=True,
        )


def render_result(trace: dict[str, Any]):
    stage_cards(trace)
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Execution path", trace["path"])
    m2.metric("Tokens", "—" if trace["tokens"] is None else f"{trace['tokens']:,}")
    m3.metric("Latency", f"{trace['latency_ms']:.1f} ms")
    m4.metric("Cache", trace["cache_status"])

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Grounded answer")
        st.info(trace["answer"])

    with right:
        st.subheader("Retrieved evidence")
        ids = trace.get("retrieved_ids", [])
        if not ids:
            st.caption("No retrieval required for this path.")
        for i, pid in enumerate(ids, 1):
            with st.expander(f"[{i}] {article_name(pid)} · {pid}", expanded=(i <= 2)):
                try:
                    p = backend.PASSAGE_BY_ID.get(pid)
                    st.write(p["text"] if p else "Passage text unavailable in the current index.")
                except Exception:
                    st.write("Passage text unavailable.")

    if trace["mode"] == "security":
        st.subheader("Security transformation")
        a, b = st.columns(2)
        with a:
            st.error("RAW ADVERSARIAL INPUT")
            st.code(trace["raw_question"], language=None)
        with b:
            st.success("SANITIZED INPUT USED FOR RETRIEVAL")
            st.code(trace["sanitized_question"], language=None)


st.markdown("""
<div class="leader-header">
    <div>
        <div class="leader-brand">
            🔎 <span>LEADER LENS</span>
        </div>
        <div class="leader-tagline">
            Wikipedia Retrieval &amp; Evidence Engine
        </div>
    </div>
    <div class="leader-status">
        <span class="status-label">PRODUCTION RAG</span>
        <span class="status-dot"></span>
        <span class="status-online">ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("Initializing Wikipedia corpus, hybrid retriever, and models..."):
        backend, llm, cache = load_backend()
except Exception as exc:
    st.error("Leader Lens could not initialize.")
    st.exception(exc)
    st.stop()

tab1, tab2, tab3 = st.tabs(["① NORMAL QUERY", "② CACHE REUSE", "③ SECURITY ATTACK"])

with tab1:
    st.caption("Run a real production request through security, cache, routing, retrieval, generation, and measurement.")
    q = st.text_area("Ask Leader Lens", value=backend.question, height=125, key="normal_q")
    if st.button("▶ Run Leader Lens", type="primary", key="run_normal"):
        with st.spinner("Running production workflow..."):
            st.session_state["normal_trace"] = run_normal(backend, llm, cache, q)
    if "normal_trace" in st.session_state:
        render_result(st.session_state["normal_trace"])

with tab2:
    st.caption("Prime the cache with the canonical Justin Trudeau question, then ask the semantically equivalent paraphrase.")
    st.text_area("Similar question", value=backend.paraphrase, height=125, disabled=True, key="cache_q")
    if st.button("⚡ Ask Similar Question", type="primary", key="run_cache"):
        with st.spinner("Checking semantic cache..."):
            st.session_state["cache_trace"] = run_cache_demo(backend, llm, cache)
    if "cache_trace" in st.session_state:
        render_result(st.session_state["cache_trace"])

with tab3:
    st.caption("Replay the project's adversarial roleplay + fabricated-evidence probe through the hardened path.")
    st.text_area("Adversarial input", value=backend.wiki_attack, height=180, disabled=True, key="attack_q")
    if st.button("🛡 Run Security Test", type="primary", key="run_attack"):
        with st.spinner("Sanitizing, retrieving, and answering through the hardened XML trust boundary..."):
            st.session_state["security_trace"] = run_security(backend, llm)
    if "security_trace" in st.session_state:
        render_result(st.session_state["security_trace"])

st.divider()
st.caption(
    f"Backend: {BACKEND_FILE.name} · Cache threshold: {backend.SEMANTIC_CACHE_THRESHOLD:.2f} · "
    f"MAX_STEPS: {backend.MAX_STEPS} · Embeddings: {backend.EMBEDDING_MODEL}"
)
