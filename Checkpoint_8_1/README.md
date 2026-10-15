# Leader Lens — Wikipedia Retrieval & Evidence Engine

**RAG and Context Engineering Capstone**  
**Author:** Lakshmi Muthukumar

Leader Lens is a retrieval-augmented generation (RAG) system for grounded, evidence-backed Wikipedia question answering on world leaders. The capstone develops the system incrementally from baseline retrieval through hybrid search, evaluation, agentic workflows, security hardening, production optimization, and a Streamlit demonstration.

> **Core principle:** Evidence before inference — retrieve the required sources, verify evidence coverage, then answer.

---

## Project Goals

Leader Lens was designed to:

- ground answers in a validated Wikipedia corpus rather than unsupported model memory;
- support precise factual retrieval, cross-article synthesis, and multi-hop reasoning;
- compare fixed, agentic, and tool-using retrieval strategies;
- measure correctness, completeness, groundedness, source compliance, latency, and retrieval quality;
- harden the RAG workflow against prompt injection and unsafe evidence handling;
- reduce production cost and latency through caching, routing, tiered models, and bounded agent behavior;
- expose the final system through a Streamlit interface.

The final validated corpus contains **65 Wikipedia world-leader articles**. The capstone presentation describes the system as a source-grounded, tool-using RAG workflow rather than a general-purpose chatbot.

---

## Capstone Progression

| Module | Focus | Main capability |
|---|---|---|
| 1 | Baseline retrieval | Establish the initial Wikipedia retrieval pipeline |
| 2 | Selectable / hybrid retrieval | Combine lexical and semantic retrieval |
| 3 | Evaluation | Build canonical questions, variants, ground truth, and retrieval evaluation |
| 4 | Advanced retrieval | Add query decomposition and graph-expanded retrieval |
| 5 | Agentic RAG | Compare Fixed, Agentic, and Tool-Using workflows |
| 6 | Security & cost | Test adversarial probes and a model security ladder |
| 7 | Production readiness | Add security middleware, cache, routing, quality gates, and evaluation |
| 8 | Final integration | Consolidate the capstone and provide a Streamlit demonstration |

---

## System Architecture

The final production-oriented workflow is:

```text
User Question
     |
     v
Security Middleware
     |
     v
Semantic Cache
  |        |
 HIT      MISS
  |        |
  v        v
Cached   Query Router
Answer    /       \
      DIRECT     AGENT
        |          |
 Hybrid Retrieval  Multi-step Retrieval
        |          |
        v          |
 Answer Quality    |
     Gate          |
   /    \          |
 PASS   FAIL ------+
   |               |
   +-------+-------+
           |
           v
    Grounded Answer
           |
           v
 Evaluation Framework
```

The production design can be summarized as:

**Protect & Reuse → Route & Retrieve → Check & Escalate → Ground & Measure**

The direct path handles simpler questions efficiently, while complex comparison and multi-hop questions can be routed or escalated to bounded agentic retrieval.

---

## Retrieval Strategy

Leader Lens combines:

- **BM25 lexical retrieval** for exact names, dates, offices, and terminology;
- **vector retrieval** for semantically related historical context;
- **hybrid fusion** to balance lexical precision and semantic recall;
- **query decomposition** for complex questions;
- **article search** for targeted source retrieval;
- **graph expansion** for related evidence;
- **bounded multi-step retrieval** for agentic workflows.

The final design intentionally spends model capability selectively rather than sending every question through the most expensive agent path.

---

## Agentic Workflows

Checkpoint/Module 5 compares three control strategies.

### Fixed Retrieval
One hybrid retrieval pass followed by an evidence-grounded answer. This is the fastest and most predictable baseline when the initial evidence is sufficient.

### Agentic Retrieval
The system retrieves evidence, inspects coverage, and refines the query when the evidence is weak or incomplete.

### Tool-Using Agent
The model selects among retrieval actions such as:

```text
hybrid_search
decompose_query
article_search
graph_expand
answer
```

Agent behavior is bounded to prevent uncontrolled retrieval loops.

---

## Evaluation

The system evaluates answer quality against ground-truth grading notes using four dimensions:

| Metric | Purpose |
|---|---|
| Correctness | Is the answer factually correct relative to the grading notes? |
| Completeness | Does it cover all requested components? |
| Groundedness | Are claims tied to retrieved evidence? |
| Source compliance | Were the required Wikipedia sources used? |

The overall score is derived from these component scores, with deterministic thresholds used for **PASS**, **PARTIAL**, and **FAIL** classification.

The final capstone also evaluates retrieval recall, routing, security safeguards, latency, token usage, cache behavior, and robustness to paraphrasing.

---

## Production Engineering

The later checkpoints extend the retrieval system with production-oriented controls:

**Security middleware.** Input sanitization, trust boundaries, and adversarial probes help prevent user instructions or retrieved text from redirecting the system.

**Semantic cache.** Validated high-similarity answers can be reused without repeating retrieval and generation.

**Query routing.** Simple questions can use a cheaper direct path while complex questions use agentic retrieval.

**Quality-gated escalation.** Weak direct-path answers can be escalated rather than accepted automatically.

**Tiered models.** Smaller models handle lightweight routing and filtering, while stronger models are reserved for more difficult reasoning.

**Bounded agents.** Multi-step retrieval is capped to control latency and token growth.

---

## Key Findings

The capstone demonstrated that retrieval depth and autonomy can improve evidence coverage, but they introduce measurable latency and cost trade-offs. The final evaluation also exposed an important distinction between **retrieving the right source** and **fully grounding every generated claim**.

A controlled semantic-cache demonstration reduced response latency from approximately **21.85 s to 0.066 s** and avoided **4,546 generation tokens** for a validated paraphrase hit.

The final production evaluation showed strong retrieval and safeguard behavior while identifying routing, source completeness, and claim-level grounding as the most important areas for continued engineering.

---

## Repository Structure

```text
RAG_ContextEng/
│
├── Checkpoint_1_1/
├── Checkpoint_2_1/
├── Checkpoint_3_1/
├── Checkpoint_4_1/
├── Checkpoint_5_1/
├── Checkpoint_6_1/
├── Checkpoint_7_1/
│
├── Checkpoint_8_1/
│   │
│   ├── Module_1/
│   │   ├── capstone_checkpoint_1_1_baseline_solution...
│   │   └── checkpoint_1_1_output.log
│   │
│   ├── Module_2/
│   │   ├── Capstone_Checkpoint_2_1_selectable_retr...
│   │   ├── checkpoint_2_1_retrieval_2.log
│   │   ├── Output_2.txt
│   │   └── README_Capstone_2_1_CHUNKED.md
│   │
│   ├── Module_3/
│   │   ├── capstone_checkpoint_3_1_evaluation_1.py
│   │   ├── nifty_minsky.csv
│   │   ├── nifty_minsky_variants.json
│   │   ├── README_Capstone_3_1.md
│   │   ├── variants_3_1_FIXED.py
│   │   ├── wikipedia_GT_15.xlsx
│   │   └── wikipedia_leader_lens_variants_eval.csv
│   │
│   ├── Module_4/
│   │   ├── capstone_checkpoint_4_1_GRAPH_EXPAN...
│   │   ├── checkpoint_4_1_advanced_retrieval_final.log
│   │   ├── Output_1.txt
│   │   └── README_Capstone_4_1.md
│   │
│   ├── Module_5/
│   │   ├── capstone_checkpoint_5_1_agent_rag_rea...
│   │   ├── checkpoint_5_1_evaluation_results.csv
│   │   ├── checkpoint_5_1_evaluation_summary.csv
│   │   ├── checkpoint_5_1_judged_report.txt
│   │   ├── checkpoint_5_1_openrouter.log
│   │   ├── evaluate_checkpoint_5_1_results.py
│   │   └── Output_3_mod.txt
│   │
│   ├── Module_6/
│   │   ├── capstone_checkpoint_6_1_security_cost_st...
│   │   ├── checkpoint_6_1_ladder_3.log
│   │   ├── leader_lens_security_probes_1.json
│   │   ├── Output_1.txt
│   │   └── README_Capstone_6_1.md
│   │
│   ├── Module_7/
│   │   ├── 7_1_model_production_2.log
│   │   ├── 7_2_evaluation_results_1.json
│   │   ├── Capstone_checkpoint_7_1.py
│   │   ├── checkpoint_7_1_test_cases_1.json
│   │   ├── Output_mod_1.txt
│   │   └── RAG System Flowchart with Escalation Pa...
│   │
│   └── Module_8/
│       ├── LeaderLens_Capstone_notes.pdf
│       └── Streamlit/
│           ├── 7_2_evaluation_results_1.json
│           ├── Capstone_checkpoint_7_1.py
│           └── leader_lens_app_1.py
│
└── README.md
```

> **Note:** A few long filenames above are displayed with `...` because Windows Explorer truncates them in the available repository screenshots. They can be replaced with their exact names after running `tree /F /A` from the repository root.

---

## Final Module: Streamlit Demo

`Checkpoint_8_1/Module_8/Streamlit/` contains the application layer for demonstrating Leader Lens interactively.

The demo brings together the production pipeline and evaluation artifacts:

```text
Streamlit UI
    ↓
Leader Lens production pipeline
    ↓
Protect → Route → Retrieve → Validate → Ground
    ↓
Evidence-backed response
```

The Streamlit directory includes the application script, the production capstone implementation, and evaluation results used by the final system.

---

## Lessons Learned

The project produced five central engineering lessons:

1. **Retrieval is not the same as grounding.** Finding the correct source does not guarantee that every generated claim is supported.
2. **More agentic is not always better.** Agentic retrieval is valuable for multi-hop questions, while simple lookups benefit from a faster direct path.
3. **Evidence completeness must be explicit.** Required articles and sub-questions should remain pending until evidence obligations are satisfied.
4. **Optimization needs guardrails.** Caching, routing, smaller models, and bounded agents save work only when paired with validation and escalation.
5. **Evaluation drives architecture.** Failures in routing, grounding, source coverage, latency, and cost become requirements for the next engineering iteration.

---

## Future Improvements

The next engineering priorities are to:

- improve router accuracy with deterministic complexity signals;
- track required articles and sub-questions as explicit retrieval obligations;
- retry targeted article retrieval before allowing completion;
- add claim-to-evidence coverage checks;
- deduplicate retained passages and summarize intermediate agent state;
- stop retrieval once all evidence requirements are satisfied;
- broaden semantic-cache validation;
- expand evaluation beyond the current capstone-scale test set.

---

## Technology Stack

The project uses Python-based RAG and evaluation tooling, including lexical and vector retrieval, persistent vector storage, graph-based evidence expansion, LLM-based planning/judging, JSON/CSV/XLSX evaluation artifacts, and Streamlit for the final interactive demonstration.

---

## Conclusion

Leader Lens evolved from a basic Wikipedia retrieval experiment into a measurable, security-aware, production-oriented RAG architecture.

The central design principle remained constant throughout the capstone:

> **Evidence before inference.**

A capable retrieval system should not merely retrieve more information. It should retrieve the **right evidence**, determine when enough evidence has been collected, preserve source support through generation, and measure whether the resulting answer is actually grounded.
