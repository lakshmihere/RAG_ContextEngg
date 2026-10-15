
## Capstone Project

**Leader Lens** is a Wikipedia-based Retrieval-Augmented Generation
(RAG) capstone project developed incrementally across eight checkpoints.
The goal is to build a conversational retrieval system that answers
questions about significant people, places, organizations, landmarks,
and related topics using evidence retrieved from a controlled collection
of Wikipedia articles.

The system evolves from a basic retrieval baseline into a
production-oriented RAG architecture with hybrid retrieval, advanced
retrieval strategies, agentic workflows, security testing, cost and
latency optimization, evaluation, and final capstone documentation.

The project supports both **single-document** and **multi-document**
questions and emphasizes grounded answers, source transparency,
retrieval quality, and reduced hallucination risk.

------------------------------------------------------------------------

## Repository Structure

``` text
RAG_ContextEng/
│
├── Checkpoint_1_1/
│   ├── capstone_checkpoint_1_1_baseline_solution...
│   ├── checkpoint_1_1_responses_1.log
│   ├── Required_Capstone_Checkpoint_1_1_Worksheet...
│   └── WikipediaRetrievalEngine.docx
│
├── Checkpoint_2_1/
│   ├── Capstone_Checkpoint_2_1_HYBRID_RETRI...
│   ├── Capstone_Checkpoint_2_1_selectable_retr...
│   ├── checkpoint_2_1_retrieval_1.log
│   ├── Output_2_1.txt
│   ├── Required_Capstone_Checkpoint_2_1_Work...
│   └── Required_Capstone_Checkpoint_2_1_Work...
│
├── Checkpoint_3_1/
│   ├── capstone_checkpoint_3_1_repaired_LMSt...
│   ├── checkpoint_3_1_hybrid_tagged.csv
│   ├── checkpoint_3_1_hybrid_tagged.json
│   ├── checkpoint_3_1_hybrid_tagged_variants.json
│   ├── checkpoint_3_1_retrieval_LMStudio_2.log
│   ├── Required_Capstone_Checkpoint_3_1_Work...
│   ├── variants_3_1.py
│   └── wikipedia_capstone_3_1_variants.csv
│
├── Checkpoint_4_1/
│   ├── Wikipedia_10_text/
│   ├── Wikipedia_chroma_topics_full/
│   ├── capstone_4_1.png
│   ├── capstone_checkpoint_4_1_advanced_retri...
│   ├── checkpoint_4_1_advanced_retrieval_1.log
│   ├── Output_log_1.txt
│   ├── README_capstone_4_1.md
│   ├── Required_Capstone_Checkpoint_4_1_Resu...
│   └── wikipedia_topics_10.json
│
├── Checkpoint_5_1/
│   ├── wikipedia_5_1_chroma_hf/
│   ├── Wikipedia_10_text/
│   ├── Capstone_Checkpoint_5_1_Final.py
│   ├── Capstone_Checkpoint_5_1_Worksheet_fin...
│   ├── Capstone_checkpoint_output_5_1.txt
│   ├── Checkpoint_output_5_1_1.txt
│   ├── Readme_5_1.png
│   ├── README_Capstone_5_1.md
│   └── Tool_using_agent_1.log
│
├── Checkpoint_6_1/
│   ├── 6_1_model_ladder_3.log
│   ├── 6_1_model_ladder_comparison_3.json
│   ├── checkpoint_6_1_SECURITY_MODEL_LADD...
│   ├── checkpoint_6_1_test_cases.json
│   ├── mermaid_diagram_6.png
│   ├── Output_final_3.txt
│   ├── README_checkpoint_6_1_SECURITY_MO...
│   └── Required_Capstone_Checkpoint_6_1_Wor...
│
├── Checkpoint_7_1/
│   ├── wikipedia_7_1_chroma_hf/
│   ├── Wikipedia_10_text/
│   ├── 7_1_model_production_2.log
│   ├── 7_2_evaluation_results.json
│   ├── Capstone_checkpoint_7_1.py
│   ├── checkpoint_7_1_test_cases.json
│   ├── mermaid_7.png
│   ├── Output_7_3.txt
│   ├── README_Capstone_7.md
│   └── Required_Capstone_Checkpoint_7_1_Wor...
│
├── Checkpoint_8_1/
│   ├── Module_1/
│   ├── Module_2/
│   ├── Module_3/
│   ├── Module_4/
│   ├── Module_5/
│   ├── Module_6/
│   ├── Module_7/
│   └── Module_8/
│
└── README.md
```

Each checkpoint builds on the previous stage of the same retrieval
system. The folders contain the implementation code, worksheets,
evaluation artifacts, logs, outputs, diagrams, and supporting files used
at that stage.

------------------------------------------------------------------------

## Checkpoint Overview

  -----------------------------------------------------------------------
  Checkpoint              Focus                   Description
  ----------------------- ----------------------- -----------------------
  **1.1**                 Baseline Retrieval      Establishes the initial
                                                  Wikipedia
                                                  retrieval-engine
                                                  baseline, project
                                                  requirements, response
                                                  logs, and worksheet
                                                  documentation.

  **2.1**                 Selectable / Hybrid     Introduces multiple
                          Retrieval               retrieval approaches
                                                  and hybrid retrieval,
                                                  with Python
                                                  implementations,
                                                  retrieval logs, output
                                                  files, and checkpoint
                                                  worksheet results.

  **3.1**                 Retrieval Evaluation    Extends retrieval
                                                  evaluation using tagged
                                                  questions, JSON/CSV
                                                  evaluation sets,
                                                  variants, logs, and
                                                  analysis of retrieval
                                                  behavior.

  **4.1**                 Advanced Retrieval      Adds advanced retrieval
                                                  techniques, topic
                                                  information,
                                                  graph-oriented
                                                  retrieval support, a
                                                  Wikipedia text corpus,
                                                  persistent retrieval
                                                  data, diagrams, logs,
                                                  and README
                                                  documentation.

  **5.1**                 Agentic RAG             Develops agent-based
                                                  retrieval workflows and
                                                  tool-using behavior,
                                                  supported by a
                                                  persistent vector
                                                  store, Wikipedia
                                                  corpus, execution logs,
                                                  outputs, worksheet
                                                  results, and
                                                  architecture
                                                  documentation.

  **6.1**                 Security & Model        Evaluates RAG security
                          Evaluation              and reliability using
                                                  adversarial test cases
                                                  and a model-security
                                                  ladder. Includes
                                                  model-comparison
                                                  results, test cases,
                                                  logs, outputs,
                                                  diagrams, and
                                                  checkpoint
                                                  documentation.

  **7.1**                 Production Readiness &  Hardens the system for
                          Optimization            production through
                                                  evaluation, routing,
                                                  caching,
                                                  performance/cost
                                                  optimization,
                                                  persistent retrieval
                                                  infrastructure, test
                                                  cases, architecture
                                                  diagrams, and
                                                  production-readiness
                                                  documentation.

  **8.1**                 Final Capstone          Organizes Modules 1--8
                          Integration             and consolidates the
                                                  work used for the final
                                                  capstone review,
                                                  presentation, and
                                                  end-to-end project
                                                  narrative.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Checkpoint 1.1 --- Baseline Retrieval

This checkpoint establishes the initial Wikipedia Retrieval Engine and
the baseline from which later RAG capabilities are developed.

**Repository contents include:** - Baseline Python solution - Response
log - Required checkpoint worksheet - Wikipedia Retrieval Engine
documentation

**Purpose:** establish the initial retrieval workflow and document the
baseline system behavior.

------------------------------------------------------------------------

## Checkpoint 2.1 --- Hybrid Retrieval

Checkpoint 2.1 expands the baseline into a selectable retrieval system
and introduces **hybrid retrieval**.

**Repository contents include:** - Hybrid-retrieval Python
implementation - Selectable-retriever Python implementation - Retrieval
log - Execution output - Required checkpoint worksheets

**Purpose:** compare retrieval strategies and create a stronger
retrieval layer that can combine complementary retrieval methods.

------------------------------------------------------------------------

## Checkpoint 3.1 --- Retrieval Evaluation

Checkpoint 3.1 focuses on systematic evaluation of retrieval
performance.

**Repository contents include:** - Retrieval/evaluation Python code -
Hybrid-tagged CSV results - Hybrid-tagged JSON evaluation data -
Query-variant JSON data - Variant-generation/evaluation script -
Evaluation logs - Required checkpoint worksheet

**Purpose:** move from qualitative retrieval testing toward repeatable,
structured evaluation using defined question sets and variants.

------------------------------------------------------------------------

## Checkpoint 4.1 --- Advanced Retrieval

Checkpoint 4.1 introduces more advanced retrieval strategies and
supporting topic/graph information.

**Repository contents include:** - `Wikipedia_10_text/` corpus -
Persistent Chroma/topic data - Advanced-retrieval Python
implementation - Retrieval logs and output - Topic JSON data -
Architecture/flow image - Checkpoint README - Required checkpoint
results worksheet

**Purpose:** improve retrieval for more complex questions by expanding
beyond a simple single-query retrieval path.

------------------------------------------------------------------------

## Checkpoint 5.1 --- Agentic RAG

Checkpoint 5.1 evolves the retrieval engine into an **agentic RAG
system** capable of selecting and using retrieval tools as part of a
multi-step workflow.

**Repository contents include:** - Wikipedia text corpus - Persistent
Chroma vector store - Agentic RAG Python implementation - Tool-using
agent log - Execution/output files - Architecture image - Checkpoint
README - Completed checkpoint worksheet

**Purpose:** evaluate when a fixed retrieval pipeline is sufficient and
when multi-step, tool-using retrieval can improve complex-question
handling.

------------------------------------------------------------------------

## Checkpoint 6.1 --- Security and Model Evaluation

Checkpoint 6.1 focuses on the security, reliability, and model-dependent
behavior of the RAG system.

**Repository contents include:** - Security/model-ladder Python
implementation - Security test cases in JSON - Model comparison results
in JSON - Model execution log - Final output report - Security
architecture diagram - Security checkpoint README - Required checkpoint
worksheet

**Purpose:** test the system against adversarial inputs and compare how
different models behave under security and reliability probes.

------------------------------------------------------------------------

## Checkpoint 7.1 --- Production Readiness

Checkpoint 7.1 brings the system closer to a production RAG architecture
by combining safeguards, routing, evaluation, caching, and efficiency
improvements.

**Repository contents include:** - Production-oriented Python
implementation - Wikipedia text corpus - Persistent Chroma vector
store - Production model log - Evaluation-results JSON - Test-case
JSON - Output report - Production architecture diagram - Checkpoint
README - Required checkpoint worksheet

**Purpose:** evaluate the trade-offs among retrieval quality,
groundedness, security, latency, token usage, cost, and operational
complexity.

The production-oriented workflow can be summarized as:

``` text
User Question
     |
     v
Security / Input Protection
     |
     v
Semantic Cache
     |
     v
Query Router
   /     \
Direct   Agentic
 Path     Path
   \       /
    Retrieval
       |
       v
Quality / Grounding Check
       |
       v
Grounded Answer
       |
       v
Evaluation & Metrics
```

------------------------------------------------------------------------

## Checkpoint 8.1 --- Final Capstone Integration

Checkpoint 8.1 organizes the complete project across **Modules 1--8**
and supports the final capstone review and presentation.

``` text
Checkpoint_8_1/
├── Module_1/
├── Module_2/
├── Module_3/
├── Module_4/
├── Module_5/
├── Module_6/
├── Module_7/
└── Module_8/
```

**Purpose:** consolidate the development history and demonstrate how the
system progressed from a baseline retrieval engine to a more secure,
evaluated, and production-oriented RAG system.

------------------------------------------------------------------------

## System Evolution

``` text
Checkpoint 1.1
Baseline Retrieval
      ↓
Checkpoint 2.1
Hybrid / Selectable Retrieval
      ↓
Checkpoint 3.1
Structured Retrieval Evaluation
      ↓
Checkpoint 4.1
Advanced Retrieval
      ↓
Checkpoint 5.1
Agentic + Tool-Using RAG
      ↓
Checkpoint 6.1
Security + Model Evaluation
      ↓
Checkpoint 7.1
Production Hardening + Optimization
      ↓
Checkpoint 8.1
Final Capstone Integration
```

------------------------------------------------------------------------

## Key Engineering Themes

Across the checkpoints, the repository demonstrates the progressive
development of a RAG application around several core engineering
concerns:

-   **Retrieval:** moving from baseline retrieval toward hybrid and
    advanced retrieval.
-   **Grounding:** answering from retrieved Wikipedia evidence rather
    than relying only on model knowledge.
-   **Evaluation:** using repeatable question sets, variants, logs, and
    structured result files.
-   **Agentic workflows:** allowing complex questions to use multi-step
    retrieval and specialized tools.
-   **Security:** testing prompt-injection and other adversarial
    behaviors.
-   **Efficiency:** evaluating routing, caching, model selection,
    latency, tokens, and operational cost.
-   **Production readiness:** combining safeguards, retrieval,
    validation, evaluation, and optimization into a coherent system.
-   **Documentation:** maintaining worksheets, README files, diagrams,
    logs, outputs, and evaluation artifacts alongside the
    implementation.

------------------------------------------------------------------------

## Repository Artifact Types

The repository contains several kinds of project artifacts:

  -----------------------------------------------------------------------
  Artifact                            Role
  ----------------------------------- -----------------------------------
  `.py`                               Retrieval, agentic, security,
                                      evaluation, and production code

  `.json`                             Test cases, evaluation sets, topic
                                      data, and structured results

  `.csv`                              Evaluation and retrieval-analysis
                                      results

  `.log`                              Execution traces and diagnostic
                                      logs

  `.txt`                              Captured program output and reports

  `.docx`                             Required checkpoint worksheets and
                                      project documentation

  `.png`                              Architecture, workflow, and README
                                      diagrams

  `.md`                               Checkpoint and repository
                                      documentation

  Chroma folders                      Persistent vector/retrieval indexes

  Wikipedia text folders              Local source corpus used by
                                      retrieval workflows
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Final Project Goal

The completed capstone demonstrates the progression from a basic
Wikipedia question-answering retrieval engine to a **grounded,
evaluated, security-aware, and production-oriented RAG system**.

The final design emphasizes a practical workflow:

**Protect → Route → Retrieve → Validate → Ground → Measure**

This progression makes the repository not just a collection of
checkpoint submissions, but a record of the engineering decisions,
experiments, failures, improvements, and trade-offs involved in building
a trustworthy retrieval-augmented application.
