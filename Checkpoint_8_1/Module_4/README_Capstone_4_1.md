# Capstone Checkpoint 4.1 --- Advanced Retrieval Implementation

## Wikipedia Leader Lens: Query Decomposition + Graph-Augmented Retrieval

This checkpoint extends the **Wikipedia Leader Lens** retrieval pipeline
from baseline retrieval and evaluation into **advanced retrieval**. It
compares four retrieval strategies over the same 65-article Wikipedia
world-leaders corpus:

1.  a simple lexical baseline,
2.  hybrid BM25 + vector retrieval,
3.  LLM-driven multi-step query decomposition, and
4.  graph-augmented retrieval seeded by hybrid search.

The purpose of Checkpoint 4.1 is not simply to add more retrieval
mechanisms. It is to examine whether **query decomposition** and **graph
expansion** improve evidence coverage for complex, multi-part,
multi-entity, and exact-evidence questions---and to diagnose cases where
advanced retrieval still fails.

------------------------------------------------------------------------

## 1. System Architecture

``` text
                         User Question
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        Lexical Baseline    Hybrid        Query Decomposition
        keyword overlap    Retrieval        2–4 sub-queries
              |          BM25 0.4 +            |
              |          Vector 0.6             v
              |               |          retrieve each sub-query
              |               |          + sum document scores
              |               |               |
              |               v               v
              |          Hybrid seed       MULTI result
              |          articles
              |               |
              |               v
              |        Wikipedia Graph
              |       +---------------+
              |       | links_to      |
              |       | relates_to*   |
              |       +---------------+
              |               |
              |               v
              |       Graph expansion
              |     seed > linked > topic
              |               |
              +---------------+---------------+
                              |
                              v
                    Compare retrieved sets
                              |
                              v
                 Graph-selected evidence
                              |
                              v
             Query-relevant article windows
               max context ≈ 16,000 chars
                              |
                              v
                    Grounded generation
                              |
                              v
                         Answer + log
```

`* relates_to` topic expansion is available when topic metadata is
present. In the supplied run, the topic metadata file was not found, so
the graph was built from article links only.

------------------------------------------------------------------------

## 2. Project Files

A typical Checkpoint 4.1 folder contains:

``` text
check_4/
│
├── capstone_checkpoint_4_1_GRAPH_EXPANDED.py
├── checkpoint_4_1_advanced_retrieval_final.log
├── Output_1.txt
│
├── Wikipedia_text/
│   ├── Abraham_Lincoln.txt
│   ├── Adolf_Hitler.txt
│   ├── Barack_Obama.txt
│   ├── Boris_Johnson.txt
│   ├── Cleopatra.txt
│   ├── Margaret_Thatcher.txt
│   ├── Nelson_Mandela.txt
│   ├── Suharto.txt
│   └── ... 65 articles total
│
├── Wikipedia_chroma_full/
│
└── wikipedia_topics_10.json       # optional topic metadata
```

------------------------------------------------------------------------

## 3. Environment Setup

Create and activate a Python environment, then install the required
packages.

``` powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install python-dotenv networkx rank-bm25 langchain-core langchain-openai langchain-chroma langchain-huggingface
```

Create a `.env` file in the project directory:

``` text
OPENROUTER_API_KEY=your-openrouter-key
```

Do not commit API keys to source control.

------------------------------------------------------------------------

## 4. Main Configuration

  Setting                   Value
  ------------------------- -----------------------------------------------
  Scenario                  `wikipedia`
  Corpus directory          `Wikipedia_text`
  Observed corpus size      65 TXT articles
  Chroma directory          `Wikipedia_chroma_full`
  Top results               3
  Hybrid candidate pool     20
  BM25 weight               0.40
  Vector weight             0.60
  Embedding model           `openai/text-embedding-3-small`
  Main configured LLM       `openai/gpt-5.4-mini`
  Answer temperature        0.0
  Graph seed count          2
  Max linked expansion      3
  Max topic expansion       3
  Optional topic metadata   `wikipedia_topics_10.json`
  Output log                `checkpoint_4_1_advanced_retrieval_final.log`
  Test queries              15

The code also defines:

``` text
DECOMPOSITION_MODEL = deepseek-coder-6.7b-instruct
ANSWER_MODEL        = meta-llama-3.1-8b-instruct
```

However, the current execution path creates its LLM through
`make_llm()`, whose configured model is `openai/gpt-5.4-mini`.
Therefore, the README reports the model actually used by this run rather
than assuming the two additional constants are independently wired into
the pipeline.

------------------------------------------------------------------------

## 5. Corpus Loading

The system recursively loads `.txt` Wikipedia articles from:

``` text
Wikipedia_text/
```

The supplied run reports:

``` text
Found 65 Wikipedia TXT files.
Successfully loaded 65 Wikipedia text documents.
```

The document ID is the relative TXT filename, for example:

``` text
Abraham_Lincoln.txt
Margaret_Thatcher.txt
Nelson_Mandela.txt
Presidency_of_Joe_Biden.txt
Suharto.txt
```

Unlike the chunk-level Checkpoint 3.1 pipeline, this Checkpoint 4.1
implementation ranks **full articles**. Before answer generation, it
extracts smaller query-relevant windows from those articles so the
evidence remains compatible with a limited local-model context window.

------------------------------------------------------------------------

## 6. Compact Evidence Windows

Sending several complete Wikipedia articles to an 8K-context local model
can exceed the available context.
`BaseRetriever._best_article_snippet()` therefore scans each retrieved
article using overlapping windows.

Default behavior:

``` text
window size       = 1,200 characters
window overlap    = 200 characters
windows/article   = 2
max total context = 16,000 characters
```

Windows are scored using query-token overlap, and earlier text receives
a slight preference on ties because Wikipedia lead sections frequently
contain the defining biographical facts.

This creates an important distinction:

``` text
Retrieval unit  = full Wikipedia article
Generation unit = compact query-relevant article excerpt
```

The design reduces context pressure, but passage selection can still
fail even when the correct article is retrieved.

------------------------------------------------------------------------

## 7. Retrieval Strategy A --- Baseline

`baseline_retrieve()` is a simple single-pass lexical baseline.

It:

1.  tokenizes the query and each full article,
2.  counts shared words,
3.  ranks all documents once,
4.  returns the top `k`.

Conceptually:

``` text
query
  |
  v
shared-word scoring
  |
  v
rank 65 articles
  |
  v
top 3
```

This intentionally simple method provides a reference point against
which the advanced methods can be compared.

------------------------------------------------------------------------

## 8. Retrieval Strategy B --- Hybrid Retrieval

`HybridRetriever` combines BM25 and Chroma vector retrieval.

### BM25 branch

The system tokenizes all full articles, removes common stopwords, and
calculates BM25 scores.

### Vector branch

Chroma performs semantic similarity search using:

``` text
openai/text-embedding-3-small
```

### Fusion

The top candidate pools are normalized separately.

For BM25:

``` text
higher score = better
```

For Chroma distance:

``` text
lower distance = better
```

The vector distances are therefore inverted after min-max normalization.

Final score:

``` text
hybrid_score =
    0.40 × normalized_BM25
  + 0.60 × normalized_vector_similarity
```

The highest fused scores become the hybrid results.

------------------------------------------------------------------------

## 9. Retrieval Strategy C --- Multi-Step Query Decomposition

The first advanced technique is **query decomposition**.

`decompose_query()` asks the LLM to convert a complex question into
approximately 2--4 focused sub-queries.

Example:

``` text
Original:
Compare Nelson Mandela and Suharto using their respective Wikipedia
articles. What country did each lead, what office did each hold, and
how did their periods of leadership differ?

                    |
                    v

Sub-query 1:
What country did Nelson Mandela lead and what office did he hold?

Sub-query 2:
What country did Suharto lead and what office did he hold?

Sub-query 3:
How did their periods of leadership differ?
```

For each sub-query, `multistep_retrieve()` retrieves an expanded
candidate set using the baseline retriever.

``` text
candidate count per sub-query = 3 × k
```

Scores are accumulated across the sub-queries:

``` text
document_score =
    score(sub-query 1)
  + score(sub-query 2)
  + ...
```

Documents relevant to several components of the original question can
therefore rise in the final ranking.

If decomposition cannot be parsed as a valid JSON list, the system
safely falls back to:

``` python
[original_query]
```

------------------------------------------------------------------------

## 10. Retrieval Strategy D --- Graph-Augmented Retrieval

The second advanced technique is graph-based retrieval.

### Graph nodes

The implemented graph can contain:

``` text
doc:<article>
topic:<topic>
```

The advanced-retrieval plan conceptually describes:

``` text
article
category
topic
```

### Graph relationships

The graph builder currently creates relationships from:

``` text
links_to
relates_to
```

`links_to` is derived from Wikipedia-style links found in the article
text when the linked target also exists in the 65-document corpus.

`relates_to` is created when optional topic/category/tag metadata is
available.

### Retrieval flow

``` text
query
  |
  v
Hybrid retrieval
  |
  v
top 2 seed articles
  |
  v
Graph expansion
  |
  +--> directly linked articles
  |
  +--> topic-sharing articles
  |
  v
rank expansion candidates
  |
  v
top 3 final articles
```

Seed articles retain their exact hybrid scores. Expanded graph
candidates receive lexical query-overlap scores plus a source-class
bonus.

Priority is:

``` text
seed > linked > topic
```

The diagnostic method `get_sources_with_labels()` makes this visible in
the output:

``` text
('Nelson_Mandela.txt', 'seed')
('Boris_Johnson.txt', 'seed')
('James_A._Garfield.txt', 'linked')
```

------------------------------------------------------------------------

## 11. Important Graph Metadata Result

The supplied run prints:

``` text
Topic metadata not found at wikipedia_topics_10.json;
graph will use text links only.

Wikipedia graph ready: 65 nodes, 39 edges.
```

This is important when interpreting the experiment.

For this execution:

``` text
article/document nodes = active
links_to edges         = active
topic nodes            = not populated
relates_to expansion   = not populated
```

Therefore, this run is best described as **link-based graph expansion
seeded by hybrid retrieval**, not a full topic-enriched graph
experiment.

Adding the intended metadata file would allow the `relates_to` branch to
participate in later experiments.

------------------------------------------------------------------------

## 12. Grounded Answer Generation

The graph retriever is used for final answer generation in `run_demo()`.

The answer prompt requires the model to:

-   use only the supplied Wikipedia evidence,
-   quote evidence when possible,
-   avoid filling missing evidence with outside knowledge,
-   explicitly say when the evidence is insufficient.

For comparison questions, the graph-specific prompt also emphasizes
using evidence from all required articles.

This behavior is visible in the run: when the correct article is
missing, the model frequently refuses to invent the answer.

That is a **retrieval failure but a grounding success**.

------------------------------------------------------------------------

## 13. Test Set

Checkpoint 4.1 evaluates **15 representative questions** covering
several retrieval difficulties.

  -----------------------------------------------------------------------
  Question group                      Main retrieval challenge
  ----------------------------------- -----------------------------------
  Mandela, Victoria, Thatcher,        Single-article factual retrieval
  Lincoln, Macron                     

  Trudeau, Hitler, Suharto            Multiple facts from one target
                                      article

  Washington/Lincoln/Obama            Multi-article evidence coverage

  Thatcher/May                        Two-article comparison

  Mandela/Suharto                     Cross-article comparison

  Boris/Truss/Sunak                   Multi-article temporal ordering

  George Washington opening           Exact quotation

  Cleopatra opening                   Exact quotation

  Presidency of Joe Biden             Article-title disambiguation
  -----------------------------------------------------------------------

These questions make it possible to compare the retrieval sets returned
by all four strategies rather than judging only the final generated
answer.

------------------------------------------------------------------------

## 14. Running the Checkpoint

From PowerShell:

``` powershell
python .\capstone_checkpoint_4_1_GRAPH_EXPANDED.py
```

The program prints:

``` text
BASELINE
HYBRID
decomposed into ...
MULTI
GRAPH
LABELS
ANSWER
```

for every test query.

It also writes graph-generated answers to:

``` text
checkpoint_4_1_advanced_retrieval_final.log
```

------------------------------------------------------------------------

## 15. Example Output Structure

For each query, the console comparison looks conceptually like:

``` text
Q1: <question>

BASELINE: [...]
HYBRID:   [...]

decomposed into:
[
  "...",
  "...",
  "..."
]

MULTI:    [...]
GRAPH:    [...]
LABELS:   [('...', 'seed'), ('...', 'linked')]

[answer generation] sending to LM Studio | context chars=...
[answer generation] LM Studio response received

ANSWER: ...
```

This output is useful because it exposes the retrieval behavior before
generation.

------------------------------------------------------------------------

## 16. Observed Results

The run demonstrates that advanced retrieval is **not automatically
superior** to the simpler retrievers.

### Successful example --- Nelson Mandela

For Q1:

``` text
BASELINE:
Nelson_Mandela.txt
Abraham_Lincoln.txt
Barack_Obama.txt

HYBRID:
Nelson_Mandela.txt
Boris_Johnson.txt
George_W._Bush.txt

GRAPH:
Nelson_Mandela.txt
Boris_Johnson.txt
```

The graph path retained the correct Mandela article, and the grounded
answer identified his presidency and service period.

### Useful multi-step example --- Margaret Thatcher vs Theresa May

For the Thatcher/May comparison, the multi-step retrieval returned:

``` text
Boris_Johnson.txt
Margaret_Thatcher.txt
Theresa_May.txt
```

This demonstrates the intended benefit of decomposition: individual
sub-questions can increase coverage of multiple target entities.

The graph result, however, contained:

``` text
Boris_Johnson.txt
Theresa_May.txt
```

and omitted the direct Margaret Thatcher article.

### Useful multi-step example --- Presidency of Joe Biden

For the article-specific Biden question, the hybrid and graph paths
missed the requested article, while multi-step retrieval included:

``` text
Presidency_of_Joe_Biden.txt
```

This shows why retrieval strategies should be evaluated independently
instead of assuming that one method wins on every query.

------------------------------------------------------------------------

## 17. Major Failure Pattern --- Seed Quality Dominates Graph Quality

Several graph failures begin before graph expansion.

Examples from the run include:

``` text
Queen Victoria
GRAPH -> Boris_Johnson.txt, Julius_Caesar.txt

Margaret Thatcher
GRAPH -> Boris_Johnson.txt, Rishi_Sunak.txt

George Washington
GRAPH -> Boris_Johnson.txt, Nelson_Mandela.txt

Emmanuel Macron
GRAPH -> Boris_Johnson.txt, Rishi_Sunak.txt
```

The graph is seeded by hybrid retrieval. If the hybrid seeds are
unrelated to the requested entity, graph traversal expands around the
**wrong neighborhood**.

Conceptually:

``` text
poor seed
   |
   v
graph expansion
   |
   v
more evidence near the poor seed
   |
   v
still poor final evidence
```

This is the most important diagnostic result of the current graph
experiment.

------------------------------------------------------------------------

## 18. Multi-Step Retrieval Is Also Imperfect

Query decomposition improves some multi-entity cases, but the
implementation still retrieves each sub-query using the simple lexical
baseline.

As a result, decomposition can produce good focused questions while the
underlying retrieval step still ranks unrelated documents.

For example, the code may correctly decompose:

``` text
What nationality is Justin Trudeau?
What leadership office did Justin Trudeau hold?
What were his term dates?
```

yet the final MULTI set can still omit `Justin_Trudeau.txt`.

Therefore:

``` text
better query formulation
        does not guarantee
better evidence retrieval
```

A logical future refinement is to use the hybrid retriever---not the
simple lexical baseline---for each decomposed sub-query.

------------------------------------------------------------------------

## 19. Exact-Quotation Failure

The Cleopatra query demonstrates another retrieval layer.

The graph result included:

``` text
Julius_Caesar.txt
Cleopatra.txt
```

so the correct article was present.

However, the compact snippet selected from the Cleopatra article
contained a references/links section rather than the article's opening
sentence.

The model therefore correctly refused to fabricate the exact quote.

This shows three separate retrieval levels:

``` text
1. Correct article retrieval
          |
          v
2. Correct passage/window selection
          |
          v
3. Correct grounded answer
```

Checkpoint 4.1 reveals that success at level 1 does not guarantee
success at level 2.

------------------------------------------------------------------------

## 20. Multi-Article Coverage Failure

For the Nelson Mandela vs Suharto comparison:

``` text
HYBRID:
Nelson_Mandela.txt
Bill_Clinton.txt
Suharto.txt
```

The hybrid retriever actually contained both target leaders.

The graph path returned:

``` text
Nelson_Mandela.txt
Bill_Clinton.txt
James_A._Garfield.txt
```

and lost `Suharto.txt`.

The final grounded answer therefore refused to complete the Suharto half
of the comparison.

This is a strong example of why graph expansion should not blindly
displace directly relevant hybrid evidence.

------------------------------------------------------------------------

## 21. What the Checkpoint Demonstrates

The current implementation successfully demonstrates the **mechanics**
of advanced retrieval:

-   LLM query decomposition;
-   score aggregation across sub-queries;
-   hybrid seed retrieval;
-   NetworkX graph construction;
-   Wikipedia link resolution;
-   optional topic metadata;
-   graph-neighbor expansion;
-   seed/linked/topic provenance labels;
-   compact evidence selection;
-   grounded generation;
-   side-by-side retrieval diagnostics.

The run also exposes important limitations that should be discussed
rather than hidden:

-   hybrid seeds can be poor;
-   graph expansion inherits seed errors;
-   missing topic metadata limits the graph;
-   decomposition currently relies on the weak lexical baseline;
-   graph expansion can displace a directly relevant hybrid result;
-   article-level retrieval can succeed while passage-window selection
    fails;
-   a top-3 limit is restrictive for multi-entity questions.

------------------------------------------------------------------------

## 22. Recommended Refinements

The observed results suggest several concrete improvements for the next
iteration:

1.  **Entity-aware seed protection**\
    If a query explicitly names `Margaret Thatcher`, `Suharto`, or
    `Queen Victoria`, strongly favor the corresponding article as a
    seed.

2.  **Use hybrid retrieval inside multi-step retrieval**\
    Replace `baseline_retrieve()` for each decomposed sub-query with the
    stronger hybrid retriever.

3.  **Preserve direct target hits during graph expansion**\
    Do not allow a weak graph neighbor to displace a directly relevant
    target article.

4.  **Load topic/category metadata**\
    Supply the intended metadata file so `relates_to` edges can be
    evaluated.

5.  **Improve passage/window selection**\
    Add lead-section awareness or dedicated exact-quotation handling.

6.  **Increase source diversity for comparison questions**\
    Reserve result slots for each named entity.

7.  **Evaluate retrieval metrics separately**\
    Measure target-source coverage, source diversity, and passage hit
    rate in addition to answer quality.

These are recommended improvements based on the supplied run; they are
not claimed as already implemented.

------------------------------------------------------------------------

## 23. Advanced Retrieval Plan

`my_advanced_plan()` defines the checkpoint strategy as:

``` text
technique = both
```

meaning:

``` text
query decomposition
        +
graph-based retrieval
```

The plan identifies conceptual node types:

``` text
article
category
topic
```

and relationships:

``` text
belongs_to_category
category_contains
relates_to
```

Its rationale is that complex Wikipedia questions often require several
facts or entities. Query decomposition isolates those needs, while graph
retrieval is intended to discover related evidence that a single ranked
retrieval pass might miss.

### Important implementation note

The plan is partly conceptual.

The graph builder in the current script directly implements:

``` text
document/article nodes
topic nodes when metadata exists

links_to
relates_to
```

The category-specific `belongs_to_category` and `category_contains`
relationships listed in the plan are not separately constructed by the
current `build_graph()` function.

The README therefore distinguishes the **design plan** from the
**executed graph structure**.

------------------------------------------------------------------------

## 24. Interpreting the Current Graph Statistics

Observed execution:

``` text
Wikipedia graph ready: 65 nodes, 39 edges.
```

Because no topic metadata was loaded:

``` text
65 document nodes
0 populated topic nodes from metadata
39 resolved text-link edges
```

This graph is relatively sparse.

That sparsity helps explain why graph expansion frequently has little
useful evidence to add beyond the hybrid seeds.

A future topic-enriched graph should be evaluated separately because it
would represent a materially different retrieval structure.

------------------------------------------------------------------------

## 25. Evaluation Framework for Checkpoint 4.1

For each query, compare the strategies on more than whether the final
answer sounds correct.

Useful retrieval measures include:

  -----------------------------------------------------------------------
  Metric                              Question
  ----------------------------------- -----------------------------------
  Target article hit                  Did the required article appear?

  Target coverage                     Were all named entities
                                      represented?

  Top-k precision                     How many returned articles were
                                      actually useful?

  Source diversity                    Did one unrelated cluster dominate?

  Multi-hop coverage                  Did decomposition recover missing
                                      entities?

  Graph gain                          Did graph expansion add useful
                                      evidence absent from seeds?

  Graph loss                          Did expansion displace directly
                                      relevant evidence?

  Passage hit                         Did the compact window contain the
                                      needed statement?

  Grounding                           Did the answer stay within supplied
                                      evidence?

  Exact quote success                 Was the requested wording actually
                                      available?
  -----------------------------------------------------------------------

This makes the comparison more meaningful than declaring one retrieval
method the universal winner.

------------------------------------------------------------------------

## 26. Troubleshooting

### `OPENROUTER_API_KEY not set`

Create a `.env` file:

``` text
OPENROUTER_API_KEY=your-key
```

### `Topic metadata not found`

The program can still run, but graph retrieval will use text links only.

To test topic expansion, provide:

``` text
wikipedia_topics_10.json
```

or set:

``` text
WIKIPEDIA_TOPICS_METADATA=<path>
```

### Vector DB appears stale

The Chroma database is persistent:

``` text
Wikipedia_chroma_full/
```

If the corpus or embedding configuration changes, rebuild the index
rather than assuming an existing index matches the current corpus.

### Correct article retrieved but answer still fails

Inspect the compact evidence window. The article may be correct while
the selected 1,200-character windows do not contain the required
sentence.

### Multi-step decomposition looks good but results are poor

The current multi-step implementation retrieves decomposed queries using
the simple lexical baseline. Improving the decomposer alone does not
strengthen that underlying retriever.

### Graph results are worse than hybrid

Inspect:

``` text
LABELS
```

and the hybrid seeds. If the seeds are wrong, the graph may simply
expand around irrelevant articles.

------------------------------------------------------------------------

## 27. Relationship to the Capstone Progression

``` text
Checkpoint 1.1
Representative questions / baseline scenario
          |
          v
Checkpoint 2.1
BM25 + Vector + Hybrid retrieval
          |
          v
Checkpoint 3.1
Evaluation infrastructure
PASS/FAIL diagnosis + robustness testing
          |
          v
Checkpoint 4.1
Advanced retrieval
Query decomposition + graph augmentation
          |
          v
Checkpoint 5.x
Agentic retrieval and tool use
          |
          v
Checkpoint 6.x
Security and reliability
          |
          v
Checkpoint 7.x
Production optimization
```

Checkpoint 4.1 therefore bridges conventional retrieval and later
agentic retrieval by allowing the system to perform **multiple retrieval
operations and relationship-based expansion** before answering.

------------------------------------------------------------------------

## 28. Key Takeaway

Checkpoint 4.1 successfully implements and exposes two advanced
retrieval ideas---**query decomposition** and **graph-augmented
retrieval**---but the experiment also demonstrates why adding complexity
does not automatically improve RAG quality.

The strongest lesson from the supplied run is:

``` text
Advanced retrieval quality
depends on
seed quality + graph quality + evidence coverage + passage selection
```

The graph retriever can preserve useful seeds and add related evidence,
but it can also amplify poor hybrid seeds or displace a directly
relevant article. Query decomposition can create excellent
sub-questions, but its benefit is constrained when those sub-queries are
sent to a weak lexical retriever.

The current results therefore provide a useful engineering baseline for
the next iteration: strengthen entity-aware retrieval, use hybrid search
inside the decomposition path, enrich the graph with metadata, protect
direct target hits, and evaluate article coverage separately from final
answer quality.
