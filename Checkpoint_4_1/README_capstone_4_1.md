# Capstone Checkpoint 4.1 --- Advanced Retrieval for the Wikipedia Retrieval Engine

This checkpoint extends the Wikipedia Retrieval Engine beyond a single
retrieval pass by combining the two advanced retrieval strategies
introduced in Module 4:

1.  **Multi-step retrieval by query decomposition** --- adapted from Lab
    4.1.
2.  **Graph-augmented retrieval** --- adapted from Lab 4.2.

The goal is not simply to add more retrieval stages. The goal is to test
whether decomposing complex questions and expanding retrieved evidence
through document relationships improves evidence coverage for
multi-part, comparative, and multi-hop Wikipedia questions.

## Advanced Retrieval Architecture

The figure below shows the combined advanced retrieval pipeline: query
decomposition creates multiple sub-queries, hybrid search retrieves
evidence for each, scores are merged into multi-step evidence, and graph
expansion adds category-, topic-, and article-related context before the
final grounded answer is generated.

![Capstone Checkpoint 4.1 --- Advanced Retrieval
Architecture](capstone_4_1.png)

*Figure: Combined multi-step query decomposition and graph-augmented
retrieval architecture for the Wikipedia Retrieval Engine.*

------------------------------------------------------------------------

## Folder layout

A typical working folder for this checkpoint contains:

``` text
Capstone_Checkpoint_4_1/
├── capstone_checkpoint_4_1_advanced_retrieval_starter.py
├── Wikipedia_10_text/
│   ├── 85th_Academy_Awards.txt
│   ├── Adolf_Hitler.txt
│   ├── Bird.txt
│   ├── Emirates_(airline).txt
│   ├── Empire_State_Building.txt
│   ├── Hallstatt.txt
│   ├── Margaret_Thatcher.txt
│   ├── Mytilidae.txt
│   ├── Queen_Victoria.txt
│   └── Steve_Jobs.txt
├── wikipedia_topics_10.json
├── Wikipedia_chroma_topics_full/
└── checkpoint_4_1_advanced_retrieval_final.log
```

The Python file expects the Wikipedia text corpus, topic/category
metadata, and a Chroma vector database. Paths can be overridden with
environment variables.

------------------------------------------------------------------------

## Setup

Create or activate your Python environment and install the required
packages:

``` bash
pip install python-dotenv langchain-openai langchain-core langchain-chroma \
            langchain-huggingface sentence-transformers rank-bm25 networkx
```

The checkpoint uses **LM Studio** through its OpenAI-compatible local
API.

Start LM Studio, load the required models, and make sure the local
server is available at:

``` text
http://127.0.0.1:1234/v1
```

The current configuration uses:

``` text
Decomposition model: deepseek-coder-6.7b-instruct
Answer model:        meta-llama-3.1-8b-instruct
Embedding model:     sentence-transformers/all-MiniLM-L6-v2
```

Then run:

``` bash
python capstone_checkpoint_4_1_advanced_retrieval_starter.py
```

------------------------------------------------------------------------

## Main configuration

The checkpoint exposes the important retrieval settings near the top of
the script.

  ----------------------------------------------------------------------------------
  Setting                       Default                          Purpose
  ----------------------------- -------------------------------- -------------------
  `WIKIPEDIA_TEXT_DIR`          `Wikipedia_10_text`              Wikipedia text
                                                                 corpus

  `WIKIPEDIA_CHROMA_DIR`        `Wikipedia_chroma_topics_full`   Chroma vector
                                                                 database

  `WIKIPEDIA_TOPICS_METADATA`   `wikipedia_topics_10.json`       Category/topic
                                                                 metadata

  `NUM_RETRIEVED`               `6`                              Number of retrieved
                                                                 items

  `CANDIDATE_POOL`              `20`                             Candidate pool used
                                                                 before narrowing

  `WEIGHT_BM25`                 `0.5`                            BM25 contribution
                                                                 to hybrid retrieval

  `WEIGHT_VECTOR`               `0.5`                            Vector contribution
                                                                 to hybrid retrieval

  `RETRIEVER_MODE`              `hybrid_tagged`                  Active baseline
                                                                 retriever

  `DECOMPOSITION_MODEL`         `deepseek-coder-6.7b-instruct`   Query decomposition
                                                                 LLM

  `ANSWER_MODEL`                `meta-llama-3.1-8b-instruct`     Final answering LLM
  ----------------------------------------------------------------------------------

Supported retrieval modes are:

``` text
bm25
vector
hybrid
hybrid_tagged
```

The advanced retrieval stages therefore sit **on top of the existing
capstone retrieval system** rather than replacing it.

------------------------------------------------------------------------

## What a run produces

A complete run:

1.  loads the Wikipedia text corpus;
2.  loads and checks category/topic metadata;
3.  connects to LM Studio;
4.  loads or builds the Chroma vector database;
5.  initializes the selected baseline retriever;
6.  builds the Wikipedia document graph;
7.  runs the checkpoint test questions;
8.  prints baseline, multi-step, graph, and combined retrieval results;
9.  generates a final answer from the combined evidence; and
10. writes answer records to the checkpoint log.

Typical output has this structure:

``` text
QUESTION
...

DECOMPOSITION:
  1. ...
  2. ...

BASELINE:
[...]

MULTI-STEP:
[...]

GRAPH:
{...}

advanced evidence:
  document: score=..., sources=[...]

COMBINED:
[...]

ADVANCED ANSWER:
...
```

This makes the retrieval process inspectable instead of showing only the
final answer.

------------------------------------------------------------------------

## Core concepts in this checkpoint

-   **Query decomposition** --- complex questions are divided into
    smaller information needs before retrieval. A comparison can become
    one query per entity; a multi-part factual question can become one
    query per requested fact.

-   **LLM inside the retrieval loop** --- the decomposition model is
    used before retrieval. Its job is to generate retrieval queries, not
    answer the user's question.

-   **Score accumulation** --- documents retrieved by several
    sub-queries accumulate evidence scores. This allows an article that
    supports several parts of the question to rise in the ranking.

-   **Graph-augmented retrieval** --- retrieved documents become graph
    seeds. The system can then add evidence connected by Wikipedia
    metadata rather than relying only on lexical or semantic similarity.

-   **Seed-and-expand** --- baseline retrieval finds relevant entry
    points; graph traversal expands from those seeds.

-   **Typed graph structure** --- the graph distinguishes articles,
    categories, and topics so that relationships have explicit meaning.

-   **Combined retrieval** --- decomposition and graph expansion are
    used together. Direct retrieval remains the primary evidence source
    while graph-derived documents provide additional context.

-   **Graceful fallback** --- if tagged routing cannot identify enough
    category-matched candidates, the system falls back to full-corpus
    hybrid retrieval instead of failing.

------------------------------------------------------------------------

## Step 1 --- Query decomposition

`decompose_query()` converts a Wikipedia question into **1--4 focused
retrieval queries**.

For example:

``` text
According to the Wikipedia article on the Empire State Building,
when was the building completed and when did it officially open?
```

can be decomposed into separate searches for completion and opening
dates.

The decomposition prompt is designed to:

-   preserve important named entities;
-   separate comparison questions by entity;
-   separate multi-part questions by requested fact;
-   separate the stages of multi-hop questions;
-   avoid answering the question during decomposition; and
-   leave simple single-fact questions as one query when decomposition
    is unnecessary.

The implementation also includes a quality check. If a question appears
complex but the model returns only one sub-query, the model is prompted
again with stronger decomposition instructions.

### Why this matters

A single search query can retrieve an article that strongly matches one
part of a question while missing evidence for another part.
Decomposition increases the chance that every information need receives
its own retrieval pass.

------------------------------------------------------------------------

## Step 2 --- Multi-step retrieval

`multistep_retrieve()` is adapted from Lab 4.1.

``` python
for sq in sub_queries:
    for doc_id, score in baseline_retrieve(sq, 3 * k):
        score_map[doc_id] = score_map.get(doc_id, 0.0) + score
```

Each sub-query retrieves an expanded candidate set. Scores are
accumulated by document ID, and the highest-scoring documents are
retained.

The important design idea is:

``` text
one complex question
        ↓
several focused searches
        ↓
merge evidence across searches
        ↓
top documents covering multiple facets
```

This is especially useful for:

-   comparisons;
-   questions asking for several facts;
-   questions involving multiple named entities; and
-   multi-hop questions.

------------------------------------------------------------------------

## Step 3 --- Wikipedia graph construction

`build_graph()` creates a directed NetworkX graph from the Wikipedia
corpus metadata.

### Graph schema

  ---------------------------------------------------------------------------
  Node type           Example                             Purpose
  ------------------- ----------------------------------- -------------------
  `article`           `doc:Hallstatt.txt`                 Represents a
                                                          Wikipedia article

  `category`          `category:place_geography`          Represents a broad
                                                          retrieval category

  `topic`             `topic:academy awards ceremonies`   Represents a
                                                          Wikipedia
                                                          topic/category
                                                          label
  ---------------------------------------------------------------------------

### Edge types

  -----------------------------------------------------------------------
  Edge                    Direction               Meaning
  ----------------------- ----------------------- -----------------------
  `belongs_to_category`   article → category      Article belongs to a
                                                  broad category

  `category_contains`     category → article      Reverse category
                                                  membership

  `relates_to`            article → topic         Article is associated
                                                  with a Wikipedia topic

  `links_to`              article → article       Optional direct article
                                                  relationship when link
                                                  metadata exists
  -----------------------------------------------------------------------

The graph is rebuilt in memory when the script runs.

### Why graph retrieval is different

BM25 and vector retrieval answer:

``` text
Which documents are most similar to this query?
```

Graph retrieval can additionally ask:

``` text
Which documents are connected to the retrieved evidence?
```

That distinction is important for multi-hop questions where supporting
evidence may be structurally related without being the highest-scoring
semantic match.

------------------------------------------------------------------------

## Step 4 --- Graph retrieval

`graph_retrieve()` first obtains baseline seed documents and then
expands from them.

For a seed article, the retriever can add:

-   directly linked Wikipedia articles; and
-   other articles associated with the same topic.

The returned dictionary records how each document entered the evidence
set:

``` text
seed
linked
topic
```

This makes graph expansion easier to inspect during debugging.

------------------------------------------------------------------------

## Step 5 --- Combined advanced retrieval

`advanced_retrieve()` combines the ideas from Labs 4.1 and 4.2.

``` text
decomposed sub-queries
        ↓
retrieve seeds for every sub-query
        ↓
accumulate direct retrieval scores
        ↓
expand seeds through graph relationships
        ↓
assign smaller graph-expansion contributions
        ↓
merge + deduplicate
        ↓
rank final evidence
```

Directly retrieved documents receive the strongest evidence
contribution. Graph-only evidence receives smaller weighted
contributions, so graph expansion supplements rather than overwhelms
direct retrieval.

The diagnostic output shows both the accumulated score and the source
type:

``` text
advanced evidence:
    Hallstatt.txt: score=1.000, sources=['seed']
```

When graph metadata is populated, a document may also show `linked` or
`topic` as an evidence source.

------------------------------------------------------------------------

## Step 6 --- Grounded answer generation

`answer_from_docs()` builds the final answer context from the selected
Wikipedia documents.

Rather than sending entire long articles to the model, the function
selects relevant chunks from the chosen documents and limits the total
context size.

The answer model is instructed to use the retrieved Wikipedia evidence
rather than unsupported outside knowledge.

The run also reports context size, for example:

``` text
Answer context: 16,611 characters from 9 chunks
```

This is useful for monitoring the tradeoff between evidence coverage and
context-window noise.

------------------------------------------------------------------------

## Advanced-retrieval plan

`my_advanced_plan()` documents the intended capstone design.

The selected technique is:

``` text
both
```

meaning the system combines:

``` text
query decomposition + graph-based retrieval
```

The plan defines:

-   the graph node types;
-   the graph edge types;
-   representative test questions;
-   the rationale for advanced retrieval; and
-   how the new design relates to the existing Wikipedia retrieval
    engine.

This function is the bridge between the implementation and the written
checkpoint report.

------------------------------------------------------------------------

## Test set

The script runs **10 Wikipedia test queries** covering different
retrieval demands, including:

-   multi-part factual questions;
-   biographical questions;
-   comparisons across articles;
-   multi-hop historical questions;
-   entity identification;
-   location and topic questions; and
-   questions requiring evidence from more than one article.

Examples include:

``` text
According to the Wikipedia article on Hallstatt, where is Hallstatt
located and what is it particularly known for?
```

and:

``` text
Compare the roles of Queen Victoria and Margaret Thatcher in British
history using evidence from their respective Wikipedia articles.
```

The purpose is to compare retrieval behavior across different question
structures rather than test only easy single-fact lookups.

------------------------------------------------------------------------

## Reading the evaluation output

For every test question, compare four evidence sets:

  ---------------------------------------------------------------------
  Retrieval output                   What it shows
  ---------------------------------- ----------------------------------
  `BASELINE`                         What the active retriever finds
                                     from the original question

  `MULTI-STEP`                       What changes when the question is
                                     decomposed

  `GRAPH`                            What graph expansion adds around
                                     baseline seeds

  `COMBINED`                         Evidence selected when
                                     decomposition and graph retrieval
                                     work together
  ---------------------------------------------------------------------

Then inspect the `ADVANCED ANSWER`.

A useful evaluation question is not simply:

``` text
Did advanced retrieval return more documents?
```

Instead ask:

``` text
Did the added retrieval stages bring in evidence needed to answer
the question more completely and accurately?
```

More retrieved context is not automatically better context.

------------------------------------------------------------------------

## Observed run behavior

The supplied run successfully loaded the 10-document Wikipedia text
corpus and initialized the tagged-hybrid retriever. It also showed an
important diagnostic issue: the loaded documents were reported with
category `other`, Chroma metadata synchronization found no matching
metadata, and the resulting graph contained **10 article nodes but 0
edges**.

``` text
GRAPH CREATED
Nodes: 10
Edges: 0
Node types: Counter({'article': 10})
Edge types: Counter()
```

Because the graph had no category, topic, or link edges in that run,
graph retrieval could not provide meaningful relationship-based
expansion. The advanced pipeline still ran because the implementation
degrades to seed-based retrieval rather than crashing.

This is an important checkpoint lesson: **graph retrieval is only as
useful as the metadata used to construct the graph.**

The supplied `wikipedia_topics_10.json` does contain broad categories
and Wikipedia topics. Therefore, the metadata-loading/matching path
should be checked when a run reports every article as `other` or
produces a zero-edge graph.

------------------------------------------------------------------------

## Example result

For the Hallstatt question, the run decomposed the question into
separate information needs and retrieved `Hallstatt.txt` as the leading
evidence. The final answer correctly described Hallstatt's location in
Upper Austria and its association with prehistoric salt production and
the Hallstatt culture.

This illustrates the intended benefit of the architecture:

``` text
multi-part question
        ↓
separate retrieval needs
        ↓
strong primary article
        ↓
focused evidence chunks
        ↓
grounded multi-part answer
```

At the same time, several other questions produced insufficient or
incomplete answers. That is useful diagnostic evidence: adding
decomposition and graph stages does not automatically guarantee better
generation.

------------------------------------------------------------------------

## Comparison with Labs 4.1 and 4.2

  ------------------------------------------------------------------------------------
                  Lab 4.1         Lab 4.2                     Capstone Checkpoint 4.1
  --------------- --------------- --------------------------- ------------------------
  Corpus          Email database  Annotated email database    Wikipedia articles

  Main technique  Query           Graph expansion             Both
                  decomposition                               

  Expands         Query           Result set                  Query + result set

  Graph           No              Yes                         Yes

  Extra           Yes             No                          Yes
  decomposition                                               
  LLM call                                                    

  Baseline        Hybrid          Hybrid                      BM25 / vector / hybrid /
  retrieval                                                   tagged hybrid

  Graph structure ---             Email/thread/person/topic   Article/category/topic

  Best use        Multi-part      Connected/timeline evidence Multi-part, comparative,
                  questions                                   multi-hop Wikipedia
                                                              questions
  ------------------------------------------------------------------------------------

Lab 4.1 changes **how the question is searched**.

Lab 4.2 changes **how retrieved evidence is expanded**.

The capstone combines both ideas while retaining the earlier retrieval
engine as the baseline search layer.

------------------------------------------------------------------------

## Important implementation notes

-   **Decomposition quality matters.** A poor sub-query can introduce
    irrelevant evidence. Inspect the printed decomposition instead of
    assuming it is correct.

-   **Graph quality depends on metadata quality.** Missing categories,
    topics, or links can reduce graph retrieval to seed-only behavior.

-   **More context can increase noise.** Advanced retrieval should be
    evaluated by evidence quality and answer completeness, not by the
    number of documents retrieved.

-   **Direct evidence should dominate graph-only evidence.** The
    implementation gives graph expansion smaller score contributions for
    this reason.

-   **Tagged retrieval has a fallback.** When category routing produces
    too few candidates, the system falls back to full-corpus hybrid
    retrieval.

-   **The decomposition and answering models have different jobs.** The
    decomposition model rewrites the search; the answer model
    synthesizes the final response.

-   **Context is chunked and bounded.** This prevents long Wikipedia
    articles from consuming the entire prompt.

------------------------------------------------------------------------

## Troubleshooting

### LM Studio is not reachable

Make sure the LM Studio local server is running at:

``` text
http://127.0.0.1:1234/v1
```

and that the configured model names are loaded.

### Tagged routing always falls back

Check that `wikipedia_topics_10.json` is being loaded and that its
filenames match the filenames in the text corpus.

### Every document shows `category=other`

Inspect the metadata attachment step. The JSON metadata should provide
fields such as:

``` text
filename
broad_category
topics
```

If those values are not attached to the in-memory documents,
category-aware retrieval and graph construction will lose their
metadata.

### Graph shows 0 edges

Check the `CATEGORY METADATA CHECK`, `CHROMA METADATA CHECK`, and
`DOCUMENT GRAPH METADATA` sections of the console output.

A graph containing only article nodes means that no usable category,
topic, or direct-link relationships were attached before `build_graph()`
ran.

### Decomposition contains unwanted instructions

Because the decomposition output is model-generated, inspect each
sub-query. The parser removes bullets and numbering, but semantic
cleanup may still be needed if the model emits instruction-like text
instead of a retrieval query.

### Final answer says evidence is insufficient

Check the evidence before changing the answer prompt:

``` text
BASELINE
MULTI-STEP
GRAPH
COMBINED
```

If the required article or fact never entered the combined evidence, the
problem is retrieval. If the correct evidence is present but the answer
is still incomplete, the issue is more likely chunk selection or answer
generation.

------------------------------------------------------------------------

## Key takeaway

This checkpoint demonstrates that advanced RAG is not simply:

``` text
retrieve more → answer better
```

A better model is:

``` text
identify the information needs
        ↓
retrieve evidence for each need
        ↓
use structure to discover related evidence
        ↓
control noise and deduplicate context
        ↓
generate only from grounded evidence
        ↓
evaluate whether the redesign actually helped
```

The most important lesson is that **additional retrieval stages must
earn their complexity**. Query decomposition can improve coverage, and
graph traversal can expose connected evidence, but both can also amplify
noise when decomposition or metadata quality is weak. The architecture
should therefore be judged by retrieval quality and grounded answer
quality, not by complexity alone.
