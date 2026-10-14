# Capstone Checkpoint 2.1 --- Chunked Retrieval Strategy Design and Baseline Implementation

This checkpoint implements and evaluates the **chunked baseline
retrieval layer** for the Wikipedia **Leader Lens** Retrieval Engine.

The system compares three selectable retrieval strategies over the same
65-article Wikipedia corpus:

1.  **BM25 keyword retrieval**
2.  **Semantic vector retrieval with Chroma**
3.  **Hybrid retrieval combining BM25 and vector similarity**

Unlike the earlier article-level version, this implementation first
splits every Wikipedia article into **overlapping text chunks**.
Retrieval therefore ranks focused passages rather than entire long
articles.

The goal is to establish a strong, inspectable RAG baseline before later
capstone checkpoints introduce more advanced retrieval, agentic
workflows, security controls, evaluation, and production optimization.

------------------------------------------------------------------------

## Retrieval Architecture

``` text
Wikipedia Leader Lens HTML Corpus
            |
            v
     HTML Text Extraction
            |
            v
  Overlapping Chunk Creation
   1200 chars / 150 overlap
            |
            v
     8,566 Text Chunks
            |
      +-----+------+
      |            |
      v            v
    BM25       Chroma Vector
      |            |
      |            |
      +-----+------+
            |
      Selectable Mode
   BM25 / Vector / Hybrid
            |
            v
       Top-6 Chunks
            |
            v
   Grounded Answer Prompt
            |
            v
      GPT-5.4-mini
            |
            v
   Console + Retrieval Log
```

For **vector** and **hybrid** modes, the chunk embeddings are stored in
a persistent Chroma database.

------------------------------------------------------------------------

## Folder Layout

A typical working folder contains:

``` text
check_2/
├── Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py
├── Leader_lens/
│   ├── Abraham_Lincoln.html
│   ├── Adolf_Hitler.html
│   ├── Barack_Obama.html
│   ├── Benjamin_Franklin.html
│   ├── Boris_Johnson.html
│   ├── Cleopatra.html
│   ├── Emmanuel_Macron.html
│   ├── George_Washington.html
│   ├── Justin_Trudeau.html
│   ├── Liz_Truss.html
│   ├── Margaret_Thatcher.html
│   ├── Nelson_Mandela.html
│   ├── Presidency_of_Joe_Biden.html
│   ├── Queen_Victoria.html
│   ├── Rishi_Sunak.html
│   ├── Suharto.html
│   ├── Theresa_May.html
│   └── ... additional leader articles
├── wikipedia_chroma_full/
├── .env
├── checkpoint_2_1_retrieval_2.log
└── Output_2.txt
```

The supplied run loaded **65 Wikipedia HTML articles**.

------------------------------------------------------------------------

## Setup

Install the required Python packages:

``` bash
pip install python-dotenv beautifulsoup4 langchain-openai \
            langchain-core langchain-chroma rank-bm25
```

Create a `.env` file beside the script:

``` text
OPENROUTER_API_KEY=your_key_here
```

The implementation uses the OpenRouter OpenAI-compatible endpoint:

``` text
https://openrouter.ai/api/v1
```

------------------------------------------------------------------------

## Model Configuration

``` text
Answer model:     openai/gpt-5.4-mini
Judge model:      openai/gpt-5.4
Embedding model:  openai/text-embedding-3-small
Temperature:      0.2
```

The answer model is used by the retrieval pipeline to generate grounded
answers from the selected chunks.

`JUDGE_MODEL` is defined in the configuration, but the current `run()`
loop does not invoke a separate judge.

------------------------------------------------------------------------

## Main Configuration

  ------------------------------------------------------------------------------
  Setting                                            Value Purpose
  --------------------- ---------------------------------- ---------------------
  `SCENARIO`                                   `wikipedia` Selected capstone
                                                           scenario

  `HTML_DIR`                                 `Leader_lens` Leader Lens HTML
                                                           corpus

  `CHROMA_DIR`                     `wikipedia_chroma_full` Persistent chunk
                                                           vector index

  `NUM_RETRIEVED`                                      `6` Chunks supplied to
                                                           the answer model

  `TOP_K`                                              `6` Retrieval results
                                                           displayed per query

  `CANDIDATE_POOL`                                    `20` Candidate chunks from
                                                           each hybrid component

  `CHUNK_SIZE`                                      `1200` Maximum chunk size in
                                                           characters

  `CHUNK_OVERLAP`                                    `150` Character overlap
                                                           between neighboring
                                                           chunks

  `WEIGHT_BM25`                                      `0.4` BM25 contribution to
                                                           hybrid score

  `WEIGHT_VECTOR`                                    `0.6` Vector contribution
                                                           to hybrid score

  `TEMPERATURE`                                      `0.2` Answer-generation
                                                           temperature

  `LOG_PATH`              `checkpoint_2_1_retrieval_2.log` Experiment log
  ------------------------------------------------------------------------------

------------------------------------------------------------------------

# Chunking Strategy

The major change in this version is `chunk_documents()`.

Each complete Wikipedia article is divided into overlapping character
windows:

``` text
Chunk size    = 1200 characters
Chunk overlap = 150 characters
Step size     = 1050 characters
```

The step is calculated as:

``` python
step = chunk_size - overlap
```

Each chunk receives:

``` text
id
source article
chunk_id
text
```

For example:

``` text
Nelson_Mandela.html::chunk_4
Queen_Victoria.html::chunk_86
Presidency_of_Joe_Biden.html::chunk_3
```

This preserves the connection between a retrieved passage and its source
article.

------------------------------------------------------------------------

## Why Use Overlapping Chunks?

Whole Wikipedia articles can be very long. Sending or ranking an entire
article can dilute the relevance of a small passage containing the
actual answer.

Chunking changes the retrieval unit from:

``` text
entire Wikipedia article
```

to:

``` text
focused local passage
```

The overlap reduces the chance that useful evidence is lost when a
sentence or fact crosses a chunk boundary.

Conceptually:

``` text
Article
------------------------------------------------------------

Chunk 0
----------------------

                 Chunk 1
                 ----------------------

                                  Chunk 2
                                  ----------------------
```

The neighboring chunks share 150 characters.

------------------------------------------------------------------------

## Observed Corpus and Index Size

The supplied hybrid run reported:

``` text
Wikipedia articles loaded: 65
Chunks created:             8,566
Chunk size:                 1,200
Chunk overlap:              150
```

The first vector-index build then embedded:

``` text
8,566 Wikipedia chunks from 65 articles
```

into:

``` text
wikipedia_chroma_full/
```

This confirms that the current checkpoint performs **chunk-level
retrieval**, not whole-document retrieval.

------------------------------------------------------------------------

# Corpus Loading

`load_wikipedia_corpus()`:

1.  finds every `.html` file in `Leader_lens/`;
2.  opens the HTML;
3.  removes `script`, `style`, and `noscript` elements;
4.  extracts visible text;
5.  skips empty documents; and
6.  stores the filename and article text.

Each loaded article has the structure:

``` python
{
    "id": filename,
    "text": extracted_text
}
```

The articles are then passed into `chunk_documents()`.

------------------------------------------------------------------------

# Supported Retrieval Modes

The command-line interface accepts:

``` text
bm25
vector
hybrid
```

Run the script with one retrieval strategy at a time.

### BM25

``` bash
python .\Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py bm25
```

### Vector

``` bash
python .\Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py vector
```

### Hybrid

``` bash
python .\Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py hybrid
```

The supplied `Output_2.txt` is a **hybrid retrieval run**.

------------------------------------------------------------------------

# Retrieval Strategy 1 --- BM25

`BM25Retriever` builds an in-memory BM25 index over all Wikipedia
**chunks**.

Before indexing, the text is:

``` text
lowercased
    ↓
tokenized
    ↓
common stopwords removed
    ↓
BM25 index
```

For each query:

``` text
query
  ↓
tokenize
  ↓
BM25 score against 8,566 chunks
  ↓
sort descending
  ↓
top-k chunks
```

BM25 scores are:

``` text
higher = better
```

### Strength

BM25 is particularly useful when the query and evidence share
distinctive words, names, dates, titles, or exact terminology.

### Limitation

Lexical matching can miss relevant evidence when the question expresses
the same meaning using different vocabulary.

------------------------------------------------------------------------

# Retrieval Strategy 2 --- Vector Retrieval

`VectorRetriever` performs semantic retrieval with:

``` text
openai/text-embedding-3-small
+
Chroma
```

Every chunk is stored with metadata:

``` text
source
chunk_id
chunk_key
```

The retrieval process is:

``` text
query
  ↓
embedding
  ↓
Chroma similarity search
  ↓
nearest chunk vectors
  ↓
top-k chunks
```

Chroma returns a distance:

``` text
lower distance = better semantic match
```

### Strength

Vector retrieval can recover relevant passages even when the query does
not use the exact vocabulary appearing in the article.

### Limitation

Semantic similarity can sometimes prefer a related passage or another
leader article instead of the exact source required by the question.

------------------------------------------------------------------------

# Retrieval Strategy 3 --- Hybrid Retrieval

`HybridRetriever` combines keyword and semantic retrieval over the same
chunk collection.

Current weights:

``` text
BM25   = 0.40
Vector = 0.60
```

The pipeline is:

``` text
                     Query
                       |
             +---------+---------+
             |                   |
             v                   v
          BM25                Vector
             |                   |
       top 20 chunks        top 20 chunks
             |                   |
             v                   v
       normalize          invert + normalize
             |                   |
             +---------+---------+
                       |
                       v
          Weighted score fusion

          0.4 × normalized BM25
                    +
          0.6 × normalized vector
                       |
                       v
                 rank chunks
                       |
                       v
                   top six
```

The final hybrid score is:

``` text
Hybrid =
    0.4 × BM25_normalized
  + 0.6 × Vector_normalized
```

After normalization:

``` text
higher hybrid score = better
```

------------------------------------------------------------------------

# Score Normalization

BM25 and vector distance cannot be combined directly because their
scales and directions differ.

BM25:

``` text
higher = better
```

Vector distance:

``` text
lower = better
```

The `_normalize()` helper min-max scales values to:

``` text
0.0 ... 1.0
```

Vector distances are inverted after normalization so both retrieval
signals become:

``` text
higher = better
```

before weighted fusion.

------------------------------------------------------------------------

# Grounded Answer Generation

All retrieval strategies inherit from `BaseRetriever`.

After retrieval, the top chunks are formatted as:

``` text
[source::chunk_id]
chunk text

---

[source::chunk_id]
chunk text
```

The answer model receives only this retrieved evidence plus the
question.

Its system instruction requires it to:

-   use only the provided documents;
-   quote evidence where possible; and
-   say that the evidence is insufficient rather than guess.

This design makes retrieval failures visible.

If a required article never reaches the top-six context, the answer
model should not silently fill the gap using outside knowledge.

------------------------------------------------------------------------

# Representative Evaluation Set

The checkpoint runs **15 representative Leader Lens questions**.

The queries are intentionally aligned with the earlier baseline so
retrieval can be compared across:

``` text
No-retrieval baseline
        ↓
BM25
        ↓
Vector
        ↓
Hybrid
        ↓
Later advanced retrieval systems
```

------------------------------------------------------------------------

## Queries 1--5 --- Single-document Factual Retrieval

These test focused retrieval for:

``` text
Nelson Mandela
Margaret Thatcher
Queen Victoria
Abraham Lincoln
Emmanuel Macron
```

The questions ask for leadership office, reign, presidency, or service
period and request supporting evidence.

------------------------------------------------------------------------

## Queries 6--8 --- Multi-fact Retrieval

These require multiple facts from one source article:

``` text
Justin Trudeau
Suharto
Adolf Hitler
```

The system must retrieve passages containing several related facts
rather than only identify the correct person.

------------------------------------------------------------------------

## Queries 9--11 --- Cross-document / Multi-entity Retrieval

These are more difficult because several source articles may be
required:

``` text
Nelson Mandela vs. Suharto

Margaret Thatcher vs. Theresa May

George Washington + Abraham Lincoln + Barack Obama
```

These queries expose an important limitation of top-k chunk retrieval:
several highly ranked chunks from one article can occupy the available
context and crowd out another required entity.

------------------------------------------------------------------------

## Queries 12--13 --- Exact-evidence Retrieval

These ask for the opening sentence of:

``` text
George Washington
Cleopatra
```

They test whether retrieval selects the correct local passage, not
merely the correct article.

------------------------------------------------------------------------

## Query 14 --- Temporal Multi-document Retrieval

This query requires evidence for:

``` text
Boris Johnson
Liz Truss
Rishi Sunak
```

and asks the system to place their prime-ministerial terms in
chronological order.

------------------------------------------------------------------------

## Query 15 --- Article Disambiguation

The final question explicitly targets:

``` text
Presidency of Joe Biden
```

and warns:

``` text
Do not answer from a general Joe Biden biography.
```

This tests whether retrieval can select a specifically titled
political-period article rather than a related biography.

------------------------------------------------------------------------

# What the Hybrid Run Produced

The supplied experiment successfully:

``` text
loaded 65 articles
created 8,566 chunks
built the Chroma chunk index
initialized hybrid retrieval
ran all 15 questions
generated grounded answers
wrote results to the retrieval log
```

The run provides useful evidence for both the strengths and weaknesses
of the chunked baseline.

------------------------------------------------------------------------

# Observed Retrieval Strengths

## Strong single-article retrieval

For many focused questions, the correct source dominated the top
results.

### Nelson Mandela

The top six results were all Nelson Mandela chunks, with:

``` text
rank 1: Nelson_Mandela.html [chunk 4]
hybrid score = 1.0000
```

The generated answer correctly identified him as President of South
Africa from 1994 to 1999.

### Queen Victoria

All six retrieved chunks came from `Queen_Victoria.html`, and the answer
recovered the exact reign dates:

``` text
20 June 1837 – 22 January 1901
```

### Abraham Lincoln

The highest-ranked chunk came from `Abraham_Lincoln.html`, and the
answer correctly identified:

``` text
16th president
1861–1865
```

### Emmanuel Macron

The highest results came from the Macron article and supported:

``` text
President of France
```

### Suharto

The top six results all came from `Suharto.html`, allowing the answer to
recover his presidency and leadership period.

### Presidency of Joe Biden

All six retrieved chunks came from the specifically requested:

``` text
Presidency_of_Joe_Biden.html
```

The highest-ranked result was:

``` text
chunk 3
hybrid score = 0.9015
```

and the answer correctly identified the inauguration on January 20, 2021
as the beginning of the presidential tenure.

------------------------------------------------------------------------

# Evidence of Multi-document Success

The Thatcher--May comparison successfully retrieved chunks from both
required articles:

``` text
Theresa_May.html
Margaret_Thatcher.html
```

The answer correctly identified the shared office as:

``` text
Prime Minister of the United Kingdom
```

with:

``` text
Margaret Thatcher: 1979–1990
Theresa May:       2016–2019
```

The Boris Johnson / Liz Truss / Rishi Sunak query also retrieved chunks
from all three relevant articles and produced the correct chronological
order:

``` text
Boris Johnson
      ↓
Liz Truss
      ↓
Rishi Sunak
```

However, the answer noted that the retrieved Boris Johnson chunk did not
itself contain the explicit term evidence needed for a direct quotation.
This demonstrates the difference between retrieving the correct
**article** and retrieving the correct **passage within that article**.

------------------------------------------------------------------------

# Important Retrieval Failures

The experiment also exposed several valuable baseline limitations.

## Query 9 --- Mandela vs. Suharto

The question required evidence from both:

``` text
Nelson Mandela
Suharto
```

but all six retrieved results were Nelson Mandela chunks.

Therefore:

``` text
Suharto evidence coverage = 0
```

The grounded answer correctly stated that it could not make the Suharto
comparison without guessing.

This is a clear example of **same-document chunk saturation**.

A single relevant article can contribute many high-scoring chunks and
consume the entire top-k result set.

------------------------------------------------------------------------

## Query 11 --- Three U.S. Presidents

The question explicitly requested:

``` text
George Washington
Abraham Lincoln
Barack Obama
```

but the retrieved top six were dominated by:

``` text
William McKinley
Donald Trump
Abraham Lincoln
James Buchanan
Jimmy Carter
```

The model was still able to find Washington and Obama names inside
general presidential timeline content embedded in other retrieved
articles, but the intended source articles were not directly retrieved.

This means the answer may be factually useful while the retrieval
behavior is still weaker than the evaluation objective.

For a source-grounded RAG evaluation, **answer correctness alone should
not be treated as retrieval success**.

------------------------------------------------------------------------

## Query 13 --- Cleopatra Opening Sentence

The system retrieved:

``` text
Cleopatra.html [chunk 0]
```

as the top result, but the answer stated:

``` text
The provided documents do not include the article’s opening sentence.
```

This is a useful diagnostic case.

The correct article was retrieved, but the retrieved chunk/context did
not provide the exact evidence in a form the answer model could
confidently quote.

This demonstrates:

``` text
correct source retrieval
        ≠
correct evidence retrieval
```

------------------------------------------------------------------------

# Key Baseline Finding

The chunked system substantially improves retrieval granularity, but
introduces a new ranking problem:

``` text
many chunks from one highly relevant article
                ↓
occupy several top-k positions
                ↓
reduce source diversity
                ↓
multi-document evidence can be missed
```

This matters especially for:

-   comparisons;
-   multi-entity questions;
-   multi-hop questions;
-   temporal synthesis across several people; and
-   questions requiring one piece of evidence from each named article.

------------------------------------------------------------------------

# Why Chunking Helps

Compared with whole-article retrieval, chunking offers several
advantages:

``` text
more focused evidence
smaller context units
better local semantic matching
less irrelevant article text
more precise quotations
```

For single-entity factual questions, the supplied run shows strong
retrieval behavior.

------------------------------------------------------------------------

# Why Chunking Alone Is Not Enough

Chunking also creates many candidates per article.

For example, one leader can have hundreds of chunks. If several chunks
from that leader score highly, the top-six result set can lose article
diversity.

A future refinement could therefore consider:

``` text
per-source diversity constraints
deduplication by article
maximum chunks per article
reranking
query decomposition
entity-aware retrieval
multi-step retrieval
```

Those are later-system improvements; they are not silently added to this
Checkpoint 2.1 baseline.

------------------------------------------------------------------------

# Reading the Output

For each query, the console shows:

``` text
QUERY
Retrieved results
rank
source article
chunk number
retrieval score
Answer
```

Example:

``` text
1. Nelson_Mandela.html [chunk 4] | hybrid score = 1.0000
```

The display converts the internal identifier:

``` text
Nelson_Mandela.html::chunk_4
```

into the more readable console form:

``` text
Nelson_Mandela.html [chunk 4]
```

------------------------------------------------------------------------

# Logging

Results are written to:

``` text
checkpoint_2_1_retrieval_2.log
```

For each query, the log stores:

``` text
retrieval mode
query number
question
retrieved chunk identifiers
retrieval scores
generated answer
```

Example:

``` text
HYBRID | QUERY 1
retrieved=[
    ('Nelson_Mandela.html::chunk_4', 1.0),
    ...
]
answer=...
```

This makes the experiment reproducible and supports later evaluation.

------------------------------------------------------------------------

# How to Evaluate the Retrieval Results

Do not evaluate only whether the final answer sounds correct.

For every question, inspect:

  -----------------------------------------------------------------------
  Criterion                           Question
  ----------------------------------- -----------------------------------
  Source hit                          Was the intended Wikipedia article
                                      retrieved?

  Passage hit                         Was the chunk containing the
                                      requested fact retrieved?

  Rank                                How high did the required evidence
                                      appear?

  Coverage                            Were all required entities
                                      represented?

  Diversity                           Did one article dominate the top-k?

  Grounding                           Could the answer be supported
                                      entirely by retrieved chunks?

  Exact evidence                      Was requested quotation-level
                                      evidence actually available?
  -----------------------------------------------------------------------

For multi-document questions, **coverage across required sources** is
especially important.

------------------------------------------------------------------------

# Suggested Evaluation Metrics

Useful metrics for this checkpoint include:

``` text
Hit@1
Hit@3
Hit@6
source coverage
required-entity coverage
passage/evidence coverage
top-k source diversity
answer completeness
grounded quotation success
```

For a three-article question, a simple source-coverage measure could be:

``` text
required articles retrieved
---------------------------
total required articles
```

For example:

``` text
3 / 3 = complete source coverage
1 / 3 = incomplete source coverage
```

------------------------------------------------------------------------

# Comparison of the Three Retrieval Strategies

  Feature                 BM25              Vector                Hybrid
  ----------------------- ----------------- --------------------- -----------------------
  Retrieval unit          Chunk             Chunk                 Chunk
  Main signal             Keyword overlap   Semantic similarity   Both
  Exact names/terms       Strong            Usually strong        Strong
  Paraphrased concepts    Weaker            Strong                Strong
  Score direction         Higher better     Lower better          Higher better
  Persistent vector DB    No                Yes                   Yes
  Current hybrid weight   ---               ---                   40% BM25 / 60% vector

The checkpoint is designed to compare these methods experimentally
rather than assume one is always superior.

------------------------------------------------------------------------

# Recommended Experimental Procedure

Run all three modes using the same 15 queries:

``` bash
python .\Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py bm25

python .\Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py vector

python .\Capstone_Checkpoint_2_1_selectable_retriever_CHUNKED.py hybrid
```

Then compare query-level evidence.

A useful evaluation table is:

  ------------------------------------------------------------------------
  Query       Required     BM25        Vector      Hybrid      Best
              source(s)    coverage    coverage    coverage    
  ----------- ------------ ----------- ----------- ----------- -----------
  1           Nelson                                           
              Mandela                                          

  2           Margaret                                         
              Thatcher                                         

  3           Queen                                            
              Victoria                                         

  ...         ...          ...         ...         ...         ...

  15          Presidency                                       
              of Joe Biden                                     
  ------------------------------------------------------------------------

------------------------------------------------------------------------

# Persistent Chroma Index

`build_or_load_db()` checks whether:

``` text
wikipedia_chroma_full/
```

already contains a vector index.

If it exists, the script loads it.

Otherwise:

``` text
8,566 chunks
      ↓
text-embedding-3-small
      ↓
Chroma
      ↓
persistent index
```

This avoids re-embedding all chunks on every run.

------------------------------------------------------------------------

## Important Index-Rebuild Note

The code reuses the existing Chroma directory whenever it is non-empty.

Therefore, if any of the following change:

``` text
Leader_lens corpus
chunk size
chunk overlap
embedding model
chunk-generation logic
```

the old vector index may no longer correspond to the current experiment.

Rebuild the Chroma directory when intentionally changing the indexed
corpus or chunking configuration.

------------------------------------------------------------------------

# Troubleshooting

## `OPENROUTER_API_KEY is not set`

Create:

``` text
.env
```

with:

``` text
OPENROUTER_API_KEY=your_key_here
```

------------------------------------------------------------------------

## `Found 0 Wikipedia HTML files`

Verify:

``` python
HTML_DIR = Path("Leader_lens")
```

and make sure the script is launched from a directory where
`Leader_lens/` is reachable.

------------------------------------------------------------------------

## `Chunks created: 0`

Check that:

-   Wikipedia articles loaded successfully;
-   extracted article text is non-empty;
-   `CHUNK_SIZE > 0`; and
-   `0 <= CHUNK_OVERLAP < CHUNK_SIZE`.

The function explicitly validates invalid chunk sizes and overlaps.

------------------------------------------------------------------------

## Vector or Hybrid Results Look Stale

Delete/rebuild the existing Chroma index if the corpus or chunking
settings changed.

The persistent index is not automatically versioned against the current
corpus configuration.

------------------------------------------------------------------------

## One Article Dominates the Top Six

This is visible in the Mandela--Suharto run.

The current retriever ranks **chunks independently**, so several chunks
from one source can occupy every top-k slot.

Treat this as a baseline limitation rather than an answer-model problem.

------------------------------------------------------------------------

## Correct Article Retrieved but Exact Quote Missing

This occurred in the Cleopatra probe.

Inspect:

``` text
retrieved chunk number
chunk text
query wording
```

The source may be correct while the exact requested passage is absent or
not clearly exposed in the retrieved context.

------------------------------------------------------------------------

## Multi-document Question Is Incomplete

First inspect whether all required articles appear among the top-six
chunks.

If a required source is absent:

``` text
retrieval problem
```

If all required evidence is present but the answer remains incomplete:

``` text
generation / evidence-use problem
```

------------------------------------------------------------------------

# Relationship to the Capstone Progression

``` text
Checkpoint 1.1
Baseline / no retrieval
        |
        v
Checkpoint 2.1
Chunked BM25 / Vector / Hybrid
        |
        v
Retrieval Evaluation and Refinement
        |
        v
Advanced Retrieval
        |
        v
Agentic Retrieval
        |
        v
Security and Reliability
        |
        v
Production Evaluation / Optimization
```

Checkpoint 2.1 provides the retrieval baseline against which later
system complexity can be justified.

------------------------------------------------------------------------

# Key Takeaway

This checkpoint demonstrates two important RAG principles.

First:

``` text
whole article
     ↓
chunking
     ↓
more focused retrieval
     ↓
better local evidence
```

Second:

``` text
more precise chunks
       does not automatically mean
complete multi-document coverage
```

The supplied hybrid run performed strongly on many **single-article
factual questions** and successfully retrieved precise evidence for
leaders such as Nelson Mandela, Queen Victoria, Abraham Lincoln,
Emmanuel Macron, Suharto, and the Presidency of Joe Biden.

At the same time, the multi-entity tests revealed an important baseline
weakness: independent chunk ranking can allow multiple passages from one
source to dominate the top-k context, causing required evidence from
another article to disappear.

The central evaluation question for this checkpoint is therefore:

``` text
Does the retrieval strategy place the correct evidence chunks
from every required source into the model's limited context?
```

That is a stronger criterion than simply asking whether the final answer
sounds plausible, and it provides a meaningful baseline for the advanced
retrieval methods introduced in later checkpoints.
