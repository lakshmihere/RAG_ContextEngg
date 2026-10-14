# Capstone Checkpoint 3.1 --- Evaluation Infrastructure and Baseline Diagnosis

## Wikipedia Leader Lens

This checkpoint extends the **Wikipedia Leader Lens** retrieval system
from retrieval testing into a repeatable **evaluation framework**. The
system evaluates grounded answers produced from a 65-article Wikipedia
corpus, assigns **PASS/FAIL** verdicts with an LLM judge, records
retrieval evidence, and runs a local **RAGAS experiment**. It also
supports an expanded test set containing paraphrased question variants
so retrieval robustness can be measured rather than inferred from a
single wording.

The checkpoint therefore answers a different question from Checkpoint
2.1:

> **Checkpoint 2.1:** Can the system retrieve useful evidence?\
> **Checkpoint 3.1:** How reliably does the complete
> retrieval-and-answer pipeline succeed across a defined evaluation set,
> and where does it fail?

------------------------------------------------------------------------

## 1. System Architecture

``` text
                    Wikipedia_text/
                     65 TXT articles
                           |
                           v
                 Load Wikipedia corpus
                           |
                           v
              Overlapping character chunking
               1,200 chars / 150 overlap
                           |
                           v
                    19,811 chunks
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
           BM25          Chroma        Hybrid
        retrieval        vector       retrieval
                         search       0.4 / 0.6
             \             |             /
              +------------+------------+
                           |
                           v
                   Top 6 retrieved chunks
                           |
                           v
                  Grounded answer prompt
                           |
                           v
                 GPT-5.4-mini answer model
                           |
                           v
          +----------------+----------------+
          |                                 |
          v                                 v
   Standard evaluation                RAGAS experiment
   15 original questions        original + paraphrase variants
          |                                 |
          v                                 v
      GPT-5.4 judge                    GPT-5.4 judge
       PASS / FAIL                      PASS / FAIL
          |                                 |
          v                                 v
      .log output               retrieved sources/context,
                                response, score, CSV results
```

------------------------------------------------------------------------

## 2. Project Files

A typical Checkpoint 3.1 working folder contains:

``` text
check_3/
│
├── capstone_checkpoint_3_1_evaluation_1.py
├── variants_3_1_FIXED.py
│
├── Wikipedia_text/
│   ├── Abraham_Lincoln.txt
│   ├── Adolf_Hitler.txt
│   ├── Barack_Obama.txt
│   ├── Boris_Johnson.txt
│   ├── Cleopatra.txt
│   ├── Nelson_Mandela.txt
│   └── ... 65 articles total
│
├── wikipedia_chroma_full/
│
├── checkpoint_3_1_hybrid.csv
├── checkpoint_3_1_hybrid.json
├── checkpoint_3_1_hybrid_variants.json
├── wikipedia_leader_lens_variants_eval.csv
│
├── checkpoint_3_1_evaluation_1.log
├── checkpoint_3_1_evaluation_2.log
│
└── ragas_experiments/
    └── experiments/
        └── <experiment-name>.csv
```

The exact RAGAS-generated experiment filename can vary between runs.

------------------------------------------------------------------------

## 3. Environment Setup

Create and activate a Python environment, then install the required
packages.

``` powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install beautifulsoup4 python-dotenv langchain-chroma langchain-core langchain-openai langchain-huggingface openai rank-bm25 ragas pandas
```

Create a `.env` file in the project directory:

``` text
OPENROUTER_API_KEY=your-openrouter-key
```

Do not commit the API key to source control.

------------------------------------------------------------------------

## 4. Core Configuration

  Setting            Value
  ------------------ ---------------------------------------
  Scenario           `wikipedia`
  Corpus             `Wikipedia_text`
  Articles           65
  Observed chunks    19,811
  Chunk size         1,200 characters
  Chunk overlap      150 characters
  Chunk step         1,050 characters
  Retrieved chunks   6
  Candidate pool     20
  BM25 weight        0.40
  Vector weight      0.60
  Answer model       `openai/gpt-5.4-mini`
  Judge model        `openai/gpt-5.4`
  Embedding model    `openai/text-embedding-3-small`
  Chroma directory   `wikipedia_chroma_full`
  Temperature        0.2
  Standard log       `checkpoint_3_1_evaluation_1.log`
  Variant input      `checkpoint_3_1_hybrid_variants.json`

The answer-generation system prompt requires the model to use **only the
retrieved documents**, quote evidence where possible, and explicitly say
when the supplied evidence is insufficient rather than guess.

------------------------------------------------------------------------

## 5. Corpus Loading and Chunking

The program loads every `.txt` file in `Wikipedia_text/` and converts
each article into overlapping character chunks.

``` text
chunk size    = 1200
overlap       = 150
step          = 1200 - 150 = 1050
```

Chunk identifiers preserve both article identity and chunk number:

``` text
Nelson_Mandela.txt::chunk_32
Suharto.txt::chunk_15
Abraham_Lincoln.txt::chunk_241
```

In the observed Checkpoint 3.1 run:

``` text
Articles loaded : 65
Chunks created  : 19811
Chunk size      : 1200
Chunk overlap   : 150
```

This is important because evaluation is performed against **retrieved
passages**, not entire Wikipedia articles.

------------------------------------------------------------------------

## 6. Supported Retrieval Strategies

The evaluation script supports three retrievers from the command line.

### BM25

Lexical retrieval based on term overlap and BM25 ranking.

``` powershell
python .\capstone_checkpoint_3_1_evaluation_1.py bm25
```

### Vector

Semantic retrieval over the persistent Chroma vector database.

``` powershell
python .\capstone_checkpoint_3_1_evaluation_1.py vector
```

### Hybrid

Combines normalized BM25 and vector scores.

``` text
hybrid score =
    0.40 × normalized BM25
  + 0.60 × normalized vector similarity
```

Run:

``` powershell
python .\capstone_checkpoint_3_1_evaluation_1.py hybrid
```

The supplied Checkpoint 3.1 results were produced with the **hybrid**
retriever.

------------------------------------------------------------------------

## 7. Evaluation Set Design

`my_eval_set()` contains **15 representative questions**. Each record
contains:

``` python
{
    "question": "...",
    "grading_notes": "..."
}
```

The questions intentionally match the representative retrieval probes
used earlier so results can be compared across the retrieval-development
lifecycle.

### Evaluation categories

  -----------------------------------------------------------------------
  Questions               Category                What is being tested
  ----------------------- ----------------------- -----------------------
  Q1--Q5                  Single-document factual Direct leader facts and
                                                  quotations

  Q6--Q8                  Multi-fact              Multiple facts from one
                                                  leader article

  Q9--Q11                 Cross-document /        Evidence coverage
                          multi-hop               across multiple leaders

  Q12--Q13                Exact-evidence          Exact opening-sentence
                                                  retrieval

  Q14                     Temporal comparison     Multiple UK prime
                                                  ministers and
                                                  chronology

  Q15                     Political-period        Correct article
                          article                 disambiguation
  -----------------------------------------------------------------------

The grading notes are deliberately evidence-oriented. A factually
plausible answer is not always sufficient if the requested supporting
article or passage was not retrieved.

------------------------------------------------------------------------

## 8. LLM-as-Judge Evaluation

The standard evaluation loop performs four operations for every
question:

``` text
Question
   |
   v
Retrieve top chunks
   |
   v
Generate grounded answer
   |
   v
Compare answer with grading_notes
   |
   v
GPT-5.4 judge -> PASS or FAIL
```

The judge is instructed to return exactly one of:

``` text
pass
fail
```

The answer model and judge are intentionally separated:

``` text
Answer model : openai/gpt-5.4-mini
Judge model  : openai/gpt-5.4
```

This provides a stronger evaluator for diagnosing the smaller answer
model and retrieval pipeline.

------------------------------------------------------------------------

## 9. Standard Baseline Result

The supplied hybrid baseline produced:

``` text
10 PASS
5 FAIL
15 total
```

Therefore:

``` text
Hybrid baseline pass rate = 10 / 15 = 66.7%
```

The straightforward single-document questions were generally strong. For
example, the system correctly retrieved and answered questions about
Nelson Mandela, Margaret Thatcher, Queen Victoria, Abraham Lincoln,
Emmanuel Macron, and Justin Trudeau.

The more informative failures occurred on questions requiring **coverage
across several articles**, **precise passage selection**, or **exact
quotation retrieval**.

------------------------------------------------------------------------

## 10. Important Failure Modes

### 10.1 Cross-document source coverage

The Nelson Mandela vs. Suharto comparison demonstrates a
retrieval-coverage failure.

The system retrieved useful Mandela evidence but no Suharto chunk in the
top results. Because the answer prompt prohibits guessing, the model
correctly stated that it could not complete the Suharto portion.

This is a desirable grounding behavior but still an evaluation failure
because the retriever did not supply all required evidence.

``` text
Good generation behavior
        !=
successful retrieval coverage
```

### 10.2 Multi-entity questions

The George Washington / Abraham Lincoln / Barack Obama question exposed
another distinction between factual correctness and retrieval quality.
Some presidential facts could be recovered from list-like passages in
other presidential articles, but the requested evidence was not
consistently retrieved from all three target articles.

For a retrieval evaluation, the intended source coverage matters.

### 10.3 Exact quotation retrieval

Exact opening-sentence questions are harder than general factual
questions.

A retriever may locate the correct article while ranking a heading,
infobox, navigation fragment, or nearby chunk above the exact opening
prose. This means:

``` text
correct article
    does not guarantee
correct evidence passage
```

### 10.4 Judge sensitivity

The first and later logs also demonstrate that grading behavior can
change when the answer or grading note changes slightly. For example,
Suharto moved from FAIL in one run to PASS in a later run despite
substantially relevant retrieved evidence.

This is one reason to preserve logs and inspect failures rather than
treating a single aggregate pass rate as the complete diagnosis.

------------------------------------------------------------------------

## 11. RAGAS Experiment Integration

Checkpoint 3.1 adds a local RAGAS dataset and experiment workflow.

The dataset can be built in two modes:

``` python
load_dataset(use_variants=False)
```

uses the original 15-question `my_eval_set()`.

``` python
load_dataset(use_variants=True)
```

loads the expanded original + paraphrase dataset.

For every RAGAS row, the experiment records:

  Column                Purpose
  --------------------- --------------------------------
  `question`            Evaluation prompt
  `grading_notes`       Expected answer/evidence
  `question_type`       `original` or `variant`
  `retriever`           Retrieval strategy used
  `retrieved_sources`   Retrieved article/chunk IDs
  `retrieved_context`   Full retrieved evidence
  `response`            Generated grounded answer
  `score`               Judge verdict: `pass` / `fail`

This makes the evaluation auditable: a failure can be traced backward
from the score to the answer, retrieved context, and exact source
chunks.

------------------------------------------------------------------------

## 12. Why `retrieved_sources` and `retrieved_context` Both Matter

The experiment intentionally stores two different retrieval fields.

### `retrieved_sources`

Compact diagnostic information:

``` text
Nelson_Mandela.txt::chunk_32;
Nelson_Mandela.txt::chunk_265;
Nelson_Mandela.txt::chunk_31
```

Use this to inspect:

-   source diversity,
-   repeated chunks from one article,
-   whether all required entities were retrieved,
-   whether the intended article appeared at all.

### `retrieved_context`

The complete text passed to the answer model.

Use this to determine:

-   whether the required fact actually appeared,
-   whether the relevant sentence was truncated,
-   whether an exact quotation was available,
-   whether the generator ignored useful retrieved evidence.

Together these fields separate **retrieval failure** from **generation
failure**.

------------------------------------------------------------------------

## 13. Creating Paraphrase Variants

`variants_3_1_FIXED.py` expands the evaluation set by asking
GPT-5.4-mini to generate alternative phrasings that preserve the same
meaning and grading notes.

Example command:

``` powershell
python .\variants_3_1_FIXED.py .\checkpoint_3_1_hybrid.csv 3 .\checkpoint_3_1_hybrid_variants.json
```

The arguments are:

``` text
1. input evaluation CSV/JSON
2. number of paraphrases per original question
3. output JSON
```

With 15 original questions and 3 variants per question:

``` text
15 originals
+ 45 variants
------------
60 questions
```

Every generated record receives:

``` json
"question_type": "original"
```

or:

``` json
"question_type": "variant"
```

The variant keeps the original `grading_notes`, because a valid
paraphrase should require the same answer.

**Important:** automatically generated paraphrases should be manually
reviewed. Remove variants that change meaning, introduce ambiguity, or
duplicate another question.

------------------------------------------------------------------------

## 14. Variant Robustness Dataset

The supplied expanded evaluation file contains:

``` text
15 original questions
45 paraphrase variants
60 total questions
```

This changes the evaluation from:

> "Can the retriever answer this exact wording?"

to:

> "Does the system remain successful when a user expresses the same
> information need differently?"

That is a stronger test of production retrieval robustness.

------------------------------------------------------------------------

## 15. Observed RAGAS / Variant Results

The supplied 60-question experiment produced:

  Question type       PASS     FAIL    Total   Pass rate
  --------------- -------- -------- -------- -----------
  Original              10        5       15       66.7%
  Variant               31       14       45       68.9%
  **Overall**       **41**   **19**   **60**   **68.3%**

The variant pass rate is close to the original baseline rather than
collapsing under rewording. This suggests that the hybrid retriever has
**reasonable paraphrase robustness overall**.

However, the overall score also shows that paraphrasing does not solve
the underlying hard-query problems. Cross-document coverage, passage
precision, and exact-evidence retrieval remain the more important
weaknesses.

------------------------------------------------------------------------

## 16. Interpreting the Results Correctly

A PASS/FAIL score alone cannot explain why the system succeeded or
failed.

For each failed question, inspect the pipeline in this order:

``` text
1. Was the correct article retrieved?
             |
             no -> source retrieval failure
             |
             yes
             v
2. Was the correct passage/chunk retrieved?
             |
             no -> passage ranking failure
             |
             yes
             v
3. Did the answer use the available evidence?
             |
             no -> generation / grounding failure
             |
             yes
             v
4. Did the judge apply the grading notes correctly?
             |
             no -> evaluation / judge failure
             |
             yes
             v
        legitimate FAIL
```

This failure taxonomy is more useful than optimizing only for the
aggregate pass rate.

------------------------------------------------------------------------

## 17. Checkpoint 3.1 Diagnostic Findings

The evaluation supports several conclusions:

1.  **Single-document factual retrieval is the strongest behavior.**
    Named-leader questions frequently retrieve the correct article and
    evidence.

2.  **Hybrid retrieval does not guarantee multi-source coverage.** Top-k
    results can be saturated by several chunks from one highly relevant
    article.

3.  **Article retrieval and passage retrieval are separate problems.**
    The correct Wikipedia page may be present while the exact requested
    sentence is absent.

4.  **Grounded refusal is preferable to hallucination.** When required
    evidence is missing, the answer model often refuses to invent the
    missing fact. This can still score FAIL for task completion while
    demonstrating good reliability behavior.

5.  **Paraphrase robustness is relatively stable.** The 45 variants
    achieved a similar pass rate to the original 15 questions.

6.  **Evaluation artifacts make failures reproducible.** Saving source
    IDs, retrieved context, answers, and judge scores enables
    question-level diagnosis rather than anecdotal assessment.

------------------------------------------------------------------------

## 18. Recommended Next Improvements

The results suggest that the next retrieval improvements should target
evidence coverage rather than simply increasing the number of chunks.

Potential refinements include:

-   source-diversity constraints for multi-entity questions;
-   maximum chunks per article in the final top-k;
-   query decomposition for questions naming multiple leaders;
-   article-aware or entity-aware retrieval;
-   reranking after hybrid candidate generation;
-   dedicated handling for exact-quotation queries;
-   stronger passage-level relevance scoring;
-   evaluation of retrieval metrics separately from answer metrics;
-   repeated judge runs or calibrated grading for borderline cases.

These are **recommended future improvements**, not features claimed to
be implemented in the current baseline.

------------------------------------------------------------------------

## 19. Output Artifacts

### Standard evaluation log

``` text
checkpoint_3_1_evaluation_1.log
```

Records each question, retriever, PASS/FAIL verdict, and generated
answer.

### Baseline experiment data

``` text
checkpoint_3_1_hybrid.csv
checkpoint_3_1_hybrid.json
```

The supplied baseline CSV contains 15 evaluated questions and includes
retrieved sources, retrieved context, responses, and scores.

### Variant dataset

``` text
checkpoint_3_1_hybrid_variants.json
wikipedia_leader_lens_variants_eval.csv
```

Stores the original and paraphrased evaluation questions.

### RAGAS experiment results

``` text
ragas_experiments/
└── experiments/
    └── <generated-experiment>.csv
```

The RAGAS result contains the full evaluation trace for every row.

------------------------------------------------------------------------

## 20. Reproducing the Checkpoint

### Step 1 --- Run the baseline hybrid evaluation

``` powershell
python .\capstone_checkpoint_3_1_evaluation_1.py hybrid
```

### Step 2 --- Generate three paraphrases per baseline question

``` powershell
python .\variants_3_1_FIXED.py .\checkpoint_3_1_hybrid.csv 3 .\checkpoint_3_1_hybrid_variants.json
```

### Step 3 --- Review the generated variants

Verify that each paraphrase:

-   preserves the original meaning,
-   requires the same answer,
-   uses the same grading notes,
-   does not introduce ambiguity,
-   is not a duplicate.

### Step 4 --- Run the evaluation script with variants enabled

The supplied script currently uses:

``` python
dataset = load_dataset(use_variants=True)
```

The RAGAS experiment then evaluates the expanded dataset and saves the
result.

------------------------------------------------------------------------

## 21. Troubleshooting

### `OPENROUTER_API_KEY is not set`

Create `.env`:

``` text
OPENROUTER_API_KEY=your-key
```

Then restart the shell or Python process if necessary.

### Variant evaluation file not found

Confirm that:

``` text
checkpoint_3_1_hybrid_variants.json
```

exists in the working directory, or change `VARIANTS_PATH`.

### Chroma results look inconsistent after changing the corpus

The vector database is persistent. If the corpus, chunking strategy, or
embedding configuration changes substantially, rebuild the appropriate
Chroma index rather than silently reusing an incompatible one.

### RAGAS CSV is not where expected

Check:

``` text
ragas_experiments/experiments/
```

The generated experiment name may differ between runs.

### Multi-leader question retrieves only one leader

This is a known baseline weakness. Increasing `NUM_RETRIEVED` may help,
but a better long-term solution is source-diverse retrieval or query
decomposition.

### Correct answer receives FAIL

Inspect:

1.  the exact grading notes;
2.  whether all requested evidence was included;
3.  whether a required quotation/source was missing;
4.  whether the judge is enforcing a stricter criterion than factual
    correctness alone.

------------------------------------------------------------------------

## 22. Relationship to the Capstone Progression

``` text
Checkpoint 1.1
Baseline / representative questions
        |
        v
Checkpoint 2.1
BM25 + Vector + Hybrid retrieval
Chunk-level evidence retrieval
        |
        v
Checkpoint 3.1
Evaluation infrastructure
LLM judge + PASS/FAIL diagnosis
RAGAS experiment
        |
        v
Variant robustness testing
Original + paraphrased questions
        |
        v
Later checkpoints
Advanced retrieval / graph retrieval
Agentic workflows
Security and reliability
Production optimization
```

Checkpoint 3.1 is therefore the point where retrieval quality becomes
**measurable and reproducible**.

------------------------------------------------------------------------

## 23. Key Takeaway

The main finding from this checkpoint is not simply that the hybrid
system achieved a **66.7% baseline pass rate** or a **68.3% pass rate on
the 60-question expanded experiment**.

The more important result is that the evaluation framework reveals
**why** the system fails.

The Leader Lens pipeline is strong on many direct single-leader
questions and remains reasonably stable under paraphrasing. Its main
weakness is **evidence coverage and passage precision on harder
queries**---especially questions requiring several articles or exact
quotations.

By storing the question, grading notes, retrieved source IDs, retrieved
context, generated response, and judge score, Checkpoint 3.1 turns those
weaknesses into concrete targets for the advanced retrieval work that
follows.
