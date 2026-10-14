"""
Standalone LLM-as-a-Judge evaluator for Leader Lens Checkpoint 5.1.

Purpose
-------
Evaluate an EXISTING checkpoint_5_1_readable_output.txt against
wikipedia_GT_15.xlsx without rerunning Fixed / Agentic / Tool-Using retrieval.

Expected files in the working directory:
    checkpoint_5_1_readable_output.txt
    wikipedia_GT_15.xlsx
    .env  (with OPENROUTER_API_KEY)

Outputs:
    checkpoint_5_1_evaluation_results.csv
    checkpoint_5_1_evaluation_summary.csv
    checkpoint_5_1_judged_report.txt

Install:
    pip install pandas openpyxl python-dotenv langchain-openai langchain-core

Run:
    python -X utf8 evaluate_checkpoint_5_1_results.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

RESULTS_FILE = Path(
    os.getenv("CHECKPOINT_RESULTS_FILE", "checkpoint_5_1_readable_output.txt")
)
GROUND_TRUTH_FILE = Path(
    os.getenv("GROUND_TRUTH_FILE", "wikipedia_GT_15.xlsx")
)
GROUND_TRUTH_SHEET = os.getenv("GROUND_TRUTH_SHEET", "Sheet1")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.4")
JUDGE_TEMPERATURE = 0.0

RESULTS_CSV = Path("checkpoint_5_1_evaluation_results.csv")
SUMMARY_CSV = Path("checkpoint_5_1_evaluation_summary.csv")
JUDGED_REPORT = Path("checkpoint_5_1_judged_report.txt")

WORKFLOW_LABELS = {
    "FIXED": "FIXED",
    "AGENTIC": "AGENTIC",
    "TOOL USING AGENT": "TOOL_USING_AGENT",
    "TOOL_USING_AGENT": "TOOL_USING_AGENT",
}


JUDGE_SYSTEM = """
You are an independent evaluator of a Wikipedia retrieval-augmented
generation system.

Evaluate the GENERATED ANSWER only against:
1. the QUESTION,
2. the GRADING NOTES,
3. the REQUIRED SOURCE ARTICLES,
4. the SCOPE AND ASSUMPTIONS.

Do not reward outside knowledge. Do not silently repair an incomplete
answer using your own knowledge.

Score each dimension from 0 to 4.

correctness:
0 = incorrect
1 = mostly incorrect
2 = partially correct
3 = mostly correct
4 = fully correct relative to the grading notes

completeness:
0 = none of the requested components are answered
1 = few components
2 = some components
3 = most components
4 = every requested component is answered

groundedness:
0 = factual claims are unsupported / citations absent
1 = mostly unsupported
2 = partially supported
3 = mostly supported by the answer's cited passage IDs
4 = claims are consistently tied to cited passage IDs

source_compliance:
0 = wrong sources
1 = major source violations
2 = partial compliance
3 = minor source issue
4 = required article/source constraints are satisfied

IMPORTANT:
The standalone report contains passage IDs such as [Article.txt#p12], but not
the full retrieved passage text. Therefore groundedness here measures citation
coverage and source alignment visible in the generated answer. It does NOT
independently verify the quoted passage contents.

Return ONLY valid JSON in exactly this shape:
{
  "correctness": 0,
  "completeness": 0,
  "groundedness": 0,
  "source_compliance": 0,
  "missing_requirements": [],
  "unsupported_claims": [],
  "reason": ""
}
"""


# ---------------------------------------------------------------------
# FILE LOADING
# ---------------------------------------------------------------------

def load_ground_truth(
    file_path: Path = GROUND_TRUTH_FILE,
    sheet_name: str = GROUND_TRUTH_SHEET,
) -> pd.DataFrame:
    """Load and validate the ground-truth workbook."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Ground-truth workbook not found: {file_path.resolve()}"
        )

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        engine="openpyxl",
    )

    required = {
        "Question",
        "grading_notes",
        "source_articles",
        "scope and assumptions",
        "evaluation_role",
        "complexity",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Ground-truth workbook is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.dropna(subset=["Question"]).copy()
    df["Question"] = df["Question"].astype(str).str.strip()

    return df.reset_index(drop=True)


def load_readable_report(file_path: Path = RESULTS_FILE) -> str:
    """Load the previously generated human-readable checkpoint report."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Readable results file not found: {file_path.resolve()}"
        )

    return file_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------
# REPORT PARSING
# ---------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    text = str(value or "").strip()

    replacements = {
        "ΓÇ£": '"',
        "ΓÇ¥": '"',
        "ΓÇÖ": "'",
        "ΓÇÿ": "'",
        "ΓÇô": "-",
        "ΓÇö": "--",
        "ΓåÆ": "->",
        "├º": "ç",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def _extract_metric(block: str, label: str) -> str | None:
    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$",
        block,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def parse_readable_report(report_text: str) -> list[dict]:
    """
    Parse checkpoint_5_1_readable_output.txt.

    Returns one record per workflow per test:
        test
        question
        workflow
        answer
        stop_reason
        elapsed_seconds
        passages_retrieved
    """

    test_pattern = re.compile(
        r"(?ms)^={20,}\s*\n"
        r"TEST\s+(\d+)\s*\n"
        r"={20,}\s*\n"
        r"QUESTION\s*\n"
        r"(.*?)"
        r"(?=^={20,}\s*\nTEST\s+\d+\s*\n|"
        r"^={20,}\s*\nEvaluation complete\.|\Z)"
    )

    workflow_pattern = re.compile(
        r"(?ms)^---\s+(FIXED|AGENTIC|TOOL USING AGENT|TOOL_USING_AGENT)\s+---\s*\n"
        r"(.*?)"
        r"(?=^---\s+(?:FIXED|AGENTIC|TOOL USING AGENT|TOOL_USING_AGENT)\s+---|"
        r"^WORKFLOW COMPARISON|\Z)"
    )

    records: list[dict] = []

    for test_match in test_pattern.finditer(report_text):
        test_number = int(test_match.group(1))
        test_block = test_match.group(2)

        first_workflow = re.search(
            r"(?m)^---\s+(?:FIXED|AGENTIC|TOOL USING AGENT|TOOL_USING_AGENT)\s+---",
            test_block,
        )
        if not first_workflow:
            continue

        question = _clean_text(
            test_block[:first_workflow.start()]
        )

        for workflow_match in workflow_pattern.finditer(test_block):
            raw_label = workflow_match.group(1).strip()
            workflow = WORKFLOW_LABELS[raw_label]
            block = workflow_match.group(2)

            answer_match = re.search(
                r"(?ms)^ANSWER\s*\n(.*?)\n\s*RUN SUMMARY\s*$",
                block,
            )

            answer = (
                _clean_text(answer_match.group(1))
                if answer_match
                else ""
            )

            stop_reason = _extract_metric(block, "Stop reason") or "n/a"

            elapsed_raw = _extract_metric(block, "Time") or "0"
            elapsed_match = re.search(r"[-+]?\d*\.?\d+", elapsed_raw)
            elapsed_seconds = (
                float(elapsed_match.group(0))
                if elapsed_match
                else 0.0
            )

            passages_raw = _extract_metric(block, "Passages retrieved") or "0"
            passages_match = re.search(r"\d+", passages_raw)
            passages_retrieved = (
                int(passages_match.group(0))
                if passages_match
                else 0
            )

            records.append(
                {
                    "test": test_number,
                    "question": question,
                    "workflow": workflow,
                    "answer": answer,
                    "stop_reason": stop_reason,
                    "elapsed_seconds": elapsed_seconds,
                    "passages_retrieved": passages_retrieved,
                }
            )

    if not records:
        raise ValueError(
            "No workflow results could be parsed from the readable report. "
            "Confirm that the file contains TEST sections and "
            "'--- FIXED ---', '--- AGENTIC ---', and "
            "'--- TOOL USING AGENT ---' headings."
        )

    return records


# ---------------------------------------------------------------------
# QUESTION MATCHING
# ---------------------------------------------------------------------

def normalize_question(value: str) -> str:
    """Normalize whitespace/typographic punctuation for safe exact matching."""

    value = _clean_text(value)
    value = value.replace("\u00a0", " ")
    value = value.replace("’", "'").replace("‘", "'")
    value = value.replace("“", '"').replace("”", '"')
    value = re.sub(r"\s+", " ", value)

    return value.strip().casefold()


def build_ground_truth_lookup(df: pd.DataFrame) -> dict[str, dict]:
    """
    Build ground-truth lookup keyed by normalized Question.

    Identical duplicate questions are allowed and collapsed to one row.
    Conflicting duplicates raise an error because it would be ambiguous
    which grading notes should be used.
    """

    lookup: dict[str, dict] = {}

    comparison_fields = [
        "grading_notes",
        "source_articles",
        "scope and assumptions",
        "evaluation_role",
        "complexity",
    ]

    for _, row in df.iterrows():
        key = normalize_question(row["Question"])
        current = row.to_dict()

        # First occurrence of this question
        if key not in lookup:
            lookup[key] = current
            continue

        # Question already exists. Check whether the duplicate
        # contains the same grading information.
        existing = lookup[key]

        conflicts = []

        for field in comparison_fields:
            old_value = str(existing.get(field, "")).strip()
            new_value = str(current.get(field, "")).strip()

            if old_value != new_value:
                conflicts.append(field)

        if conflicts:
            raise ValueError(
                "\nConflicting duplicate Question found in ground truth:\n"
                f"{row['Question']}\n\n"
                "The duplicate rows disagree in:\n"
                f"{', '.join(conflicts)}\n\n"
                "Please remove one duplicate or reconcile those fields "
                "in wikipedia_GT_15.xlsx."
            )

        # Otherwise it is an identical duplicate.
        print(
            "WARNING: Identical duplicate ground-truth question "
            "found; using the first occurrence:\n"
            f"  {row['Question']}"
        )

    return lookup


# ---------------------------------------------------------------------
# LLM JUDGE
# ---------------------------------------------------------------------

def make_judge_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Put it in your .env file or environment."
        )

    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def _parse_judge_json(raw: str) -> dict:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge returned invalid JSON:\n{raw}"
        ) from exc

    required_scores = (
        "correctness",
        "completeness",
        "groundedness",
        "source_compliance",
    )

    for key in required_scores:
        value = data.get(key)

        if not isinstance(value, int) or not 0 <= value <= 4:
            raise ValueError(
                f"Judge field {key!r} must be an integer from 0 to 4. "
                f"Received: {value!r}"
            )

    for key in ("missing_requirements", "unsupported_claims"):
        if not isinstance(data.get(key, []), list):
            data[key] = [str(data.get(key))]

    data["reason"] = str(data.get("reason", "")).strip()

    return data


def evaluate_answer(
    judge_llm: ChatOpenAI,
    *,
    question: str,
    answer: str,
    grading_notes: str,
    source_articles: str,
    scope: str,
    evaluation_role: str,
    complexity: str,
) -> dict:
    """Judge one generated answer against one ground-truth row."""

    prompt = f"""
QUESTION
{question}

GRADING NOTES
{grading_notes}

REQUIRED SOURCE ARTICLES
{source_articles}

SCOPE AND ASSUMPTIONS
{scope}

EVALUATION ROLE
{evaluation_role}

COMPLEXITY
{complexity}

GENERATED ANSWER
{answer}

Evaluate the generated answer now.
""".strip()

    raw = judge_llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=prompt),
        ]
    ).content

    return _parse_judge_json(raw)


# ---------------------------------------------------------------------
# DETERMINISTIC SCORING
# ---------------------------------------------------------------------

def calculate_overall_score(evaluation: dict) -> float:
    dimensions = [
        evaluation["correctness"],
        evaluation["completeness"],
        evaluation["groundedness"],
        evaluation["source_compliance"],
    ]

    return round((sum(dimensions) / 16.0) * 100.0, 1)


def assign_verdict(score: float) -> str:
    if score >= 85:
        return "PASS"
    if score >= 60:
        return "PARTIAL"
    return "FAIL"


# ---------------------------------------------------------------------
# MAIN STANDALONE EVALUATION FUNCTION
# ---------------------------------------------------------------------

def evaluate_saved_results(
    results_file: Path = RESULTS_FILE,
    ground_truth_file: Path = GROUND_TRUTH_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate an existing readable TXT report against wikipedia_GT_15.xlsx.

    This function does NOT rerun retrieval or answer generation.
    """

    ground_truth = load_ground_truth(ground_truth_file)
    gt_lookup = build_ground_truth_lookup(ground_truth)

    report_text = load_readable_report(results_file)
    generated_results = parse_readable_report(report_text)

    judge_llm = make_judge_llm()

    rows: list[dict] = []

    print(
        f"Parsed {len(generated_results)} generated workflow answers "
        f"from {results_file.name}."
    )
    print(
        f"Loaded {len(ground_truth)} ground-truth rows "
        f"from {ground_truth_file.name}."
    )
    print(f"Judge model: {JUDGE_MODEL}\n")

    for item in generated_results:
        question = item["question"]
        key = normalize_question(question)
        gt = gt_lookup.get(key)

        if gt is None:
            print(
                f"WARNING: No exact ground-truth question match for "
                f"TEST {item['test']} / {item['workflow']}."
            )
            rows.append(
                {
                    **item,
                    "category": "",
                    "source_articles": "",
                    "evaluation_role": "",
                    "complexity": "",
                    "correctness": None,
                    "completeness": None,
                    "groundedness": None,
                    "source_compliance": None,
                    "overall_score": None,
                    "verdict": "NO_GT_MATCH",
                    "missing_requirements": "",
                    "unsupported_claims": "",
                    "judge_reason": "No exact ground-truth question match.",
                }
            )
            continue

        print(
            f"Evaluating TEST {item['test']} / {item['workflow']} ..."
        )

        evaluation = evaluate_answer(
            judge_llm,
            question=question,
            answer=item["answer"],
            grading_notes=str(gt["grading_notes"]),
            source_articles=str(gt["source_articles"]),
            scope=str(gt["scope and assumptions"]),
            evaluation_role=str(gt["evaluation_role"]),
            complexity=str(gt["complexity"]),
        )

        score = calculate_overall_score(evaluation)
        verdict = assign_verdict(score)

        rows.append(
            {
                **item,
                "category": str(gt.get("category", "")),
                "source_articles": str(gt["source_articles"]),
                "evaluation_role": str(gt["evaluation_role"]),
                "complexity": str(gt["complexity"]),
                "correctness": evaluation["correctness"],
                "completeness": evaluation["completeness"],
                "groundedness": evaluation["groundedness"],
                "source_compliance": evaluation["source_compliance"],
                "overall_score": score,
                "verdict": verdict,
                "missing_requirements": json.dumps(
                    evaluation.get("missing_requirements", []),
                    ensure_ascii=False,
                ),
                "unsupported_claims": json.dumps(
                    evaluation.get("unsupported_claims", []),
                    ensure_ascii=False,
                ),
                "judge_reason": evaluation["reason"],
            }
        )

    results_df = pd.DataFrame(rows)

    scored = results_df[
        pd.to_numeric(
            results_df["overall_score"],
            errors="coerce",
        ).notna()
    ].copy()

    if scored.empty:
        summary_df = pd.DataFrame(
            columns=[
                "workflow",
                "questions_evaluated",
                "mean_score",
                "pass_rate",
                "partial_rate",
                "fail_rate",
                "mean_correctness",
                "mean_completeness",
                "mean_groundedness",
                "mean_source_compliance",
                "mean_latency_seconds",
                "mean_passages_retrieved",
            ]
        )
    else:
        summary_rows = []

        for workflow, group in scored.groupby("workflow", sort=False):
            summary_rows.append(
                {
                    "workflow": workflow,
                    "questions_evaluated": len(group),
                    "mean_score": round(group["overall_score"].mean(), 2),
                    "pass_rate": round(
                        (group["verdict"] == "PASS").mean() * 100,
                        2,
                    ),
                    "partial_rate": round(
                        (group["verdict"] == "PARTIAL").mean() * 100,
                        2,
                    ),
                    "fail_rate": round(
                        (group["verdict"] == "FAIL").mean() * 100,
                        2,
                    ),
                    "mean_correctness": round(group["correctness"].mean(), 2),
                    "mean_completeness": round(group["completeness"].mean(), 2),
                    "mean_groundedness": round(group["groundedness"].mean(), 2),
                    "mean_source_compliance": round(
                        group["source_compliance"].mean(),
                        2,
                    ),
                    "mean_latency_seconds": round(
                        group["elapsed_seconds"].mean(),
                        2,
                    ),
                    "mean_passages_retrieved": round(
                        group["passages_retrieved"].mean(),
                        2,
                    ),
                }
            )

        summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    write_judged_report(results_df, summary_df)

    return results_df, summary_df


# ---------------------------------------------------------------------
# HUMAN-READABLE JUDGED REPORT
# ---------------------------------------------------------------------

def write_judged_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:

    lines = [
        "LEADER LENS - CHECKPOINT 5.1 GROUND-TRUTH EVALUATION",
        "=" * 78,
        f"Judge model: {JUDGE_MODEL}",
        f"Generated-results source: {RESULTS_FILE}",
        f"Ground truth: {GROUND_TRUTH_FILE}",
        "",
    ]

    for test_number in sorted(results_df["test"].unique()):
        test_rows = results_df[results_df["test"] == test_number]

        if test_rows.empty:
            continue

        question = test_rows.iloc[0]["question"]

        lines.extend(
            [
                "=" * 78,
                f"TEST {test_number}",
                "=" * 78,
                "QUESTION",
                str(question),
                "",
            ]
        )

        for _, row in test_rows.iterrows():
            lines.extend(
                [
                    f"--- {row['workflow']} ---",
                    f"Correctness       : {row['correctness']}/4",
                    f"Completeness      : {row['completeness']}/4",
                    f"Groundedness      : {row['groundedness']}/4",
                    f"Source compliance : {row['source_compliance']}/4",
                    f"Overall score     : {row['overall_score']}%",
                    f"Verdict           : {row['verdict']}",
                    f"Time              : {row['elapsed_seconds']} s",
                    f"Passages retrieved: {row['passages_retrieved']}",
                    f"Judge reason      : {row['judge_reason']}",
                    f"Missing           : {row['missing_requirements']}",
                    f"Unsupported       : {row['unsupported_claims']}",
                    "",
                ]
            )

    lines.extend(
        [
            "=" * 78,
            "WORKFLOW SUMMARY",
            "=" * 78,
            summary_df.to_string(index=False)
            if not summary_df.empty
            else "No scored results.",
            "",
            "NOTE",
            "Groundedness in this standalone evaluator measures citation coverage "
            "and source alignment visible in the saved TXT. Full passage-content "
            "verification requires the original retrieved passage text.",
        ]
    )

    JUDGED_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    results_df, summary_df = evaluate_saved_results()

    print("\n" + "=" * 78)
    print("GROUND-TRUTH EVALUATION COMPLETE")
    print("=" * 78)

    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    print(f"\nDetailed CSV : {RESULTS_CSV.resolve()}")
    print(f"Summary CSV  : {SUMMARY_CSV.resolve()}")
    print(f"Judged report: {JUDGED_REPORT.resolve()}")


if __name__ == "__main__":
    main()
