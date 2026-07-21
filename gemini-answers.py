from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator


# ============================================================
# CONFIGURATION
# ============================================================

# Input:
# questions/<company_name>.txt
QUESTIONS_DIR = Path("questions")

# Output:
# answers_gemini/<YYYY-MM-DD>/<company_name>_answers_gemini.txt
ANSWERS_ROOT_DIR = Path("answers_gemini")

# Logs:
# gemini_logs/gemini_usage_<YYYY-MM-DD>.csv
LOGS_DIR = Path("gemini_logs")

# Raw JSON responses are retained for reproducibility/debugging.
SAVE_RAW_JSON_RESPONSES = True

# Stable, lower-cost Gemini model.
MODEL_NAME = "gemini-3.1-flash-lite"

EXPECTED_QUESTION_COUNT = 30

# The visible response should normally be around 3,000–4,500 tokens.
# Thinking tokens also count toward output usage, so this provides headroom.
MAX_OUTPUT_TOKENS = 7_000

# Gemini 3.1 Flash-Lite supports configurable thinking.
# "low" reduces thinking cost while retaining some reasoning capability.
THINKING_LEVEL = "low"

# Lower temperature improves consistency between daily runs.
TEMPERATURE = 0.2

# Delay between companies to avoid request-per-minute pressure.
SECONDS_BETWEEN_COMPANIES = 8

# The first request plus this many retries.
MAX_RETRIES = 3

# Safety limit for one execution.
# A normal 100-company run makes 100 generation requests.
MAX_GENERATION_ATTEMPTS_PER_RUN = 140

# Skip a company when a complete output already exists for the date.
SKIP_COMPLETED_FILES = True

# Stop rather than retry all 100 companies when the project has zero quota.
STOP_ON_PERMANENT_QUOTA_ERROR = True


# ============================================================
# STRUCTURED GEMINI RESPONSE
# ============================================================

class AnswerItem(BaseModel):
    """
    One Gemini-generated answer.

    The question itself is deliberately excluded. It will be inserted
    locally from the original question file.
    """

    question_number: int = Field(
        ge=1,
        le=EXPECTED_QUESTION_COUNT,
        description=(
            "The number of the question being answered. "
            "Use each number from 1 through 30 exactly once."
        ),
    )

    answer: str = Field(
        min_length=20,
        description=(
            "A current, evidence-based answer of approximately 60-100 words. "
            "Do not repeat the question."
        ),
    )

    confidence: Literal["High", "Medium", "Low"] = Field(
        description=(
            "Confidence in the answer based on the quality, recency and "
            "agreement of the evidence found."
        )
    )

    evidence_date: str = Field(
        description=(
            "The most relevant evidence date in YYYY-MM-DD format where "
            "possible, or 'Unknown' if no reliable date is available."
        )
    )

    outlook: Literal[
        "Positive",
        "Negative",
        "Mixed",
        "Neutral",
        "Insufficient evidence",
    ] = Field(
        description=(
            "The implication of the evidence for the company's outlook."
        )
    )

    @field_validator("answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()

        if not value:
            raise ValueError("Answer cannot be empty.")

        return value

    @field_validator("evidence_date")
    @classmethod
    def clean_evidence_date(cls, value: str) -> str:
        value = value.strip()

        return value or "Unknown"


class CompanyResearchResponse(BaseModel):
    answers: list[AnswerItem] = Field(
        description=(
            "Exactly 30 answers, ordered by question_number from 1 to 30."
        )
    )

    overall_company_outlook: str = Field(
        min_length=20,
        description=(
            "A concise overall company outlook of no more than 120 words."
        ),
    )

    @field_validator("overall_company_outlook")
    @classmethod
    def clean_outlook(cls, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()


# ============================================================
# RUN TOTALS
# ============================================================

@dataclass
class UsageTotals:
    generation_attempts: int = 0
    completed_companies: int = 0
    skipped_companies: int = 0
    incomplete_companies: int = 0
    failed_companies: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0


# ============================================================
# CUSTOM ERRORS
# ============================================================

class PermanentQuotaError(RuntimeError):
    """Raised when the selected project/model has no usable quota."""


class IncompleteResponseError(RuntimeError):
    """Raised when Gemini does not return one valid answer per question."""


# ============================================================
# FILE UTILITIES
# ============================================================

def safe_filename(name: str) -> str:
    """
    Convert a company name into a Windows/macOS/Linux-safe filename.
    """
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.rstrip(". ")


def natural_sort_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def company_name_from_path(question_file: Path) -> str:
    return question_file.stem.strip()


def find_question_files() -> list[Path]:
    if not QUESTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Questions directory not found:\n{QUESTIONS_DIR.resolve()}"
        )

    question_files = sorted(
        QUESTIONS_DIR.glob("*.txt"),
        key=natural_sort_key,
    )

    if not question_files:
        raise FileNotFoundError(
            f"No .txt question files were found in:\n"
            f"{QUESTIONS_DIR.resolve()}"
        )

    return question_files


# ============================================================
# QUESTION PARSING
# ============================================================

def parse_questions(question_text: str) -> list[str]:
    """
    Parse numbered questions while preserving complete multi-line text.

    Supported formats include:

        1. Question text
        2) Question text
        3: Question text
        Question 4: Question text
        Q5. Question text
        **Question 6:** Question text
    """
    text = (
        question_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if not text:
        return []

    # Flags are passed separately to avoid the previous:
    # "global flags not at the start" regex error.
    numbered_question_pattern = re.compile(
        r"""
        ^\s*
        (?:\*\*)?
        (?:
            question\s*
            |
            q\s*
        )?
        \d{1,3}
        \s*
        [.)\-:]
        \s*
        (?:\*\*)?
        """,
        flags=(
            re.IGNORECASE
            | re.MULTILINE
            | re.VERBOSE
        ),
    )

    matches = list(
        numbered_question_pattern.finditer(text)
    )

    questions: list[str] = []

    if matches:
        for index, match in enumerate(matches):
            question_start = match.end()

            if index + 1 < len(matches):
                question_end = matches[index + 1].start()
            else:
                question_end = len(text)

            question = text[
                question_start:question_end
            ].strip()

            question = question.strip("* \t\n")
            question = re.sub(r"\s+", " ", question).strip()

            if question:
                questions.append(question)

        return questions

    # Fallback: blank-line-separated questions.
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if len(paragraphs) > 1:
        return paragraphs

    # Final fallback: one question per line.
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]


def format_questions_for_prompt(
    questions: list[str],
) -> str:
    """
    Number the original questions for the input prompt.

    Gemini sees the question text but is explicitly instructed not to
    repeat it in its output.
    """
    return "\n\n".join(
        f"{number}. {question}"
        for number, question in enumerate(
            questions,
            start=1,
        )
    )


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_INSTRUCTION = """
You are an expert financial research analyst conducting a longitudinal
study of publicly traded companies.

Use Google Search to research current, publicly available evidence and
answer every supplied question accurately and consistently.

Your output will be compared with outputs produced by a local language
model using a separate Finnhub and retrieval-augmented generation
pipeline. Therefore, factual grounding, consistency and coverage of all
questions are essential.

Research rules:

1. Use Google Search for current information relevant to each question.
2. Prioritise authoritative sources:
   - official company investor-relations pages;
   - regulatory filings;
   - earnings releases and presentations;
   - recognised exchanges and regulators;
   - official company announcements;
   - reputable financial news organisations.
3. Give greater weight to recent evidence, while using older information
   only when it provides necessary context.
4. Distinguish confirmed facts from reasonable interpretation.
5. Never invent figures, dates, events, quotations or sources.
6. If reliable current evidence cannot be found, state:
   "Insufficient current evidence was found."
7. Focus on information potentially relevant to business performance,
   risk, investor sentiment or future stock performance.
8. Do not provide personalised investment advice.
9. Do not recommend buying, selling or holding a security.
10. Avoid generic financial explanations and unnecessary background.
11. Avoid repeating the same evidence unless it is directly necessary
    for answering another question.
12. Return only the structured response requested by the supplied schema.
""".strip()


def build_prompt(
    company_name: str,
    questions: list[str],
    analysis_date: str,
) -> str:
    """
    Build one grounded request containing all questions for one company.
    """
    question_block = format_questions_for_prompt(
        questions
    )

    return f"""
COMPANY
{company_name}

ANALYSIS DATE
{analysis_date}

NUMBER OF QUESTIONS
{len(questions)}

TASK

Use Google Search to answer all {len(questions)} questions about
{company_name}.

The answers form part of a repeated daily research experiment. Use only
information publicly available on or before {analysis_date}. Do not refer
to information published after this date.

ANSWER REQUIREMENTS

- Return exactly one answer for every question.
- Use question numbers 1 through {len(questions)} exactly once.
- Preserve the original order.
- Do not omit, merge or combine questions.
- Do not repeat the question text in the answer.
- Keep each answer approximately 60-100 words.
- Include enough concrete detail to explain:
  1. what happened or what the current evidence shows;
  2. why it may matter to the company;
  3. the potential implication for the outlook.
- Include material dates, figures and named developments when reliable.
- Remain concise and avoid repeated background information.
- Set confidence to High, Medium or Low.
- Set evidence_date to the most relevant evidence date in YYYY-MM-DD
  format where possible. Use "Unknown" if no defensible date exists.
- Set outlook to Positive, Negative, Mixed, Neutral or
  Insufficient evidence.
- Keep the final overall company outlook to no more than 120 words.
- Do not place the original question text in any response field.

QUESTIONS

{question_block}
""".strip()


# ============================================================
# RESPONSE USAGE AND VALIDATION
# ============================================================

def get_integer_attribute(
    source: Any,
    *attribute_names: str,
) -> int:
    if source is None:
        return 0

    for attribute_name in attribute_names:
        value = getattr(
            source,
            attribute_name,
            None,
        )

        if isinstance(value, int):
            return value

    return 0


def extract_usage(
    response: Any,
) -> dict[str, int]:
    """
    Extract usage metadata while tolerating SDK field-name changes.
    """
    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    input_tokens = get_integer_attribute(
        usage,
        "prompt_token_count",
        "input_token_count",
        "total_input_tokens",
    )

    output_tokens = get_integer_attribute(
        usage,
        "candidates_token_count",
        "output_token_count",
        "total_output_tokens",
    )

    thinking_tokens = get_integer_attribute(
        usage,
        "thoughts_token_count",
        "thinking_token_count",
    )

    total_tokens = get_integer_attribute(
        usage,
        "total_token_count",
        "total_tokens",
    )

    if total_tokens == 0:
        total_tokens = (
            input_tokens
            + output_tokens
            + thinking_tokens
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
    }


def validate_company_response(
    parsed_response: CompanyResearchResponse,
    expected_count: int,
) -> CompanyResearchResponse:
    """
    Verify that Gemini returned exactly one answer for every question.
    """
    if len(parsed_response.answers) != expected_count:
        raise IncompleteResponseError(
            f"Expected {expected_count} answer objects but received "
            f"{len(parsed_response.answers)}."
        )

    question_numbers = [
        item.question_number
        for item in parsed_response.answers
    ]

    expected_numbers = list(
        range(1, expected_count + 1)
    )

    if sorted(question_numbers) != expected_numbers:
        missing = sorted(
            set(expected_numbers)
            - set(question_numbers)
        )

        duplicates = sorted({
            number
            for number in question_numbers
            if question_numbers.count(number) > 1
        })

        raise IncompleteResponseError(
            "Invalid question-number coverage. "
            f"Missing: {missing or 'none'}; "
            f"duplicates: {duplicates or 'none'}."
        )

    # Enforce deterministic order for final-file generation.
    parsed_response.answers.sort(
        key=lambda item: item.question_number
    )

    return parsed_response


# ============================================================
# ERROR HANDLING
# ============================================================

def is_permanent_quota_error(
    error: Exception,
) -> bool:
    error_text = str(error).lower()

    permanent_indicators = (
        "limit: 0",
        "quota limit is 0",
        "billing account",
        "billing is not enabled",
        "grounding with google search is not available",
    )

    return any(
        indicator in error_text
        for indicator in permanent_indicators
    )


def is_retryable_error(
    error: Exception,
) -> bool:
    error_text = str(error).lower()

    retryable_indicators = (
        "429",
        "resource_exhausted",
        "too many requests",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "internal",
        "unavailable",
        "deadline",
        "timeout",
        "connection",
        "temporarily",
    )

    return any(
        indicator in error_text
        for indicator in retryable_indicators
    )


def get_retry_delay(
    error: Exception,
    retry_number: int,
) -> float:
    """
    Honour Google's stated retry delay where it can be extracted.
    Otherwise, use exponential backoff with jitter.
    """
    error_text = str(error)

    retry_match = re.search(
        r"retry(?:ing)?\s+in\s+([\d.]+)s",
        error_text,
        flags=re.IGNORECASE,
    )

    if retry_match:
        return float(retry_match.group(1)) + 2

    delay_match = re.search(
        r"retryDelay['\":\s]+(\d+)s",
        error_text,
        flags=re.IGNORECASE,
    )

    if delay_match:
        return float(delay_match.group(1)) + 2

    return min(
        180,
        15 * (2 ** retry_number)
        + random.uniform(1, 5),
    )


def save_failure(
    failure_directory: Path,
    company_name: str,
    error: Exception | str,
) -> None:
    failure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_file = failure_directory / (
        f"{safe_filename(company_name)}"
        f"_gemini_failure.txt"
    )

    failure_file.write_text(
        (
            f"COMPANY: {company_name}\n"
            f"TIME: "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
            f"ERROR:\n{error}\n"
        ),
        encoding="utf-8",
    )


# ============================================================
# OUTPUT CONSTRUCTION
# ============================================================

def build_final_output(
    company_name: str,
    analysis_date: str,
    questions: list[str],
    result: CompanyResearchResponse,
    usage: dict[str, int],
    generated_at: str,
) -> str:
    """
    Construct the final text file locally.

    The original questions come directly from the question file.
    Gemini does not spend output tokens repeating them.
    """
    answer_by_number = {
        item.question_number: item
        for item in result.answers
    }

    lines: list[str] = [
        "PIPELINE STATUS: COMPLETE",
        f"COMPANY: {company_name}",
        f"ANALYSIS DATE: {analysis_date}",
        f"MODEL: {MODEL_NAME}",
        "GOOGLE SEARCH GROUNDING: ENABLED",
        f"GENERATED AT: {generated_at}",
        f"QUESTIONS EXPECTED: {len(questions)}",
        f"ANSWERS RECEIVED: {len(result.answers)}",
        f"INPUT TOKENS: {usage['input_tokens']}",
        f"VISIBLE OUTPUT TOKENS: {usage['output_tokens']}",
        f"THINKING TOKENS: {usage['thinking_tokens']}",
        f"TOTAL TOKENS: {usage['total_tokens']}",
        "=" * 76,
        "",
    ]

    for question_number, question in enumerate(
        questions,
        start=1,
    ):
        answer_item = answer_by_number[
            question_number
        ]

        lines.extend([
            f"QUESTION {question_number}",
            question,
            "",
            "ANSWER",
            answer_item.answer,
            "",
            "CONFIDENCE",
            answer_item.confidence,
            "",
            "EVIDENCE DATE",
            answer_item.evidence_date,
            "",
            "OUTLOOK",
            answer_item.outlook,
            "",
            "-" * 76,
            "",
        ])

    lines.extend([
        "OVERALL COMPANY OUTLOOK",
        result.overall_company_outlook,
        "",
    ])

    return "\n".join(lines)


def existing_output_is_complete(
    output_file: Path,
) -> bool:
    if not output_file.exists():
        return False

    try:
        text = output_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False

    if "PIPELINE STATUS: COMPLETE" not in text:
        return False

    question_headings = re.findall(
        r"(?m)^QUESTION\s+(\d+)\s*$",
        text,
    )

    return (
        len(set(question_headings))
        == EXPECTED_QUESTION_COUNT
        and "OVERALL COMPANY OUTLOOK" in text
    )


# ============================================================
# CSV AND JSON LOGGING
# ============================================================

CSV_FIELDNAMES = [
    "timestamp",
    "company",
    "question_file",
    "output_file",
    "status",
    "question_count",
    "answer_count",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "total_tokens",
    "api_attempts_for_company",
    "generation_attempt_number",
    "duration_seconds",
    "error",
]


def append_csv_log(
    log_file: Path,
    row: dict[str, Any],
) -> None:
    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = log_file.exists()

    with log_file.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDNAMES,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_run_summary(
    summary_file: Path,
    selected_files: list[Path],
    totals: UsageTotals,
) -> None:
    summary = {
        "last_updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model": MODEL_NAME,
        "google_search_grounding": True,
        "thinking_level": THINKING_LEVEL,
        "companies_selected": len(selected_files),
        "generation_attempts": totals.generation_attempts,
        "completed_companies": totals.completed_companies,
        "skipped_companies": totals.skipped_companies,
        "incomplete_companies": totals.incomplete_companies,
        "failed_companies": totals.failed_companies,
        "input_tokens": totals.input_tokens,
        "visible_output_tokens": totals.output_tokens,
        "thinking_tokens": totals.thinking_tokens,
        "total_tokens": totals.total_tokens,
    }

    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# GEMINI REQUEST
# ============================================================

def request_company_answers(
    client: genai.Client,
    company_name: str,
    questions: list[str],
    analysis_date: str,
    totals: UsageTotals,
) -> tuple[
    CompanyResearchResponse,
    dict[str, int],
    str,
    int,
]:
    """
    Send all 30 questions in one grounded request.

    Returns:
        parsed response
        usage information
        raw JSON response
        API attempts used for the company
    """
    prompt = build_prompt(
        company_name=company_name,
        questions=questions,
        analysis_date=analysis_date,
    )

    last_error: Exception | None = None
    attempts_for_company = 0

    for retry_number in range(MAX_RETRIES + 1):
        if (
            totals.generation_attempts
            >= MAX_GENERATION_ATTEMPTS_PER_RUN
        ):
            raise RuntimeError(
                "The run-level API-attempt safety limit "
                "has been reached."
            )

        attempts_for_company += 1
        totals.generation_attempts += 1

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,

                    # Keep reasoning cost controlled.
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_LEVEL,
                    ),

                    # Native Gemini Google Search.
                    tools=[
                        types.Tool(
                            google_search=types.GoogleSearch()
                        )
                    ],

                    # Structured output avoids fragile regex parsing.
                    response_mime_type="application/json",
                    response_json_schema=(
                        CompanyResearchResponse
                        .model_json_schema()
                    ),
                ),
            )

            raw_json = getattr(
                response,
                "text",
                None,
            )

            if not raw_json or not raw_json.strip():
                raise IncompleteResponseError(
                    "Gemini returned an empty response."
                )

            try:
                parsed_response = (
                    CompanyResearchResponse
                    .model_validate_json(raw_json)
                )
            except ValidationError as validation_error:
                raise IncompleteResponseError(
                    "Gemini returned JSON that did not match "
                    f"the required schema:\n{validation_error}"
                ) from validation_error

            parsed_response = validate_company_response(
                parsed_response=parsed_response,
                expected_count=len(questions),
            )

            usage = extract_usage(response)

            return (
                parsed_response,
                usage,
                raw_json,
                attempts_for_company,
            )

        except Exception as error:
            last_error = error

            if (
                STOP_ON_PERMANENT_QUOTA_ERROR
                and is_permanent_quota_error(error)
            ):
                raise PermanentQuotaError(
                    "The selected project or model has no usable "
                    "quota for this request. Check that billing is "
                    "active and that Google Search grounding is "
                    "available to the project.\n\n"
                    f"Original error:\n{error}"
                ) from error

            retryable = (
                is_retryable_error(error)
                or isinstance(
                    error,
                    IncompleteResponseError,
                )
            )

            if not retryable:
                raise

            if retry_number >= MAX_RETRIES:
                break

            delay = get_retry_delay(
                error=error,
                retry_number=retry_number,
            )

            print(
                f"  Attempt {attempts_for_company} failed:\n"
                f"  {error}\n"
                f"  Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Gemini failed after {attempts_for_company} attempts.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model_access(
    client: genai.Client,
) -> None:
    """
    Confirm that the configured model is visible to the API key.

    This does not consume a generation request.
    """
    try:
        model = client.models.get(
            model=MODEL_NAME
        )

        model_name = getattr(
            model,
            "name",
            MODEL_NAME,
        )

        print(f"Model available: {model_name}")

    except Exception as error:
        raise RuntimeError(
            f"The configured model '{MODEL_NAME}' could not be "
            f"accessed by this API key.\n\n{error}"
        ) from error


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    dry_run: bool = False,
    limit: int | None = None,
    start_at: int = 1,
) -> None:
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY was not found.\n\n"
            "Create a .env file containing:\n"
            "GEMINI_API_KEY=your_actual_api_key"
        )

    all_question_files = find_question_files()

    if start_at < 1:
        raise ValueError("--start-at must be 1 or greater.")

    selected_files = all_question_files[
        start_at - 1:
    ]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be 1 or greater.")

        selected_files = selected_files[:limit]

    if not selected_files:
        raise ValueError(
            "No question files were selected."
        )

    analysis_date = date.today().isoformat()

    output_directory = (
        ANSWERS_ROOT_DIR / analysis_date
    )

    failure_directory = (
        output_directory / "_failures"
    )

    raw_response_directory = (
        output_directory / "_raw_json"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_RAW_JSON_RESPONSES:
        raw_response_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    log_file = LOGS_DIR / (
        f"gemini_usage_{analysis_date}.csv"
    )

    summary_file = LOGS_DIR / (
        f"gemini_summary_{analysis_date}.json"
    )

    print("=" * 76)
    print("GEMINI GROUNDED COMPANY QUESTION PIPELINE")
    print("=" * 76)
    print(f"Model:                   {MODEL_NAME}")
    print(f"Google Search:           Enabled")
    print(f"Thinking level:          {THINKING_LEVEL}")
    print(f"Analysis date:           {analysis_date}")
    print(f"All question files:      {len(all_question_files)}")
    print(f"Files selected:          {len(selected_files)}")
    print(f"Expected questions/file: {EXPECTED_QUESTION_COUNT}")
    print(f"Max output tokens:       {MAX_OUTPUT_TOKENS:,}")
    print(f"Output directory:        {output_directory.resolve()}")
    print(f"Dry run:                 {dry_run}")
    print("=" * 76)

    print("\nSelected companies:")

    for number, question_file in enumerate(
        selected_files,
        start=start_at,
    ):
        print(
            f"{number:03d}. "
            f"{company_name_from_path(question_file)}"
        )

    if dry_run:
        print(
            "\nDry run complete. "
            "No API generation requests were made."
        )
        return

    client = genai.Client(api_key=api_key)

    validate_model_access(client)

    totals = UsageTotals()

    for run_index, question_file in enumerate(
        selected_files,
        start=1,
    ):
        company_name = company_name_from_path(
            question_file
        )

        output_file = output_directory / (
            f"{safe_filename(company_name)}"
            f"_answers_gemini.txt"
        )

        raw_json_file = raw_response_directory / (
            f"{safe_filename(company_name)}"
            f"_response.json"
        )

        print("\n" + "-" * 76)
        print(
            f"[{run_index}/{len(selected_files)}] "
            f"{company_name}"
        )

        if (
            SKIP_COMPLETED_FILES
            and existing_output_is_complete(output_file)
        ):
            totals.skipped_companies += 1

            print(
                "  Skipped: complete output already exists."
            )

            append_csv_log(
                log_file,
                {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "company": company_name,
                    "question_file": str(question_file),
                    "output_file": str(output_file),
                    "status": "skipped_complete",
                    "question_count": EXPECTED_QUESTION_COUNT,
                    "answer_count": EXPECTED_QUESTION_COUNT,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "api_attempts_for_company": 0,
                    "generation_attempt_number":
                        totals.generation_attempts,
                    "duration_seconds": 0,
                    "error": "",
                },
            )

            continue

        started_at = time.monotonic()

        questions: list[str] = []

        try:
            raw_question_text = question_file.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )

            questions = parse_questions(
                raw_question_text
            )

            print(
                f"  Questions detected: {len(questions)}"
            )

            if len(questions) != EXPECTED_QUESTION_COUNT:
                raise ValueError(
                    f"Expected exactly {EXPECTED_QUESTION_COUNT} "
                    f"questions, but detected {len(questions)} in "
                    f"{question_file.name}."
                )

            (
                parsed_result,
                usage,
                raw_json,
                attempts_for_company,
            ) = request_company_answers(
                client=client,
                company_name=company_name,
                questions=questions,
                analysis_date=analysis_date,
                totals=totals,
            )

            generated_at = datetime.now().isoformat(
                timespec="seconds"
            )

            final_output = build_final_output(
                company_name=company_name,
                analysis_date=analysis_date,
                questions=questions,
                result=parsed_result,
                usage=usage,
                generated_at=generated_at,
            )

            # Write to a temporary file first to reduce the chance of
            # leaving a partially written "complete" output.
            temporary_output = output_file.with_suffix(
                ".tmp"
            )

            temporary_output.write_text(
                final_output,
                encoding="utf-8",
            )

            temporary_output.replace(output_file)

            if SAVE_RAW_JSON_RESPONSES:
                raw_json_file.write_text(
                    raw_json,
                    encoding="utf-8",
                )

            totals.completed_companies += 1
            totals.input_tokens += usage["input_tokens"]
            totals.output_tokens += usage["output_tokens"]
            totals.thinking_tokens += usage["thinking_tokens"]
            totals.total_tokens += usage["total_tokens"]

            duration = (
                time.monotonic()
                - started_at
            )

            print(
                f"  Complete: {len(parsed_result.answers)} "
                f"answers saved."
            )

            print(
                f"  Tokens: {usage['input_tokens']:,} input, "
                f"{usage['output_tokens']:,} visible output, "
                f"{usage['thinking_tokens']:,} thinking, "
                f"{usage['total_tokens']:,} total"
            )

            print(
                f"  Duration: {duration:.1f} seconds"
            )

            print(
                f"  File: {output_file}"
            )

            append_csv_log(
                log_file,
                {
                    "timestamp": generated_at,
                    "company": company_name,
                    "question_file": str(question_file),
                    "output_file": str(output_file),
                    "status": "complete",
                    "question_count": len(questions),
                    "answer_count":
                        len(parsed_result.answers),
                    "input_tokens":
                        usage["input_tokens"],
                    "output_tokens":
                        usage["output_tokens"],
                    "thinking_tokens":
                        usage["thinking_tokens"],
                    "total_tokens":
                        usage["total_tokens"],
                    "api_attempts_for_company":
                        attempts_for_company,
                    "generation_attempt_number":
                        totals.generation_attempts,
                    "duration_seconds":
                        round(duration, 2),
                    "error": "",
                },
            )

        except PermanentQuotaError as error:
            duration = (
                time.monotonic()
                - started_at
            )

            totals.failed_companies += 1

            print("\n" + "=" * 76)
            print("PERMANENT QUOTA OR BILLING ERROR")
            print("=" * 76)
            print(error)
            print(
                "\nThe pipeline has stopped. Completed company files "
                "have been preserved. After resolving the project or "
                "billing issue, rerun the same command; complete files "
                "will be skipped."
            )

            save_failure(
                failure_directory=failure_directory,
                company_name=company_name,
                error=error,
            )

            append_csv_log(
                log_file,
                {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "company": company_name,
                    "question_file": str(question_file),
                    "output_file": str(output_file),
                    "status": "quota_failure",
                    "question_count": len(questions),
                    "answer_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "api_attempts_for_company": 0,
                    "generation_attempt_number":
                        totals.generation_attempts,
                    "duration_seconds":
                        round(duration, 2),
                    "error": str(error),
                },
            )

            save_run_summary(
                summary_file=summary_file,
                selected_files=selected_files,
                totals=totals,
            )

            break

        except IncompleteResponseError as error:
            duration = (
                time.monotonic()
                - started_at
            )

            totals.incomplete_companies += 1

            print(
                f"  Incomplete response: {error}"
            )

            save_failure(
                failure_directory=failure_directory,
                company_name=company_name,
                error=error,
            )

            append_csv_log(
                log_file,
                {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "company": company_name,
                    "question_file": str(question_file),
                    "output_file": str(output_file),
                    "status": "incomplete",
                    "question_count": len(questions),
                    "answer_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "api_attempts_for_company": 0,
                    "generation_attempt_number":
                        totals.generation_attempts,
                    "duration_seconds":
                        round(duration, 2),
                    "error": str(error),
                },
            )

        except Exception as error:
            duration = (
                time.monotonic()
                - started_at
            )

            totals.failed_companies += 1

            print(f"  Failed: {error}")

            save_failure(
                failure_directory=failure_directory,
                company_name=company_name,
                error=error,
            )

            append_csv_log(
                log_file,
                {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "company": company_name,
                    "question_file": str(question_file),
                    "output_file": str(output_file),
                    "status": "failed",
                    "question_count": len(questions),
                    "answer_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "api_attempts_for_company": 0,
                    "generation_attempt_number":
                        totals.generation_attempts,
                    "duration_seconds":
                        round(duration, 2),
                    "error": str(error),
                },
            )

        save_run_summary(
            summary_file=summary_file,
            selected_files=selected_files,
            totals=totals,
        )

        if run_index < len(selected_files):
            print(
                f"  Waiting "
                f"{SECONDS_BETWEEN_COMPANIES} seconds..."
            )

            time.sleep(
                SECONDS_BETWEEN_COMPANIES
            )

    save_run_summary(
        summary_file=summary_file,
        selected_files=selected_files,
        totals=totals,
    )

    print("\n" + "=" * 76)
    print("FINAL RUN SUMMARY")
    print("=" * 76)
    print(
        f"Companies selected:       "
        f"{len(selected_files)}"
    )
    print(
        f"Completed:                "
        f"{totals.completed_companies}"
    )
    print(
        f"Skipped as complete:      "
        f"{totals.skipped_companies}"
    )
    print(
        f"Incomplete:               "
        f"{totals.incomplete_companies}"
    )
    print(
        f"Failed:                   "
        f"{totals.failed_companies}"
    )
    print(
        f"Generation attempts:      "
        f"{totals.generation_attempts}"
    )
    print(
        f"Input tokens:             "
        f"{totals.input_tokens:,}"
    )
    print(
        f"Visible output tokens:    "
        f"{totals.output_tokens:,}"
    )
    print(
        f"Thinking tokens:          "
        f"{totals.thinking_tokens:,}"
    )
    print(
        f"Total tokens:             "
        f"{totals.total_tokens:,}"
    )
    print(
        f"Answer directory:         "
        f"{output_directory.resolve()}"
    )
    print(
        f"Usage log:                "
        f"{log_file.resolve()}"
    )
    print(
        f"Summary file:             "
        f"{summary_file.resolve()}"
    )
    print("=" * 76)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer all company question files using "
            "Gemini 3.1 Flash-Lite with native Google Search."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Display selected companies without making "
            "generation requests."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N selected companies. "
            "Useful for pilot testing."
        ),
    )

    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help=(
            "Start from this 1-based position in the sorted "
            "question-file list."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_pipeline(
        dry_run=arguments.dry_run,
        limit=arguments.limit,
        start_at=arguments.start_at,
    )