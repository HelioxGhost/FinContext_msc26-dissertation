from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_DIR = Path("questions")

# Output files:
ANSWERS_ROOT_DIR = Path("answers_gemini")

LOGS_DIR = Path("gemini_logs")

MODEL_NAME = "gemini-2.5-flash"

# Every question file should contain exactly 30 questions.
EXPECTED_QUESTION_COUNT = 30

# Maximum input prompt size for one company.
MAX_INPUT_TOKENS = 10_000

# Maximum output size for all 30 answers from one company.
MAX_OUTPUT_TOKENS = 12_000

MAX_GENERATION_REQUESTS_PER_RUN = 150

# Total attempts includes failed attempts and retries.
MAX_TOTAL_API_ATTEMPTS_PER_RUN = 150

# Wait between companies to reduce requests-per-minute errors.
SECONDS_BETWEEN_COMPANIES = 7

# Number of retries after the first failed attempt.
MAX_RETRIES = 3

# Low temperature produces more consistent daily answers.
TEMPERATURE = 0.2

# Skip a company when its existing output appears complete.
SKIP_COMPLETED_FILES = True

# Number of answer headings required for a complete output.
MINIMUM_ACCEPTABLE_ANSWERS = 30


# ============================================================
# RUNNING TOTALS
# ============================================================

@dataclass
class UsageTotals:
    generation_requests: int = 0
    total_api_attempts: int = 0
    successful_companies: int = 0
    skipped_companies: int = 0
    incomplete_companies: int = 0
    failed_companies: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# ============================================================
# FILE UTILITIES
# ============================================================

def safe_filename(name: str) -> str:
   
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.rstrip(". ")


def natural_sort_key(path: Path) -> list[Any]:
   
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def company_name_from_path(question_file: Path) -> str:
   
    return question_file.stem.strip()


def find_question_files() -> list[Path]:
    
    if not QUESTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Questions directory not found:\n"
            f"{QUESTIONS_DIR.resolve()}"
        )

    question_files = sorted(
        QUESTIONS_DIR.glob("*.txt"),
        key=natural_sort_key,
    )

    if not question_files:
        raise FileNotFoundError(
            f"No .txt files were found in:\n"
            f"{QUESTIONS_DIR.resolve()}"
        )

    return question_files


# ============================================================
# QUESTION PARSING
# ============================================================

def parse_questions(question_text: str) -> list[str]:
    
    text = (
        question_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if not text:
        return []

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
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )

    matches = list(numbered_question_pattern.finditer(text))

    questions: list[str] = []

    if matches:
        for index, match in enumerate(matches):
            question_start = match.end()

            if index + 1 < len(matches):
                question_end = matches[index + 1].start()
            else:
                question_end = len(text)

            question = text[question_start:question_end].strip()

            # Remove markdown formatting around the question.
            question = question.strip("* \t\n")

            # Join multi-line questions into one complete line.
            question = re.sub(r"\s+", " ", question).strip()

            if question:
                questions.append(question)

        return questions

    # Fallback: questions separated by blank lines.
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if len(paragraphs) > 1:
        return paragraphs

    # Final fallback: one question per non-empty line.
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]


def format_numbered_questions(questions: list[str]) -> str:
   
    return "\n\n".join(
        f"QUESTION {number}\n{question}"
        for number, question in enumerate(questions, start=1)
    )


# ============================================================
# PROMPT
# ============================================================

SYSTEM_INSTRUCTION = """
You are a financial research assistant conducting a repeated daily study of
publicly traded companies.

Your task is to answer every supplied question using current and verifiable
public web information available on or before the analysis date.

Use Google Search grounding whenever current information is required.

Follow these rules:

1. Answer every supplied question.
2. Answer the questions in their original order.
3. Repeat the complete original question before each answer.
4. Never shorten, rename, merge, omit or reorder a question.
5. Use company-specific evidence rather than generic financial commentary.
6. Prioritise recent information that may affect future stock performance.
7. Use older information only when it provides necessary context.
8. Prefer authoritative sources, including:
   - company investor-relations pages;
   - regulatory filings;
   - official company announcements;
   - earnings releases;
   - earnings-call transcripts;
   - regulators and recognised exchanges;
   - reliable financial news organisations.
9. Clearly distinguish confirmed facts from analysis or interpretation.
10. Do not invent numbers, events, dates, quotations or sources.
11. When reliable current evidence is unavailable, say:
    "Insufficient current evidence was found."
12. Include relevant dates for recent developments.
13. Keep each answer concise but analytically useful.
14. Label the outlook for each answer as:
    POSITIVE, NEGATIVE, MIXED, NEUTRAL or INSUFFICIENT EVIDENCE.
15. Do not provide personalised investment advice.
16. Do not tell the reader to buy, sell or hold the stock.
17. Return plain text only.
""".strip()


def build_prompt(
    company_name: str,
    questions: list[str],
    analysis_date: str,
) -> str:
    """
    Build one prompt containing all 30 questions for one company.
    """
    question_block = format_numbered_questions(questions)

    return f"""
COMPANY: {company_name}
ANALYSIS DATE: {analysis_date}
TOTAL QUESTIONS: {len(questions)}

TASK

Use current Google Search-grounded information to answer all questions below
about {company_name}.

This response is part of a repeated daily stock-performance research
experiment. Answers may be compared against answers generated on future dates.

Only use information that was publicly available on or before
{analysis_date}.

Prioritise information that may influence the company's future business
performance, investor sentiment, risk profile or stock performance.

You must answer all {len(questions)} questions. Do not stop after answering
only part of the list.

REQUIRED FORMAT

COMPANY: {company_name}
ANALYSIS DATE: {analysis_date}

QUESTION 1
[Repeat the complete original question exactly.]

ANSWER
[Provide a concise, evidence-based answer of approximately 80-180 words.]

OUTLOOK
[POSITIVE, NEGATIVE, MIXED, NEUTRAL or INSUFFICIENT EVIDENCE]

Continue using exactly the same structure through QUESTION {len(questions)}.

After answering every question, include:

OVERALL DAILY SUMMARY
[Summarise the most material current factors affecting the company.]

MOST IMPORTANT CURRENT SIGNALS
1. [Signal]
2. [Signal]
3. [Signal]
4. [Signal]
5. [Signal]

QUESTIONS

{question_block}
""".strip()


# ============================================================
# RESPONSE AND TOKEN UTILITIES
# ============================================================

def get_usage_value(
    usage_metadata: Any,
    *possible_attribute_names: str,
) -> int:
    
    if usage_metadata is None:
        return 0

    for attribute_name in possible_attribute_names:
        value = getattr(
            usage_metadata,
            attribute_name,
            None,
        )

        if isinstance(value, int):
            return value

    return 0


def extract_usage(response: Any) -> dict[str, int]:
    
    usage_metadata = getattr(
        response,
        "usage_metadata",
        None,
    )

    input_tokens = get_usage_value(
        usage_metadata,
        "prompt_token_count",
        "input_token_count",
        "total_input_tokens",
    )

    output_tokens = get_usage_value(
        usage_metadata,
        "candidates_token_count",
        "output_token_count",
        "total_output_tokens",
    )

    total_tokens = get_usage_value(
        usage_metadata,
        "total_token_count",
        "total_tokens",
    )

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def count_answer_sections(answer_text: str) -> int:
    
    matches = re.findall(
        r"(?im)^\s*QUESTION\s+(\d{1,3})\s*$",
        answer_text,
    )

    return len(set(matches))


def response_is_complete(answer_text: str) -> bool:
    
    answer_count = count_answer_sections(answer_text)

    has_summary = (
        "OVERALL DAILY SUMMARY" in answer_text.upper()
    )

    return (
        answer_count >= MINIMUM_ACCEPTABLE_ANSWERS
        and has_summary
    )


def existing_output_is_complete(output_file: Path) -> bool:
   
    if not output_file.exists():
        return False

    try:
        existing_text = output_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return response_is_complete(existing_text)

    except OSError:
        return False


# ============================================================
# ERROR HANDLING
# ============================================================

def is_retryable_error(error: Exception) -> bool:
  
    error_text = str(error).lower()

    retryable_terms = (
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
    )

    return any(
        term in error_text
        for term in retryable_terms
    )


def save_failure(
    failure_directory: Path,
    company_name: str,
    error: Exception | str,
) -> None:
    """
    Save failure details without stopping the entire experiment.
    """
    failure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_file = failure_directory / (
        f"{safe_filename(company_name)}_gemini_failure.txt"
    )

    failure_file.write_text(
        f"Company: {company_name}\n"
        f"Time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Error: {error}\n",
        encoding="utf-8",
    )


# ============================================================
# LOGGING
# ============================================================

def append_csv_log(
    log_file: Path,
    row: dict[str, Any],
) -> None:
    
    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "timestamp",
        "company",
        "question_file",
        "output_file",
        "status",
        "question_count",
        "answer_sections_detected",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "generation_request_number",
        "api_attempts_for_company",
        "error",
    ]

    file_already_exists = log_file.exists()

    with log_file.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )

        if not file_already_exists:
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
        "companies_selected": len(selected_files),
        "generation_requests": totals.generation_requests,
        "total_api_attempts": totals.total_api_attempts,
        "successful_companies": totals.successful_companies,
        "skipped_companies": totals.skipped_companies,
        "incomplete_companies": totals.incomplete_companies,
        "failed_companies": totals.failed_companies,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
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

def answer_company(
    client: genai.Client,
    company_name: str,
    questions: list[str],
    analysis_date: str,
    totals: UsageTotals,
) -> tuple[str, dict[str, int], int]:
    
    prompt = build_prompt(
        company_name=company_name,
        questions=questions,
        analysis_date=analysis_date,
    )

    # Count the prompt before generating the answer.
    token_count_response = client.models.count_tokens(
        model=MODEL_NAME,
        contents=prompt,
    )

    estimated_input_tokens = getattr(
        token_count_response,
        "total_tokens",
        0,
    )

    print(
        f"  Input tokens: "
        f"{estimated_input_tokens:,}"
    )

    if estimated_input_tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"The prompt contains approximately "
            f"{estimated_input_tokens:,} input tokens. "
            f"The configured maximum is "
            f"{MAX_INPUT_TOKENS:,}."
        )

    attempts_for_company = 0
    last_error: Exception | None = None

    for retry_number in range(MAX_RETRIES + 1):
        if (
            totals.generation_requests
            >= MAX_GENERATION_REQUESTS_PER_RUN
        ):
            raise RuntimeError(
                "The maximum grounded generation request "
                "limit for this run has been reached."
            )

        if (
            totals.total_api_attempts
            >= MAX_TOTAL_API_ATTEMPTS_PER_RUN
        ):
            raise RuntimeError(
                "The maximum total API-attempt limit "
                "for this run has been reached."
            )

        attempts_for_company += 1
        totals.total_api_attempts += 1
        totals.generation_requests += 1

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="text/plain",

                    # Enable Google Search grounding.
                    tools=[
                        types.Tool(
                            google_search=types.GoogleSearch()
                        )
                    ],
                ),
            )

            answer_text = getattr(
                response,
                "text",
                None,
            )

            if not answer_text or not answer_text.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            usage = extract_usage(response)

            # Use the count_tokens result when response metadata
            # does not contain an input-token count.
            if usage["input_tokens"] == 0:
                usage["input_tokens"] = (
                    estimated_input_tokens
                )

            if usage["total_tokens"] == 0:
                usage["total_tokens"] = (
                    usage["input_tokens"]
                    + usage["output_tokens"]
                )

            return (
                answer_text.strip(),
                usage,
                attempts_for_company,
            )

        except Exception as error:
            last_error = error

            if not is_retryable_error(error):
                raise

            if retry_number >= MAX_RETRIES:
                break

            delay = min(
                120,
                15 * (2 ** retry_number)
                + random.uniform(0, 5),
            )

            print(
                f"  Temporary Gemini error:\n"
                f"  {error}\n"
                f"  Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Gemini failed after "
        f"{attempts_for_company} attempts. "
        f"Last error: {last_error}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(dry_run: bool = False) -> None:
    
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY was not found.\n\n"
            "Create a .env file in the project directory:\n\n"
            "GEMINI_API_KEY=your_api_key_here"
        )

    selected_files = find_question_files()

    current_date = datetime.now().date().isoformat()

    output_directory = (
        ANSWERS_ROOT_DIR / current_date
    )

    failure_directory = (
        output_directory / "_failures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = LOGS_DIR / (
        f"gemini_usage_{current_date}.csv"
    )

    summary_file = LOGS_DIR / (
        f"gemini_summary_{current_date}.json"
    )

    print("=" * 72)
    print("GEMINI COMPANY QUESTION PIPELINE")
    print("=" * 72)
    print(f"Model:                  {MODEL_NAME}")
    print(f"Analysis date:          {current_date}")
    print(f"Companies detected:     {len(selected_files)}")
    print(f"Questions expected:     {EXPECTED_QUESTION_COUNT}")
    print(f"Max input tokens:       {MAX_INPUT_TOKENS:,}")
    print(f"Max output tokens:      {MAX_OUTPUT_TOKENS:,}")
    print(
        f"Max generation calls:   "
        f"{MAX_GENERATION_REQUESTS_PER_RUN}"
    )
    print(f"Output directory:       {output_directory.resolve()}")
    print(f"Dry run:                {dry_run}")
    print("=" * 72)

    print("\nCompanies selected:")

    for number, question_file in enumerate(
        selected_files,
        start=1,
    ):
        company_name = company_name_from_path(
            question_file
        )

        print(
            f"{number:03d}. {company_name}"
        )

    if dry_run:
        print(
            "\nDry run complete. "
            "No Gemini API requests were made."
        )
        return

    client = genai.Client(api_key=api_key)
    totals = UsageTotals()

    for company_index, question_file in enumerate(
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

        print("\n" + "-" * 72)
        print(
            f"[{company_index}/{len(selected_files)}] "
            f"{company_name}"
        )

        # Skip companies already completed successfully.
        if (
            SKIP_COMPLETED_FILES
            and existing_output_is_complete(output_file)
        ):
            print(
                "  Skipped: a complete output file "
                "already exists."
            )

            totals.skipped_companies += 1

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
                    "answer_sections_detected":
                        EXPECTED_QUESTION_COUNT,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "generation_request_number":
                        totals.generation_requests,
                    "api_attempts_for_company": 0,
                    "error": "",
                },
            )

            continue

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
                    f"Expected exactly "
                    f"{EXPECTED_QUESTION_COUNT} questions, "
                    f"but detected {len(questions)} in "
                    f"{question_file.name}."
                )

            (
                answer_text,
                usage,
                attempts_for_company,
            ) = answer_company(
                client=client,
                company_name=company_name,
                questions=questions,
                analysis_date=current_date,
                totals=totals,
            )

            detected_answer_sections = (
                count_answer_sections(answer_text)
            )

            complete = response_is_complete(
                answer_text
            )

            pipeline_status = (
                "COMPLETE"
                if complete
                else "INCOMPLETE"
            )

            output_contents = (
                f"PIPELINE STATUS: {pipeline_status}\n"
                f"MODEL: {MODEL_NAME}\n"
                f"GENERATED AT: "
                f"{datetime.now().isoformat(timespec='seconds')}\n"
                f"QUESTIONS EXPECTED: "
                f"{EXPECTED_QUESTION_COUNT}\n"
                f"ANSWER SECTIONS DETECTED: "
                f"{detected_answer_sections}\n"
                f"INPUT TOKENS: "
                f"{usage['input_tokens']}\n"
                f"OUTPUT TOKENS: "
                f"{usage['output_tokens']}\n"
                f"TOTAL TOKENS: "
                f"{usage['total_tokens']}\n"
                f"{'=' * 72}\n\n"
                f"{answer_text}\n"
            )

            output_file.write_text(
                output_contents,
                encoding="utf-8",
            )

            totals.input_tokens += (
                usage["input_tokens"]
            )

            totals.output_tokens += (
                usage["output_tokens"]
            )

            totals.total_tokens += (
                usage["total_tokens"]
            )

            if complete:
                totals.successful_companies += 1
                log_status = "complete"

                print(
                    f"  Complete output saved:\n"
                    f"  {output_file}"
                )

            else:
                totals.incomplete_companies += 1
                log_status = "incomplete"

                print(
                    f"  Warning: Gemini returned only "
                    f"{detected_answer_sections} detectable "
                    f"answer sections."
                )

                print(
                    "  The response was saved as incomplete. "
                    "It will be attempted again when the "
                    "script is rerun."
                )

            print(
                f"  Token usage: "
                f"{usage['input_tokens']:,} input, "
                f"{usage['output_tokens']:,} output, "
                f"{usage['total_tokens']:,} total"
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
                    "status": log_status,
                    "question_count": len(questions),
                    "answer_sections_detected":
                        detected_answer_sections,
                    "input_tokens":
                        usage["input_tokens"],
                    "output_tokens":
                        usage["output_tokens"],
                    "total_tokens":
                        usage["total_tokens"],
                    "generation_request_number":
                        totals.generation_requests,
                    "api_attempts_for_company":
                        attempts_for_company,
                    "error": "",
                },
            )

        except Exception as error:
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
                    "question_count": 0,
                    "answer_sections_detected": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "generation_request_number":
                        totals.generation_requests,
                    "api_attempts_for_company": 0,
                    "error": str(error),
                },
            )

        # Save the summary after every company in case the
        # script is interrupted.
        save_run_summary(
            summary_file=summary_file,
            selected_files=selected_files,
            totals=totals,
        )

        # Wait unless this was the final company.
        if company_index < len(selected_files):
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

    print("\n" + "=" * 72)
    print("FINAL RUN SUMMARY")
    print("=" * 72)
    print(
        f"Companies selected:     "
        f"{len(selected_files)}"
    )
    print(
        f"Successfully completed: "
        f"{totals.successful_companies}"
    )
    print(
        f"Already complete:       "
        f"{totals.skipped_companies}"
    )
    print(
        f"Incomplete responses:   "
        f"{totals.incomplete_companies}"
    )
    print(
        f"Failed companies:       "
        f"{totals.failed_companies}"
    )
    print(
        f"Generation requests:    "
        f"{totals.generation_requests}"
    )
    print(
        f"Total API attempts:     "
        f"{totals.total_api_attempts}"
    )
    print(
        f"Input tokens:           "
        f"{totals.input_tokens:,}"
    )
    print(
        f"Output tokens:          "
        f"{totals.output_tokens:,}"
    )
    print(
        f"Total tokens:           "
        f"{totals.total_tokens:,}"
    )
    print(
        f"Answer directory:       "
        f"{output_directory.resolve()}"
    )
    print(
        f"Usage log:              "
        f"{log_file.resolve()}"
    )
    print(
        f"Summary file:           "
        f"{summary_file.resolve()}"
    )
    print("=" * 72)


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer all company question files using "
            "Gemini 2.5 Flash and Google Search grounding."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show all detected companies without "
            "making Gemini API requests."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_pipeline(
        dry_run=arguments.dry_run,
    )