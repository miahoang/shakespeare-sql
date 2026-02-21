"""
Original author prompt templates for Ambrosia RAG-based SQL evaluation.

This module uses a simple prompt format inspired by the "Disambiguate First, Parse Later"
paper authors, adapted for Ambrosia's interpretation-based format.

The approach emphasizes:
- Learning from examples through in-context learning
- Minimal instruction overhead
- Letting the model infer patterns from examples
- Questions can be ambiguous (multiple interpretations) or unambiguous (single interpretation)
"""

from typing import Dict


# Single system message for all questions (matching AmbiQT authors' style)
SYSTEM_MESSAGE_SQL_GENERATION = """You are an expert SQL query generator for ambiguous questions.

Your task: Write SQL queries based on the provided question. Questions can be ambiguous, meaning they can be interpreted in different ways. In such cases, write all possible SQL queries corresponding to different interpretations.

Return your response in JSON format:
{
  "sql_queries": ["SELECT ...", "SELECT ..."]
}

Requirements:
- **DO NOT SELECT EXTRA COLUMNS** beyond those requested in the question
- Do not include any explanations or interpretations
- Review the actual database schema to identify all possible interpretations (e.g., column synonyms, table synonyms)
- Return valid JSON only - start with { and end with }
- In JSON strings, escape backslashes as \\\\ (double backslash)
- In JSON strings, escape double quotes as \\"
- Use double quotes for SQL string literals (e.g., WHERE name = "John")
- No markdown code blocks, no extra explanations
- Make sure your JSON response is valid before returning it"""


# Single user template (matching AmbiQT authors' ICL style)
USER_PROMPT_SQL_GENERATION_TEMPLATE = """Some example questions and their SQL queries are provided below based on similar problems:

{examples_text}

Given the following SQLite database schema:
{db_schema}

Answer the following:
{question}

Return JSON format:
{{
  "sql_queries": [
    "query1",
    "query2"
  ]
}}"""


def create_sql_generation_prompt(
    question: str,
    db_schema: str,
    examples_text: str,
    category: str = "AA",
    retry_guidance: str = None,
    previous_errors: list = None,
    schema_ambiguities: str = None,
    predicted_ambig_type: str = None
) -> Dict[str, str]:
    """
    Create prompts using the original authors' simple ICL format for Ambrosia.

    Args:
        question: The question (ambiguous or unambiguous)
        db_schema: Database schema
        examples_text: Formatted RAG examples showing interpretations
        category: Question category ("AU" or "AA") - kept for API compatibility but not used
        retry_guidance: Optional guidance from previous failure analysis (for retries only)
        previous_errors: Optional list of previous SQL execution errors (for retries only)
        schema_ambiguities: Optional analysis (kept for API compatibility but not used)
        predicted_ambig_type: Optional predicted ambiguity type from hybrid classifier
                              (e.g., "attachment", "scope", "vague")

    Returns:
        Dictionary with 'system' and 'user' keys containing the prompts
    """
    # Start with base system message
    system_message = SYSTEM_MESSAGE_SQL_GENERATION

    user_prompt = USER_PROMPT_SQL_GENERATION_TEMPLATE.format(
        db_schema=db_schema,
        question=question,
        examples_text=examples_text
    )

    # Add retry context ONLY if there are execution errors
    if retry_guidance or previous_errors:
        retry_context = "\n\n**RETRY CONTEXT - Previous Attempt Failed:**\n"

        if previous_errors:
            retry_context += "\n**Previous SQL Execution Errors:**\n"
            for i, error in enumerate(previous_errors[:5], 1):
                retry_context += f"{i}. {error}\n"

        if retry_guidance:
            retry_context += f"\n**Failure Analysis and Guidance:**\n{retry_guidance}\n"

        retry_context += "\n**IMPORTANT:** Use the schema and error information above to avoid making the same mistakes. Generate corrected SQL queries.\n"

        user_prompt = retry_context + user_prompt

    return {
        'system': system_message,
        'user': user_prompt
    }

SYSTEM_MESSAGE_SQL_CORRECTION = """You are an expert SQL validator for SQLite.

You will be given a question, a database schema, and a list of SQL queries that were generated to answer the question.

Your task: Decide whether the queries correctly answer the question and whether they contain any syntax errors, then fix any issues you have identified.

Rules:
- Review the question and database schema to understand the intent
- Identify SQLite syntax errors (missing commas, incorrect keywords, etc.)
- Identify selection errors (wrong/missing/extra columns, incorrect table references)
- Check whether the query sorts, filters, and/or aggregates data as intended by the question
- If there are any errors, correct them to ensure the query runs successfully and returns the data the question is asking for

CRITICAL JSON FORMATTING RULES:
- Return valid JSON only - start with { and end with }
- In JSON strings, escape backslashes as \\\\ (double backslash)
- In JSON strings, escape double quotes as \\"
- Use double quotes for SQL string literals (e.g., WHERE name = "John")
- No markdown code blocks, no extra explanations

Return format:
{
  "corrected_queries": ["SELECT ...", "SELECT ..."]
}"""


USER_PROMPT_SQL_CORRECTION_TEMPLATE = """Some example questions and their SQL queries are provided below based on similar problems:
{examples_text}

Given the following SQLite database schema:
{db_schema}

Question: 
{question}

Interpretations and queries to review:
{queries_text}

Review the question, interpretations and queries thoroughly, then fix any issues you can identify.

IMPORTANT: When reviewing queries and removing dupicates, only remove **EXACT** duplicates (identical SQL text).

Return JSON only:
{{
  "corrected_queries": [
    "corrected_query1",
    "corrected_query2"
  ]
}}"""


def create_sql_correction_prompt(
    question: str,
    db_schema: str,
    sql_queries: list,
    execution_errors: list = None,
    examples_text: str = ""
) -> Dict[str, str]:
    """
    Create prompts for SQL correction.

    Args:
        question: Original question
        db_schema: Database schema
        sql_queries: List of SQL query strings
        execution_errors: Optional list of execution error messages
        examples_text: Formatted RAG examples showing correct SQL queries

    Returns:
        Dictionary with 'system' and 'user' keys containing the prompts
    """
    # Format queries for the prompt
    queries_text = "\n\n".join([
        f"Query {i+1}:\n{q}"
        for i, q in enumerate(sql_queries)
    ])

    # Use empty examples message if not provided
    if not examples_text:
        examples_text = "No examples provided."

    user_prompt = USER_PROMPT_SQL_CORRECTION_TEMPLATE.format(
        examples_text=examples_text,
        db_schema=db_schema,
        question=question,
        queries_text=queries_text
    )

    # Add execution errors if provided
    if execution_errors:
        errors_text = "\n\nExecution Errors:\n" + "\n".join([
            f"- {err}" for err in execution_errors if err
        ])
        user_prompt = user_prompt.replace(
            "Review the question, interpretations and queries thoroughly",
            f"The following execution errors occurred:{errors_text}\n\nReview the question, interpretations and queries thoroughly"
        )

    return {
        'system': SYSTEM_MESSAGE_SQL_CORRECTION,
        'user': user_prompt
    }
