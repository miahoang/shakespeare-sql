"""
Utilities for parsing SQL queries from LLM responses.

This module provides robust parsing of SQL queries, supporting both
JSON and fallback text-based parsing.
"""

import json
import re
import logging
from typing import List, Tuple, Optional

from rag.evaluation_config import (
    SQL_CODE_FENCE,
    SQL_CODE_FENCE_END,
    SQL_START_KEYWORDS,
    AMBIGUITY_MARKERS,
    INTERPRETATION_PATTERN
)

logger = logging.getLogger(__name__)


def fix_escape_sequences(text: str) -> str:
    """
    Fix common invalid escape sequences in JSON strings.

    Args:
        text: JSON text with potentially invalid escapes

    Returns:
        Text with fixed escape sequences
    """
    # This regex finds strings in JSON (between quotes) and fixes invalid escapes
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # Anything else like \d, \s, \w, etc should have the backslash escaped

    def fix_string_escapes(match):
        string_content = match.group(1)

        # First pass: Fix invalid escapes (anything not followed by valid escape chars)
        # Valid: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        result = re.sub(
            r'\\(?!["\\/bfnrtu])',
            r'\\\\',
            string_content
        )

        # Second pass: Fix unescaped backslashes at end of strings
        result = re.sub(r'\\$', r'\\\\', result)

        return f'"{result}"'

    # Match quoted strings (non-greedy, handles escaped quotes)
    # This pattern properly handles escaped quotes within strings
    pattern = r'"((?:[^"\\]|\\.)*)"'
    try:
        fixed = re.sub(pattern, fix_string_escapes, text)
        logger.debug("Applied escape sequence fixes")
        return fixed
    except Exception as e:
        logger.debug(f"Error fixing escape sequences: {e}")
        return text


def clean_json_response(text: str) -> str:
    """
    Clean response text to extract JSON content.

    Args:
        text: Raw response text

    Returns:
        Cleaned text containing only JSON
    """
    # Remove markdown code fences
    text = text.replace('```json', '').replace('```', '').strip()

    # Try to find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]

    return text


def parse_json_response(text: str) -> Optional[dict]:
    """
    Parse JSON response from LLM.

    Args:
        text: Response text from LLM

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # First, always clean the text to remove markdown fences
    text = text.strip()

    try:
        # Try direct parsing
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e}")

        # If error is about invalid escape, try fixing escapes FIRST
        if "Invalid" in str(e) and "escape" in str(e):
            logger.debug("Attempting to fix escape sequences...")
            try:
                fixed = fix_escape_sequences(text)
                return json.loads(fixed)
            except json.JSONDecodeError as e2:
                logger.debug(f"Still failed after fixing escapes: {e2}")

        # If error is "Extra data", try to parse just the first JSON object
        if "Extra data" in str(e):
            try:
                # Use JSONDecoder to get just the first JSON object
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(text)
                logger.debug(f"Parsed JSON with extra text after position {idx}")
                return obj
            except json.JSONDecodeError:
                pass

        # Try cleaning the response more aggressively
        logger.debug("Attempting aggressive cleaning...")
        try:
            cleaned = clean_json_response(text)

            # Try fixing escape sequences on cleaned text FIRST
            try:
                fixed_cleaned = fix_escape_sequences(cleaned)
                result = json.loads(fixed_cleaned)
                logger.debug("Successfully parsed after cleaning and fixing escapes")
                return result
            except json.JSONDecodeError:
                pass

            # Try again with just cleaned text
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e2:
                # Try raw_decode on cleaned text too
                if "Extra data" in str(e2):
                    decoder = json.JSONDecoder()
                    obj, idx = decoder.raw_decode(cleaned)
                    logger.debug(f"Parsed cleaned JSON with extra text after position {idx}")
                    return obj
                raise
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text (first 200 chars): {text[:200]}...")
            return None


def extract_sql_from_json(json_data: dict) -> Tuple[List[str], Optional[str]]:
    """
    Extract SQL queries and ambiguity analysis from parsed JSON.

    Supports multiple JSON formats:
    - {"sql_queries": ["SELECT ...", "SELECT ..."]}
    - {"corrected_queries": ["SELECT ...", "SELECT ..."]}
    - {"interpretations": [{"interpretation": "...", "sql_query": "SELECT ..."}]}

    Args:
        json_data: Parsed JSON dictionary

    Returns:
        Tuple of (sql_queries, ambiguity_analysis)
    """
    sql_queries = []
    ambiguity_analysis = None

    # Extract ambiguity analysis
    if 'ambiguity_analysis' in json_data:
        ambiguity_analysis = json_data['ambiguity_analysis']

    # Handle 'interpretations' format (Ambrosia)
    if 'interpretations' in json_data:
        interpretations = json_data['interpretations']
        if isinstance(interpretations, list):
            for interp in interpretations:
                if isinstance(interp, dict) and 'sql_query' in interp:
                    query = interp['sql_query'].strip()
                    if query:
                        sql_queries.append(query)
        else:
            logger.warning(f"Expected interpretations to be a list, got {type(interpretations)}")

    # Handle 'sql_queries' format (direct list)
    elif 'sql_queries' in json_data:
        queries = json_data['sql_queries']
        if isinstance(queries, list):
            sql_queries = [q.strip() for q in queries if q and q.strip()]
        else:
            logger.warning(f"Expected sql_queries to be a list, got {type(queries)}")

    # Handle 'corrected_queries' format (correction responses)
    elif 'corrected_queries' in json_data:
        queries = json_data['corrected_queries']
        if isinstance(queries, list):
            sql_queries = [q.strip() for q in queries if q and q.strip()]
        else:
            logger.warning(f"Expected corrected_queries to be a list, got {type(queries)}")

    return sql_queries, ambiguity_analysis


def remove_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from text.

    Args:
        text: Text potentially containing markdown fences

    Returns:
        Text with fences removed
    """
    text = text.replace(SQL_CODE_FENCE, '').replace(SQL_CODE_FENCE_END, '')
    return text.strip()


def extract_ambiguity_analysis_from_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract ambiguity analysis from text-based response.

    Args:
        text: Response text

    Returns:
        Tuple of (remaining_text, ambiguity_analysis)
    """
    ambiguity_analysis = None
    lines = text.split('\n')

    for i, line in enumerate(lines):
        # Check if line starts with ambiguity marker
        line_stripped = line.strip()

        # Remove markdown bold markers first
        line_cleaned = line_stripped.replace('**', '')

        # Check for markers
        for marker in AMBIGUITY_MARKERS:
            marker_cleaned = marker.replace('**', '')
            if line_cleaned.startswith(marker_cleaned):
                # Extract the analysis text (everything after the marker)
                ambiguity_analysis = line_cleaned.replace(marker_cleaned, '', 1).strip()
                # Remove leading colon if present
                if ambiguity_analysis.startswith(':'):
                    ambiguity_analysis = ambiguity_analysis[1:].strip()
                # Remove the analysis line from the text
                remaining_text = '\n'.join(lines[i+1:])
                return remaining_text, ambiguity_analysis

    return text, ambiguity_analysis


def parse_sql_from_text(text: str) -> List[str]:
    """
    Parse SQL queries from text-based response (fallback method).

    Supports multiple formats:
    1. Queries separated by interpretation markers (-- Interpretation N:)
    2. Queries separated by double newlines
    3. Queries in markdown code blocks

    Args:
        text: Response text

    Returns:
        List of SQL queries
    """
    sql_queries = []

    # Remove markdown code fences
    text = remove_markdown_fences(text)

    # Try to split by interpretation/query markers
    parts = re.split(INTERPRETATION_PATTERN, text, flags=re.IGNORECASE)

    if len(parts) > 2:  # Found interpretation markers
        # parts will be like: [text_before, 'Interpretation', query1, 'Interpretation', query2, ...]
        for i in range(2, len(parts), 2):
            if i < len(parts):
                query_text = parts[i].strip()
                query = extract_sql_from_section(query_text)
                if query:
                    sql_queries.append(query)
    else:
        # No explicit markers, fall back to double newline splitting
        sections = text.split('\n\n')

        for section in sections:
            section = section.strip()

            # Skip empty sections or standalone comments
            if not section or (section.startswith('--') and '\n' not in section):
                continue

            # Skip divider lines
            if section.startswith('---'):
                continue

            query = extract_sql_from_section(section)
            if query:
                sql_queries.append(query)

    return sql_queries


def extract_sql_from_section(section: str) -> Optional[str]:
    """
    Extract SQL query from a text section.

    Args:
        section: Text section potentially containing SQL

    Returns:
        SQL query string or None if no SQL found
    """
    # Clean up inline comments at the start
    lines = section.split('\n')
    sql_lines = []
    sql_started = False

    for line in lines:
        line_stripped = line.strip()

        # Check if this line contains SQL keywords (start of actual SQL)
        if any(keyword in line_stripped.upper() for keyword in SQL_START_KEYWORDS):
            sql_started = True

        # Once SQL has started, collect lines
        if sql_started and line_stripped:
            # Skip comment-only lines
            if not (line_stripped.startswith('--') and
                    not any(kw in line_stripped.upper() for kw in SQL_START_KEYWORDS)):
                sql_lines.append(line_stripped)

    query = '\n'.join(sql_lines).strip()

    # Verify this looks like SQL
    if query and any(keyword in query.upper() for keyword in SQL_START_KEYWORDS):
        return query

    return None


def parse_sql_queries(generated_text: str) -> Tuple[List[str], Optional[str]]:
    """
    Parse SQL queries and ambiguity analysis from LLM response.

    Tries JSON parsing first, then falls back to text parsing.

    Args:
        generated_text: Raw text from LLM

    Returns:
        Tuple of (sql_queries, ambiguity_analysis)
    """
    # Try JSON parsing first
    json_data = parse_json_response(generated_text)

    if json_data:
        logger.debug("Successfully parsed JSON response")
        sql_queries, ambiguity_analysis = extract_sql_from_json(json_data)
        if sql_queries:  # If we got queries from JSON, return them
            return sql_queries, ambiguity_analysis

    # Fallback to text parsing
    logger.debug("Falling back to text parsing")

    # Extract ambiguity analysis from text
    text, ambiguity_analysis = extract_ambiguity_analysis_from_text(generated_text)

    # Parse SQL queries from text
    sql_queries = parse_sql_from_text(text)

    return sql_queries, ambiguity_analysis


def parse_corrected_queries(generated_text: str) -> List[str]:
    """
    Parse corrected SQL queries from LLM response.

    Args:
        generated_text: Raw text from LLM correction response

    Returns:
        List of corrected SQL queries
    """
    # Try JSON parsing first
    json_data = parse_json_response(generated_text)

    if json_data:
        logger.debug("Successfully parsed JSON correction response")
        sql_queries, _ = extract_sql_from_json(json_data)
        if sql_queries:
            return sql_queries

    # Fallback to text parsing
    logger.debug("Falling back to text parsing for correction")

    # Remove markdown fences
    text = remove_markdown_fences(generated_text)

    corrected_queries = []
    for section in text.split('\n\n'):
        section = section.strip()

        # Skip empty sections or comments
        if not section or section.startswith('#') or section.startswith('--'):
            continue

        # Remove "Query N:" prefixes if present
        if section.startswith('Query'):
            lines = section.split('\n')
            if len(lines) > 1:
                section = '\n'.join(lines[1:]).strip()

        # Verify this looks like SQL
        if section and any(kw in section.upper() for kw in SQL_START_KEYWORDS):
            corrected_queries.append(section)

    return corrected_queries
