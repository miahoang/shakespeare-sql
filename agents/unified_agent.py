#!/usr/bin/env python3
"""
Unified SQL Agent - Single agent with routing, SQL generation, and evaluation as tools.

This module replaces the three separate agents (RoutingAgent, SQLExpertAgent,
EvaluationAgent) with a single UnifiedSQLAgent whose tools are:

  Tool 1 – route_question     : DeBERTa classification + U validation
  Tool 2 – generate_sql       : RAG retrieval, SQL generation, correction, LoRA
  Tool 3 – evaluate_sql_result: Precision/recall/F1 against gold standard
  Tool 4 – generate_u_explanation          : User-friendly U explanation
  Tool 5 – generate_production_explanation : Quality-scored explanation (no GT)

All implementation details are preserved exactly from the originals.
"""

import logging
import sys
import os
import pathlib
import json
import re
import sqlite3
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_vectordb import VectorDatabase
from rag.evaluation_ambrosia_prompts_authors import (
    create_sql_generation_prompt,
    create_sql_correction_prompt,
)
from rag.sql_parsing_utils import parse_sql_queries, parse_corrected_queries
from rag.evaluation_config import ModelConfig
from evaluation.metrics import (
    evaluate_predicted_statements,
    evaluate_predicted_statements_flex,
)

try:
    from rag.hybrid_ambiguity_retrieval import HybridAmbiguityRetriever
except ImportError:
    HybridAmbiguityRetriever = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths and names
# ---------------------------------------------------------------------------

DEFAULT_MODEL_URL          = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME         = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_LORA_MODEL_URL     = None
DEFAULT_LORA_ADAPTER_NAME  = "llama-grpo"
DEFAULT_LORA_ADAPTER_PATH  = "finetuning/models/llama_grpo/best/checkpoint-1300"
DEFAULT_LORA_BASE_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DEBERTA_MODEL_PATH = "models/deberta-v3-base_20251029_221116"
DEFAULT_DEBERTA_BASE       = "microsoft/deberta-v3-base"
DEFAULT_SEMANTIC_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTORDB_CACHE_DIR = "data/vectordb_cache"
DEFAULT_CSV_PATH           = "data/ambrosia/ambrosia_with_unanswerable_validated.csv"
DEFAULT_PROMPT_MODULE      = "evaluation_ambrosia_prompts_authors"


# ---------------------------------------------------------------------------
# Data-classes (unchanged from originals)
# ---------------------------------------------------------------------------

@dataclass
class SQLGenerationResult:
    """Result from the SQL-generation tool."""
    category: str
    original_category: str
    interpretations: List[str]
    sql_queries: List[str]
    original_queries: Optional[List[str]]
    corrected_queries: Optional[List[str]]
    execution_successful: bool
    execution_errors: List[str]
    num_valid_queries: int
    lora_interpretations: Optional[List[str]]
    missed_interpretations: Optional[List[str]]
    similar_examples: List[Dict]
    reasoning: str
    reclassified_to_u: bool
    num_lora_sql_queries: int = 0
    prompts_used: Optional[Dict[str, str]] = None


@dataclass
class UValidationResult:
    """Result from U-question validation."""
    is_unanswerable: bool
    confidence: float
    explanation: str
    reasoning: str
    should_reclassify: bool
    suggested_category: Optional[str]


@dataclass
class SQLEvaluationResult:
    """Result from the SQL-evaluation tool."""
    precision: float
    recall: float
    f1_score: float
    all_found: bool
    precision_flex: float
    recall_flex: float
    f1_flex: float
    all_found_flex: bool
    num_predicted: int
    num_gold: int
    num_correct: int
    num_correct_flex: int
    execution_errors: List[str]
    num_executable: int
    explanation: str
    score_breakdown: str


@dataclass
class InterpretationQualityScore:
    """Quality score for a single interpretation+SQL pair."""
    interpretation: str
    sql_query: str
    semantic_similarity: float
    schema_correctness: float
    execution_success: bool
    overall_score: float
    issues: List[str]


# ---------------------------------------------------------------------------
# Unified agent
# ---------------------------------------------------------------------------

class UnifiedSQLAgent:
    """
    Single agent that exposes routing, SQL generation, and evaluation as tools.

    Tools
    -----
    route_question              – Classify question (U/AA/AU) with optional U validation
    generate_sql                – Full RAG + LoRA + correction pipeline
    evaluate_sql_result         – Precision/recall/F1 vs gold queries
    generate_u_explanation      – Plain-language explanation for U outcomes
    generate_production_explanation – Quality-scored explanation without ground truth
    """

    def __init__(
        self,
        # SQL-generation / routing parameters
        model_url: str = DEFAULT_MODEL_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        lora_model_url: Optional[str] = DEFAULT_LORA_MODEL_URL,
        lora_adapter_name: str = DEFAULT_LORA_ADAPTER_NAME,
        lora_adapter_path: str = DEFAULT_LORA_ADAPTER_PATH,
        lora_base_model: str = DEFAULT_LORA_BASE_MODEL,
        vectordb_cache_dir: str = DEFAULT_VECTORDB_CACHE_DIR,
        csv_path: str = DEFAULT_CSV_PATH,
        rag_k: int = 5,
        rag_k_other: int = 1,
        temperature: float = 0.0,
        lora_temperature: Optional[float] = None,
        use_sql_correction: bool = True,
        use_lora_validation: bool = True,
        hybrid_confidence_threshold: float = 0.7,
        seed: Optional[int] = 42,
        prompt_module: str = DEFAULT_PROMPT_MODULE,
        deberta_model_path: str = DEFAULT_DEBERTA_MODEL_PATH,
        deberta_base: str = DEFAULT_DEBERTA_BASE,
        validate_u_predictions: bool = True,
        max_correction_iterations: int = 3,
        # Evaluation parameters
        semantic_model: str = DEFAULT_SEMANTIC_MODEL,
    ):
        self.model_url = model_url
        self.model_name = model_name
        self.max_correction_iterations = max_correction_iterations
        self.lora_model_url = lora_model_url
        self.lora_adapter_name = lora_adapter_name
        self.lora_adapter_path = lora_adapter_path
        self.lora_base_model = lora_base_model
        self.rag_k = rag_k
        self.rag_k_other = rag_k_other
        self.prompt_module_name = prompt_module
        self.temperature = temperature
        self.lora_temperature = lora_temperature if lora_temperature is not None else temperature
        self.use_sql_correction = use_sql_correction
        self.use_lora_validation = use_lora_validation
        self.hybrid_confidence_threshold = hybrid_confidence_threshold
        self.validate_u_predictions = validate_u_predictions
        self.seed = seed

        # Dynamically import prompt functions from the specified module
        import importlib
        prompt_mod = importlib.import_module(f"rag.{prompt_module}")
        self.create_sql_generation_prompt = prompt_mod.create_sql_generation_prompt
        self.create_sql_correction_prompt = prompt_mod.create_sql_correction_prompt

        self.model_config = ModelConfig()

        # ---- Tool 1: DeBERTa classifier (routing) ----
        logger.info(f"Loading DeBERTa classifier from {deberta_model_path}")
        self.device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
        deberta_model_path = pathlib.Path(deberta_model_path).resolve()
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
            deberta_model_path
        )
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_base)
        self.deberta_model.to(self.device)
        self.deberta_model.eval()
        self.id2label = {0: "U", 1: "AA", 2: "AU"}
        logger.info(f"DeBERTa classifier loaded on {self.device}")

        # ---- Shared LLM client (SQL generation + evaluation) ----
        logger.info(f"Connecting to LLM server at {model_url}")
        self.client = OpenAI(base_url=model_url, api_key="dummy")
        try:
            models = self.client.models.list()
            logger.info(f"Connected to LLM server. Available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.error(f"Could not connect to LLM server: {e}")
            raise

        # ---- Tool 2: LoRA client or local model ----
        self.lora_client = None
        self.lora_model = None
        self.lora_tokenizer = None

        if use_lora_validation:
            if lora_model_url:
                logger.info(f"Connecting to LoRA validation server at {lora_model_url}")
                self.lora_client = OpenAI(base_url=lora_model_url, api_key="dummy")
                try:
                    models = self.lora_client.models.list()
                    logger.info(f"Connected to LoRA server. Available models: {[m.id for m in models.data]}")
                except Exception as e:
                    logger.error(f"Could not connect to LoRA server: {e}")
                    logger.warning("LoRA validation will be disabled")
                    self.use_lora_validation = False
                    self.lora_client = None
            else:
                logger.info(f"LoRA validation enabled. Will load model locally from {lora_adapter_path}")
                logger.warning(
                    "Local LoRA loading requires significant GPU memory. "
                    "Consider using --lora-model-url for separate vLLM server."
                )

        # ---- RAG vector database ----
        logger.info("Initializing RAG vector database...")
        self.vectordb = VectorDatabase(cache_dir=vectordb_cache_dir)
        self.vectordb.build_from_csv(
            csv_path=csv_path,
            split="train",
            ambiguous_only=False,
            force_rebuild=False,
        )
        logger.info(f"RAG database ready with {len(self.vectordb.examples)} examples")

        # ---- Hybrid retriever (AA questions) ----
        self.hybrid_retriever = None
        if HybridAmbiguityRetriever is not None:
            logger.info("Initializing hybrid ambiguity-aware retrieval for AA questions...")
            try:
                self.hybrid_retriever = HybridAmbiguityRetriever(
                    vectordb=self.vectordb,
                    semantic_weight=0.3,
                    structural_weight=0.7,
                )
                logger.info("Hybrid retrieval initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retrieval: {e}")
                logger.warning("Will use standard retrieval only")
                self.hybrid_retriever = None
        else:
            logger.warning("HybridAmbiguityRetriever not available. Will use standard retrieval only.")
            logger.warning("Install spacy: python -m spacy download en_core_web_sm")

        # ---- Tool 3: Sentence transformer for evaluation ----
        logger.info(f"Loading sentence transformer model: {semantic_model}")
        self.semantic_model = SentenceTransformer(semantic_model)

        logger.info("UnifiedSQLAgent initialized successfully")

    # =========================================================================
    # TOOL 1 — route_question
    # =========================================================================

    def route_question(
        self,
        question: str,
        db_dump: str,
        db_file: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float], bool]:
        """
        Tool 1: Classify question into U, AA, or AU using DeBERTa.

        Optionally validates U predictions with LLM reasoning
        (when validate_u_predictions=True and db_file is provided).

        Returns
        -------
        (category, probs, validated) where
          category  – "U", "AA", or "AU"
          probs     – {'U': p, 'AA': p, 'AU': p}
          validated – True when U prediction was confirmed/overridden by LLM
        """
        # Remove INSERT statements to reduce sequence length
        db_dump_clean = self._remove_insert_statements(db_dump)

        # Tokenize input
        inputs = self.deberta_tokenizer(
            db_dump_clean,
            question,
            return_tensors="pt",
            truncation="only_first",
            max_length=1024,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.deberta_model(**inputs)
            logits = outputs.logits
            probs_tensor = torch.softmax(logits, dim=-1)[0]

        pred_idx = probs_tensor.argmax().item()
        prediction = self.id2label[pred_idx]

        probs = {
            "U": probs_tensor[0].item(),
            "AA": probs_tensor[1].item(),
            "AU": probs_tensor[2].item(),
        }

        logger.info(f"DeBERTa prediction: {prediction} (confidence: {probs[prediction]:.3f})")

        # Validate U predictions if enabled using LLM reasoning
        validated = False
        if prediction == "U" and self.validate_u_predictions and db_file:
            logger.info("=" * 80)
            logger.info("U VALIDATION: Using LLM reasoning to confirm unanswerability")
            logger.info("=" * 80)
            try:
                validation_prompt = f"""Validate whether the following question is truly UNANSWERABLE given the database schema.

Question: {question}

Database Schema:
{db_dump}

Your task:
1. Carefully analyze whether this question can be answered with the given database
2. Check if ALL required information exists in the schema (tables, columns, data)
3. A question is UNANSWERABLE if:
   - It asks for data/columns that don't exist in the schema (e.g., "box office revenue" when only "budget" exists)
   - It asks about entities not in the database (e.g., "Action" genre when only "Horror" and "Thriller" exist)
   - The required relationships between tables don't exist
   - The question is logically impossible to answer with the available data

4. A question is ANSWERABLE if:
   - All required columns exist in the schema
   - All required data values exist or could exist in the tables
   - The question can be translated to a valid SQL query that would return meaningful results

Provide your analysis in the following format:

VALIDATION: [CONFIRMED_U / SHOULD_RECLASSIFY]
CONFIDENCE: [0.0-1.0]

REASONING:
[Detailed explanation of why this question is or isn't answerable, specifically listing what's missing or available]"""

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert at validating SQL question answerability. "
                                "Be strict - if ANY required information is missing from the "
                                "schema, the question is UNANSWERABLE."
                            ),
                        },
                        {"role": "user", "content": validation_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )

                reasoning = response.choices[0].message.content.strip()
                logger.info(f"U VALIDATION Agent Response:\n{reasoning[:500]}...")

                should_reclassify = False
                validation_match = re.search(
                    r"VALIDATION:\s*(CONFIRMED_U|SHOULD_RECLASSIFY)", reasoning, re.IGNORECASE
                )
                if validation_match:
                    if validation_match.group(1).upper() == "SHOULD_RECLASSIFY":
                        should_reclassify = True

                if should_reclassify:
                    logger.info("=" * 80)
                    logger.info("U VALIDATION RESULT: Agent says question is ANSWERABLE")
                    logger.info("RECLASSIFYING from U to AA (question is actually answerable)")
                    logger.info("Will proceed with full SQL generation pipeline")
                    logger.info("=" * 80)
                    prediction = "AA"
                    validated = True
                else:
                    logger.info("=" * 80)
                    logger.info("U VALIDATION RESULT: Agent confirms question is UNANSWERABLE")
                    logger.info("CONFIRMED as U (question is truly unanswerable)")
                    logger.info("Will return early - no further processing")
                    logger.info("=" * 80)
                    validated = True

            except Exception as e:
                logger.warning(f"U validation failed with exception: {e}")
                logger.info("=" * 80)
                logger.info("U VALIDATION RESULT: Validation failed")
                logger.info("Keeping DeBERTa prediction as U (question treated as unanswerable)")
                logger.info("Will return early - no further processing")
                logger.info("=" * 80)

        return prediction, probs, validated

    # =========================================================================
    # TOOL 2 — generate_sql
    # =========================================================================

    def generate_sql(
        self,
        question: str,
        db_dump: str,
        db_file: str,
        category: Optional[str] = None,
        retry_guidance: Optional[str] = None,
        previous_errors: Optional[List[str]] = None,
    ) -> SQLGenerationResult:
        """
        Tool 2: Generate SQL queries with integrated classification.

        When category is None the route_question tool is called first.
        Covers retrieval, generation, execution, iterative correction,
        LoRA interpretation validation, deduplication, and U reclassification.
        """
        # Step 0: Classify using route_question tool (if category not provided)
        if category is None:
            category, classification_probs, u_validated = self.route_question(
                question, db_dump, db_file
            )
            logger.info(f"Classified as: {category} (probs: {classification_probs})")
        else:
            logger.info(f"Using provided category: {category}")
            u_validated = False

        # If question is unanswerable, return early
        if category == "U":
            logger.info("=" * 80)
            logger.info("RETURNING EARLY: Question confirmed as unanswerable (U)")
            logger.info("No further SQL generation or LoRA validation will be performed")
            logger.info("=" * 80)
            return SQLGenerationResult(
                category="U",
                original_category="U",
                interpretations=["Question is unanswerable with the given database"],
                sql_queries=[],
                original_queries=None,
                corrected_queries=None,
                execution_successful=False,
                execution_errors=["Question classified as unanswerable"],
                num_valid_queries=0,
                lora_interpretations=None,
                missed_interpretations=None,
                similar_examples=[],
                reasoning=f"Question classified as unanswerable (U) by DeBERTa. Validated: {u_validated}",
                reclassified_to_u=False,
                prompts_used={},
            )

        logger.info(f"Generating SQL for {category} question (NOT a U question)")

        # Step 1: Retrieve similar examples
        similar_examples, predicted_ambig_type, ambig_confidence = self._retrieve_examples(
            question, db_dump, category
        )
        logger.info(f"Retrieved {len(similar_examples)} similar examples")
        if predicted_ambig_type:
            logger.info(f"Predicted ambiguity type: {predicted_ambig_type} (conf: {ambig_confidence:.3f})")

        # Step 2: Generate SQL
        sql_queries, interpretations, ambiguity_analysis, prompts_used = self._generate_sql_with_rag(
            question, db_dump, category, similar_examples, retry_guidance, previous_errors,
            predicted_ambig_type=predicted_ambig_type,
        )
        logger.info(f"Generated {len(sql_queries)} SQL queries")

        # Step 3: Execute queries
        execution_successful, execution_errors, num_valid, success_flags = self._execute_queries(
            sql_queries, db_file
        )
        logger.info(f"Execution: {num_valid}/{len(sql_queries)} queries succeeded")

        # Step 4: Iterative SQL correction
        original_queries_before_correction = None
        corrected_queries = None
        if self.use_sql_correction and sql_queries and not execution_successful:
            logger.info(f"Applying iterative SQL correction (up to {self.max_correction_iterations} attempts)...")
            original_queries_before_correction = sql_queries.copy()
            corrected_queries, execution_errors, num_valid = self._iterative_review_and_fix(
                sql_queries, question, db_dump, db_file, similar_examples
            )
            execution_successful = num_valid == len(corrected_queries)
            logger.info(f"After iterative correction: {num_valid}/{len(corrected_queries)} queries succeeded")
        elif execution_successful:
            logger.info("All queries successful - skipping correction")

        # Step 5: LoRA interpretation validation (AA only)
        lora_interpretations = None
        missed_interpretations = None
        additional_sql_queries = []

        if category == "U":
            logger.error("BUG: Category is U but code reached LoRA section (should have returned early)")
            raise RuntimeError("U question reached LoRA section - this should never happen")

        if category == "AA" and self.use_lora_validation:
            logger.info("Running LoRA to generate ALL interpretations (AA question only)")
            lora_interpretations, missed_interpretations = self._validate_interpretations_with_lora(
                question, db_dump, interpretations, similar_examples
            )
            if missed_interpretations:
                logger.info(f"LoRA model generated {len(missed_interpretations)} interpretations")

                logger.info(f"Generating SQL for {len(missed_interpretations)} interpretations...")
                successful_missed_interpretations = []
                for missed_interp in missed_interpretations:
                    try:
                        additional_queries = self._generate_sql_for_interpretation(
                            interpretation=missed_interp,
                            db_dump=db_dump,
                            similar_examples=similar_examples,
                        )
                        if additional_queries:
                            additional_sql_queries.extend(additional_queries)
                            successful_missed_interpretations.append(missed_interp)
                            logger.info(
                                f"Generated {len(additional_queries)} SQL queries for: {missed_interp[:80]}..."
                            )
                    except Exception as e:
                        logger.warning(f"Failed to generate SQL for additional interpretation: {e}")

                if additional_sql_queries:
                    logger.info(
                        f"Adding {len(additional_sql_queries)} additional SQL queries from LoRA interpretations"
                    )
                    interpretations.extend(successful_missed_interpretations)
                    logger.info(
                        f"Added {len(successful_missed_interpretations)} LoRA interpretations to interpretations list"
                    )
                    missed_interpretations = successful_missed_interpretations

                    num_initial_valid = num_valid

                    logger.info("Checking LoRA queries for errors...")
                    lora_execution_successful, lora_errors, num_lora_valid, lora_success_flags = (
                        self._execute_queries(additional_sql_queries, db_file)
                    )
                    logger.info(
                        f"LoRA queries execution: {num_lora_valid}/{len(additional_sql_queries)} successful"
                    )

                    final_lora_queries = additional_sql_queries
                    if not lora_execution_successful and self.use_sql_correction:
                        logger.info(
                            f"Fixing {len(additional_sql_queries) - num_lora_valid} failed LoRA queries..."
                        )
                        final_lora_queries, lora_errors, num_lora_valid = self._iterative_review_and_fix(
                            additional_sql_queries, question, db_dump, db_file, similar_examples
                        )
                        logger.info(
                            f"After fixing: {num_lora_valid}/{len(final_lora_queries)} LoRA queries successful"
                        )
                    elif lora_execution_successful:
                        logger.info("All LoRA queries successful - no correction needed")

                    current_queries = corrected_queries if corrected_queries else sql_queries
                    current_queries.extend(final_lora_queries)

                    num_valid = num_initial_valid + num_lora_valid
                    execution_successful = num_valid == len(current_queries)
                    execution_errors = execution_errors + lora_errors

                    if corrected_queries:
                        corrected_queries = current_queries
                    else:
                        sql_queries = current_queries

                    logger.info(
                        f"Total after LoRA: {num_valid}/{len(current_queries)} queries successful "
                        f"({num_initial_valid} initial + {num_lora_valid} LoRA)"
                    )
                else:
                    logger.warning("No SQL queries could be generated for any LoRA interpretations")
                    missed_interpretations = []
        elif category == "AU":
            logger.info("Skipping LoRA validation (AU question has single interpretation)")

        # Step 5.5: Deduplicate SQL queries based on execution results
        final_queries = corrected_queries if corrected_queries else sql_queries
        if final_queries:
            from evaluation.metrics import remove_duplicate_results, compare_query_results

            original_count = len(final_queries)
            all_exec_outputs = {}
            for query in final_queries:
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute(query)
                    all_exec_outputs[query] = cursor.fetchall()
                    conn.close()
                except sqlite3.DatabaseError as e:
                    from evaluation.exceptions import PredQueryExecutionError
                    all_exec_outputs[query] = PredQueryExecutionError(query, e)

            deduplicated_outputs = remove_duplicate_results(all_exec_outputs)
            deduplicated_queries = list(deduplicated_outputs.keys())

            if len(deduplicated_queries) < original_count:
                logger.info(
                    f"Removed {original_count - len(deduplicated_queries)} queries with duplicate results "
                    f"({original_count} -> {len(deduplicated_queries)})"
                )
                if corrected_queries:
                    corrected_queries = deduplicated_queries
                else:
                    sql_queries = deduplicated_queries

        # Step 6: Reclassify as U if no valid queries
        reclassified_to_u = False
        final_category = category
        reasoning = ambiguity_analysis or "SQL queries generated successfully"

        if num_valid == 0:
            reclassified_to_u = True
            final_category = "U"
            reasoning = (
                f"Reclassified as U: No valid SQL queries could be generated or executed. "
                f"Original category: {category}"
            )
            logger.info(reasoning)

        return SQLGenerationResult(
            category=final_category,
            original_category=category,
            interpretations=interpretations,
            sql_queries=corrected_queries if corrected_queries else sql_queries,
            original_queries=original_queries_before_correction,
            corrected_queries=corrected_queries,
            execution_successful=execution_successful,
            execution_errors=execution_errors,
            num_valid_queries=num_valid,
            lora_interpretations=lora_interpretations,
            missed_interpretations=missed_interpretations,
            similar_examples=similar_examples,
            reasoning=reasoning,
            reclassified_to_u=reclassified_to_u,
            num_lora_sql_queries=len(additional_sql_queries),
            prompts_used=prompts_used,
        )

    # =========================================================================
    # TOOL 3 — evaluate_sql_result
    # =========================================================================

    def evaluate_sql_result(
        self,
        predicted_queries: List[str],
        gold_queries: List[str],
        db_file: str,
        category: str = "AA",
        remove_duplicates: bool = True,
    ) -> SQLEvaluationResult:
        """
        Tool 3: Evaluate predicted SQL queries against gold standard.

        Computes both standard (exact-match) and flex (extra-columns-allowed) metrics.
        """
        logger.info(
            f"Evaluating {len(predicted_queries)} predicted queries "
            f"against {len(gold_queries)} gold queries"
        )

        # Ground truth is U – all metrics are 0
        gold_is_unanswerable = len(gold_queries) == 0 or any(
            q.strip().upper() == "UNANSWERABLE" for q in gold_queries
        )
        if gold_is_unanswerable:
            logger.info("Gold query is UNANSWERABLE - ground truth is U category")
            if predicted_queries:
                logger.info(
                    f"Model generated {len(predicted_queries)} queries for U question - "
                    "assigning 0 metrics"
                )
            return SQLEvaluationResult(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                all_found=False,
                precision_flex=0.0,
                recall_flex=0.0,
                f1_flex=0.0,
                all_found_flex=False,
                num_predicted=len(predicted_queries),
                num_gold=0,
                num_correct=0,
                num_correct_flex=0,
                execution_errors=[],
                num_executable=len(predicted_queries),
                explanation=(
                    "Ground truth is UNANSWERABLE - model incorrectly generated SQL "
                    "for unanswerable question"
                ),
                score_breakdown=(
                    "Ground truth category is U (unanswerable). "
                    "Any SQL predictions are incorrect."
                ),
            )

        try:
            metrics = evaluate_predicted_statements(
                file_name=db_file,
                pred_statements=predicted_queries,
                gold_sql_queries=gold_queries,
                remove_duplicates_predictions=remove_duplicates,
                calculate_unique=True,
            )
            metrics_flex = evaluate_predicted_statements_flex(
                file_name=db_file,
                pred_statements=predicted_queries,
                gold_sql_queries=gold_queries,
                remove_duplicates_predictions=remove_duplicates,
                calculate_unique=True,
            )

            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            f1_score = metrics.get("f1_score", 0.0)
            all_found = metrics.get("all_found", False)
            execution_errors = metrics.get("execution_errors", [])

            precision_flex = metrics_flex.get("precision_flex", 0.0)
            recall_flex = metrics_flex.get("recall_flex", 0.0)
            f1_flex = metrics_flex.get("f1_flex", 0.0)
            all_found_flex = metrics_flex.get("all_found_flex", False)

            num_executable = len(predicted_queries) - len(execution_errors)
            num_correct = int(recall * len(gold_queries))
            num_correct_flex = int(recall_flex * len(gold_queries))

            explanation = self._generate_sql_explanation(
                precision, recall, f1_score, all_found,
                len(predicted_queries), len(gold_queries),
                num_correct, num_executable, category,
            )
            score_breakdown = self._generate_score_breakdown(
                precision, recall, f1_score,
                len(predicted_queries), len(gold_queries),
                num_correct, num_executable, execution_errors,
                precision_flex, recall_flex, f1_flex, num_correct_flex,
            )

            result = SQLEvaluationResult(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                all_found=all_found,
                precision_flex=precision_flex,
                recall_flex=recall_flex,
                f1_flex=f1_flex,
                all_found_flex=all_found_flex,
                num_predicted=len(predicted_queries),
                num_gold=len(gold_queries),
                num_correct=num_correct,
                num_correct_flex=num_correct_flex,
                execution_errors=execution_errors,
                num_executable=num_executable,
                explanation=explanation,
                score_breakdown=score_breakdown,
            )

            logger.info(
                f"Evaluation: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f} | "
                f"Flex: P={precision_flex:.3f}, R={recall_flex:.3f}, F1={f1_flex:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error evaluating SQL queries: {e}", exc_info=True)
            return SQLEvaluationResult(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                all_found=False,
                precision_flex=0.0,
                recall_flex=0.0,
                f1_flex=0.0,
                all_found_flex=False,
                num_predicted=len(predicted_queries),
                num_gold=len(gold_queries),
                num_correct=0,
                num_correct_flex=0,
                execution_errors=[f"Evaluation error: {str(e)}"],
                num_executable=0,
                explanation=f"Error during evaluation: {str(e)}",
                score_breakdown=f"Evaluation failed: {str(e)}",
            )

    # =========================================================================
    # TOOL 4 — generate_u_explanation
    # =========================================================================

    def generate_u_explanation(
        self,
        question: str,
        db_dump: str,
        routing_reasoning: str,
        sql_generation_attempted: bool = True,
        sql_generation_failed: bool = True,
    ) -> str:
        """
        Tool 4: Generate a user-friendly explanation for a confirmed U question.

        Called after the routing tool has already validated the U classification;
        does not re-validate, only produces the explanation text.
        """
        logger.info("Generating explanation for confirmed U question")
        return self._generate_u_explanation(
            is_unanswerable=True,
            confidence=0.95,
            question=question,
            sql_generation_attempted=sql_generation_attempted,
            sql_generation_failed=sql_generation_failed,
        )

    # =========================================================================
    # TOOL 5 — generate_production_explanation
    # =========================================================================

    def generate_production_explanation(
        self,
        question: str,
        interpretations: List[str],
        sql_queries: List[str],
        db_dump: str,
        db_file: str,
    ) -> Tuple[str, str, List[InterpretationQualityScore]]:
        """
        Tool 5: Generate quality-scored explanation for AA/AU results (no ground truth).

        Returns (explanation_text, score_breakdown, quality_scores).
        """
        if not sql_queries:
            return ("No SQL queries were generated for this question.", "No scores to report.", [])

        logger.info(f"Generating explanations for {len(sql_queries)} SQL queries")
        query_explanations = []
        for sql in sql_queries:
            explanation = self._generate_query_explanation(sql, db_dump)
            query_explanations.append(explanation)

        quality_scores = []
        for explanation, sql in zip(query_explanations, sql_queries):
            score = self.calculate_interpretation_quality(
                question=question,
                interpretation=explanation,
                sql_query=sql,
                db_dump=db_dump,
                db_file=db_file,
            )
            quality_scores.append(score)

        quality_scores.sort(key=lambda x: x.overall_score, reverse=True)

        explanation_text = f"This question has {len(sql_queries)} possible interpretation(s):\n\n"
        for i, score in enumerate(quality_scores, 1):
            explanation_text += f"{i}. {score.interpretation}\n"
            explanation_text += f"   SQL Query:\n   ```sql\n   {score.sql_query}\n   ```\n"
            explanation_text += f"   Quality Score: {score.overall_score:.2f}/10.0\n"
            explanation_text += f"   - Semantic alignment: {score.semantic_similarity:.2%}\n"
            explanation_text += f"   - Schema correctness: {score.schema_correctness:.2%}\n"
            explanation_text += f"   - Execution: {'✓ Success' if score.execution_success else '✗ Failed'}\n"
            if score.issues:
                explanation_text += f"   - Issues: {'; '.join(score.issues)}\n"
            explanation_text += "\n"

        score_breakdown = self._generate_production_score_breakdown(quality_scores)
        return explanation_text, score_breakdown, quality_scores

    # =========================================================================
    # Private helpers – SQL generation (from SQLExpertAgent)
    # =========================================================================

    def _remove_insert_statements(self, db_dump: str) -> str:
        lines = db_dump.split("\n")
        return "\n".join(
            line for line in lines if not line.strip().upper().startswith("INSERT INTO")
        )

    def _retrieve_examples(
        self, question: str, db_dump: str, category: str
    ) -> Tuple[List[Dict], Optional[str], float]:
        predicted_type = None
        confidence = 0.0

        if self.hybrid_retriever is not None and category == "AA":
            try:
                predicted_type, confidence = self.hybrid_retriever.predict_ambiguity_type(
                    question=question,
                    db_dump=db_dump,
                    k_neighbors=20,
                    min_confidence=0.3,
                )
                logger.info(f"Ambiguity type prediction: {predicted_type} (conf: {confidence:.3f})")

                if predicted_type in ["attachment", "scope"] and confidence >= self.hybrid_confidence_threshold:
                    logger.info(f"Using type-specific hybrid retrieval for {predicted_type} ambiguity")
                    type_specific_examples = self.hybrid_retriever.retrieve_by_ambiguity_type(
                        question=question,
                        db_dump=db_dump,
                        ambig_type=predicted_type,
                        k=self.rag_k,
                        fallback_to_semantic=True,
                    )
                    if type_specific_examples:
                        if self.rag_k_other > 0:
                            other_examples = self.vectordb.retrieve_similar(
                                question=question,
                                db_dump=db_dump,
                                k=self.rag_k + self.rag_k_other,
                                exclude_exact_match=True,
                            )
                            type_specific_ids = {
                                ex.get("id", ex.get("question", "")) for ex in type_specific_examples
                            }
                            other_examples = [
                                ex for ex in other_examples
                                if ex.get("id", ex.get("question", "")) not in type_specific_ids
                            ][: self.rag_k_other]
                            similar_examples = type_specific_examples + other_examples
                            logger.info(
                                f"Hybrid retrieval: {len(type_specific_examples)} {predicted_type} "
                                f"+ {len(other_examples)} other"
                            )
                        else:
                            similar_examples = type_specific_examples
                            logger.info(
                                f"Hybrid retrieval: {len(similar_examples)} {predicted_type} examples"
                            )
                        return similar_examples, predicted_type, confidence
                    else:
                        logger.warning("Type-specific retrieval failed, falling back to standard RAG")
                else:
                    logger.info("Not attachment/scope or low confidence - using standard RAG")
            except Exception as e:
                logger.warning(f"Hybrid retrieval error: {e}. Using standard RAG.")
                predicted_type = None
                confidence = 0.0

        if hasattr(self.vectordb, "retrieve_similar_with_category_distribution"):
            similar_examples = self.vectordb.retrieve_similar_with_category_distribution(
                question=question,
                db_dump=db_dump,
                question_category=category,
                k_same_category=self.rag_k,
                k_other_category=self.rag_k_other,
                exclude_exact_match=True,
            )
            if not similar_examples:
                similar_examples = self.vectordb.retrieve_similar(
                    question=question, db_dump=db_dump, k=self.rag_k, exclude_exact_match=True
                )
        else:
            similar_examples = self.vectordb.retrieve_similar(
                question=question, db_dump=db_dump, k=self.rag_k, exclude_exact_match=True
            )

        return similar_examples, predicted_type, confidence

    def _generate_sql_with_rag(
        self,
        question: str,
        db_dump: str,
        category: str,
        similar_examples: List[Dict],
        retry_guidance: Optional[str] = None,
        previous_errors: Optional[List[str]] = None,
        schema_ambiguities: Optional[str] = None,
        predicted_ambig_type: Optional[str] = None,
    ) -> Tuple[List[str], List[str], str, Dict[str, str]]:
        if predicted_ambig_type is not None and category == "AA":
            logger.info(
                f"Using author prompts with {predicted_ambig_type}-specific examples from hybrid retrieval"
            )

        if not similar_examples:
            logger.info("No RAG examples provided - using zero-shot SQL generation")
            examples_text = "No examples provided. Generate SQL based on the schema and question alone."
        else:
            examples_text = self.vectordb.format_examples_for_prompt(
                similar_examples,
                include_sql=True,
                question_category=category,
                k_same_category=self.rag_k,
                k_other_category=self.rag_k_other,
            )

        prompts = self.create_sql_generation_prompt(
            question, db_dump, examples_text, category=category,
            retry_guidance=retry_guidance, previous_errors=previous_errors,
            schema_ambiguities=schema_ambiguities,
            predicted_ambig_type=predicted_ambig_type,
        )

        try:
            gen_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                "temperature": self.temperature,
                "max_tokens": self.model_config.max_tokens_generation,
                "response_format": {"type": "json_object"},
            }
            if self.seed is not None:
                gen_params["seed"] = self.seed

            response = self.client.chat.completions.create(**gen_params)
            generated_text = response.choices[0].message.content.strip()

            sql_queries, ambiguity_analysis = parse_sql_queries(generated_text)
            interpretations = self._extract_interpretations(generated_text)

            if interpretations:
                logger.info(f"Extracted {len(interpretations)} interpretations from Qwen response")
            else:
                logger.debug("No interpretations in Qwen response (expected with SQL-only format)")

            return sql_queries, interpretations, ambiguity_analysis, prompts

        except Exception as e:
            logger.error(f"Error generating SQL: {e}", exc_info=True)
            return [], [], f"Error: {str(e)}", {}

    def _generate_sql_for_interpretation(
        self, interpretation: str, db_dump: str, similar_examples: List[Dict]
    ) -> List[str]:
        examples_text = ""
        if similar_examples:
            examples_text = self.vectordb.format_examples_for_prompt(
                similar_examples,
                include_sql=True,
                question_category="AA",
                k_same_category=self.rag_k,
                k_other_category=self.rag_k_other,
            )

        system_message = (
            "You are an expert SQL query generator. Generate a SQL query that answers "
            "the given interpretation of a question based on the provided database schema."
        )
        user_message = f"""Database Schema:
{db_dump}

Similar examples:
{examples_text}

Interpretation:
{interpretation}

Task: Generate a SQL query that answers this specific interpretation.

Return ONLY the SQL query, no explanations or additional text."""

        try:
            gen_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.temperature,
                "max_tokens": 512,
            }
            if self.seed is not None:
                gen_params["seed"] = self.seed

            response = self.client.chat.completions.create(**gen_params)
            generated_text = response.choices[0].message.content.strip()

            sql_queries = []
            text = generated_text.strip()
            if "```" in text:
                sql_blocks = re.findall(r"```(?:sql)?\s*\n(.*?)\n```", text, re.DOTALL)
                sql_queries.extend([block.strip() for block in sql_blocks if block.strip()])
            else:
                sql_queries.append(text)

            return sql_queries

        except Exception as e:
            logger.error(f"Error generating SQL for interpretation '{interpretation[:50]}...': {e}")
            return []

    def _extract_interpretations(self, generated_text: str) -> List[str]:
        interpretations = []
        try:
            json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "interpretations" in data:
                    raw_interps = data["interpretations"]
                    for item in raw_interps:
                        if isinstance(item, str):
                            interpretations.append(item)
                        elif isinstance(item, dict):
                            if "interpretation" in item:
                                interpretations.append(item["interpretation"])
                            elif "text" in item:
                                interpretations.append(item["text"])
                elif "interpretation" in data:
                    interp = data["interpretation"]
                    if isinstance(interp, list):
                        interpretations = interp
                    else:
                        interpretations = [interp]
        except Exception as e:
            logger.debug(f"Could not extract interpretations from JSON: {e}")
        return interpretations

    def _execute_queries(
        self, sql_queries: List[str], db_file: str
    ) -> Tuple[bool, List[str], int, List[bool]]:
        if not sql_queries:
            return False, ["No queries to execute"], 0, []

        execution_errors = []
        success_flags = []
        num_valid = 0

        for i, query in enumerate(sql_queries):
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute(query)
                cursor.fetchall()
                conn.close()
                success_flags.append(True)
                num_valid += 1
            except Exception as e:
                error_msg = f"Query {i+1} failed: {str(e)}"
                execution_errors.append(error_msg)
                success_flags.append(False)
                logger.debug(error_msg)

        execution_successful = num_valid == len(sql_queries)
        return execution_successful, execution_errors, num_valid, success_flags

    def _iterative_review_and_fix(
        self,
        sql_queries: List[str],
        question: str,
        db_dump: str,
        db_file: str,
        similar_examples: List[Dict] = None,
    ) -> Tuple[List[str], List[str], int]:
        current_queries = sql_queries
        iteration = 0

        while iteration < self.max_correction_iterations:
            execution_successful, execution_errors, num_valid, success_flags = self._execute_queries(
                current_queries, db_file
            )
            logger.info(
                f"Correction iteration {iteration + 1}/{self.max_correction_iterations}: "
                f"{num_valid}/{len(current_queries)} queries successful"
            )

            if execution_successful:
                logger.info("All queries executing successfully - stopping correction")
                return current_queries, execution_errors, num_valid

            logger.info(
                f"Reviewing all {len(current_queries)} queries to fix {len(execution_errors)} errors..."
            )
            corrected_queries = self._correct_sql_queries(
                current_queries, db_dump, question, execution_errors, success_flags, similar_examples
            )

            if corrected_queries == current_queries:
                logger.info("Review didn't change queries - stopping iteration")
                return current_queries, execution_errors, num_valid

            current_queries = corrected_queries
            iteration += 1

        logger.info(f"Max correction iterations ({self.max_correction_iterations}) reached")
        execution_successful, execution_errors, num_valid, success_flags = self._execute_queries(
            current_queries, db_file
        )
        return current_queries, execution_errors, num_valid

    def _correct_sql_queries(
        self,
        sql_queries: List[str],
        db_dump: str,
        question: str,
        execution_errors: List[str] = None,
        success_flags: List[bool] = None,
        similar_examples: List[Dict] = None,
    ) -> List[str]:
        if not sql_queries:
            return []

        try:
            examples_text = ""
            if similar_examples:
                examples_text = self.vectordb.format_examples_for_prompt(
                    similar_examples,
                    include_sql=True,
                    question_category="AA",
                    k_same_category=self.rag_k,
                    k_other_category=self.rag_k_other,
                )

            prompts = self.create_sql_correction_prompt(
                question, db_dump, sql_queries, execution_errors, examples_text
            )

            gen_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                "temperature": self.temperature,
                "max_tokens": self.model_config.max_tokens_correction,
                "response_format": {"type": "json_object"},
            }
            if self.seed is not None:
                gen_params["seed"] = self.seed

            response = self.client.chat.completions.create(**gen_params)
            corrected_text = response.choices[0].message.content.strip()
            corrected_queries = parse_corrected_queries(corrected_text)

            logger.info(
                f"Correction: Reviewed {len(sql_queries)} queries, "
                f"returned {len(corrected_queries)} corrected queries"
            )
            return corrected_queries

        except Exception as e:
            logger.error(f"Error correcting SQL: {e}", exc_info=True)
            return sql_queries

    def _validate_interpretations_with_lora(
        self,
        question: str,
        db_dump: str,
        qwen_interpretations: List[str],
        similar_examples: List[Dict],
        schema_ambiguities: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        if not self.use_lora_validation:
            return None, None

        try:
            rag_examples_text = ""
            if similar_examples:
                rag_examples_text = self.vectordb.format_examples_for_prompt(
                    similar_examples,
                    include_sql=False,
                    question_category="AA",
                    k_same_category=self.rag_k,
                    k_other_category=self.rag_k_other,
                )
                rag_examples_text = (
                    f"\nExamples of similar ambiguous questions and their interpretations:\n\n"
                    f"{rag_examples_text}\n\n---\n\n"
                )

            user_message = f"""Given:

Database Schema:
{db_dump}

Question:
{question}

Similar examples:
{rag_examples_text}

Task: List every distinct way the question could be intepreted, given the database schema above.

Return JSON format:
{{
  "interpretations": ["interpretation 1", "interpretation 2", ...]
}}"""

            system_message = """You are an expert database question analyst specializing in disambiguating ambiguous natural language queries.

Your task: Given a database schema and an ambiguous question, identify and generate ALL semantically distinct interpretations of the question.

The question can contain different types of linguistic ambiguity:

1. SCOPE AMBIGUITY: Quantifier scope ("each", "every", "all") is unclear
   - Example: "Show me the toys every cat has."
   - Collective interpretation (property shared by ALL entities together): "List the toys that all cats have in common."
   - Distributive interpretation (property of EACH entity individually): "For each cat, list their toys."

2. ATTACHMENT AMBIGUITY: "X and Y [modifier]" - modifier attachment is unclear
   - Example: "Give me dogs and cats who are orange. Show them in one table."
   - Broad scope (modifier applies to BOTH X and Y): "Select all dogs who are orange AND cats who are orange."
   - Narrow scope (modifier applies ONLY to Y): "Select ALL dogs (no color filter) AND cats who are orange."

3. VAGUENESS: Underspecified terms have multiple possible referents
   - Example: "What technology does Rio know?"
   - Interpretation 1: "What software does Rio know?
   - Interpretation 2: "What programming languages does Rio know?"
   - Interpretation 3: "What software and programming language does Rio know?"

Guidelines for generating interpretations:
- Interpretations must be in natural language (not SQL)
- Follow the examples above for each ambiguity type closely (how many interpretations you should generate and how you should phrase them)
- Each interpretation must represent a DIFFERENT possible meaning or intent
- Interpretations should differ in WHAT data is being requested, not just HOW it's phrased (if they lead to the same SQL query/result, they are NOT distinct)
- Ground each interpretation in the actual schema (tables, columns, relationships)
- Make each interpretation specific and executable (could be translated to SQL)
- Avoid synonym variations - focus on semantic differences"""

            generated_text = None

            if self.lora_client is not None:
                logger.debug("Using vLLM server for LoRA validation")
                lora_model_name = self.lora_adapter_name
                try:
                    models = self.lora_client.models.list()
                    available_models = [m.id for m in models.data]
                    if lora_model_name not in available_models:
                        logger.warning(
                            f"LoRA model '{lora_model_name}' not found on server. "
                            f"Available: {available_models}"
                        )
                        lora_model_name = self.lora_base_model
                except Exception as e:
                    logger.debug(f"Could not list models: {e}")

                gen_params = {
                    "model": lora_model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": self.lora_temperature,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                    "response_format": {"type": "json_object"},
                }
                if self.seed is not None:
                    gen_params["seed"] = self.seed

                response = self.lora_client.chat.completions.create(**gen_params)
                generated_text = response.choices[0].message.content.strip()

            else:
                logger.debug("Using local LoRA model for validation")
                self._load_lora_model()
                if self.lora_model is None:
                    return None, None

                inputs = self.lora_tokenizer(
                    user_message, return_tensors="pt"
                ).to(self.lora_model.device)

                with torch.no_grad():
                    outputs = self.lora_model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=self.lora_temperature if self.lora_temperature > 0 else 0.8,
                        do_sample=True,
                        pad_token_id=self.lora_tokenizer.eos_token_id,
                    )
                generated_text = self.lora_tokenizer.decode(outputs[0], skip_special_tokens=True)

            if generated_text is None:
                return None, None

            logger.info(f"LoRA raw output: {generated_text[:500]}...")

            try:
                parsed = json.loads(generated_text)
                all_interpretations = parsed.get("interpretations", [])

                if not isinstance(all_interpretations, list):
                    logger.warning(
                        f"LoRA output 'interpretations' is not a list: {type(all_interpretations)}"
                    )
                    return None, None

                all_interpretations = [
                    interp.strip()
                    for interp in all_interpretations
                    if isinstance(interp, str) and interp.strip()
                ]

                if all_interpretations:
                    logger.info(f"LoRA generated {len(all_interpretations)} interpretations:")
                    for i, interp in enumerate(all_interpretations, 1):
                        logger.info(f"  {i}. {interp[:100]}{'...' if len(interp) > 100 else ''}")
                    return all_interpretations, all_interpretations
                else:
                    logger.info("LoRA returned no interpretations")
                    return None, None

            except json.JSONDecodeError as e:
                logger.warning(f"LoRA output is not valid JSON: {e}")
                logger.debug(f"Raw output: {generated_text}")
                return None, None

        except Exception as e:
            logger.error(f"Error in LoRA validation: {e}", exc_info=True)
            return None, None

    def _load_lora_model(self):
        if self.lora_model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer
            from peft import PeftModel

            logger.info(f"Loading LoRA model: {self.lora_adapter_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.lora_base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.lora_model = PeftModel.from_pretrained(
                base_model, self.lora_adapter_path, torch_dtype=torch.float16
            )
            self.lora_model.eval()
            self.lora_tokenizer = HFAutoTokenizer.from_pretrained(self.lora_base_model)
            logger.info("LoRA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
            logger.warning("Continuing without LoRA validation")
            self.use_lora_validation = False

    def _parse_numbered_list(self, text: str) -> List[str]:
        pattern = r"^\s*\d+\.\s*(.+)$"
        items = []
        for line in text.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                items.append(match.group(1).strip())
        return items

    # =========================================================================
    # Private helpers – evaluation (from EvaluationAgent)
    # =========================================================================

    def _generate_u_explanation(
        self,
        is_unanswerable: bool,
        confidence: float,
        question: str,
        sql_generation_attempted: bool,
        sql_generation_failed: bool,
    ) -> str:
        if is_unanswerable:
            if sql_generation_failed:
                return (
                    f"This question is classified as UNANSWERABLE (confidence: {confidence:.0%}). "
                    f"The database schema does not contain sufficient information to answer this question, "
                    f"and SQL generation failed, confirming it cannot be answered with the available data."
                )
            else:
                return (
                    f"This question is classified as UNANSWERABLE (confidence: {confidence:.0%}). "
                    f"After analysis, the database schema does not provide the necessary information "
                    f"to answer this question."
                )
        else:
            return (
                f"Upon re-evaluation, this question may be ANSWERABLE (confidence: {confidence:.0%}). "
                f"The database schema appears to contain information that could address this question. "
                f"Consider reclassifying to AA (ambiguous) or AU (unambiguous)."
            )

    def _generate_sql_explanation(
        self,
        precision: float,
        recall: float,
        f1_score: float,
        all_found: bool,
        num_predicted: int,
        num_gold: int,
        num_correct: int,
        num_executable: int,
        category: str,
    ) -> str:
        # Kept for backward compatibility; detailed explanation is generated by
        # generate_production_explanation.
        return ""

    def _generate_score_breakdown(
        self,
        precision: float,
        recall: float,
        f1_score: float,
        num_predicted: int,
        num_gold: int,
        num_correct: int,
        num_executable: int,
        execution_errors: List[str],
        precision_flex: float = 0.0,
        recall_flex: float = 0.0,
        f1_flex: float = 0.0,
        num_correct_flex: int = 0,
    ) -> str:
        breakdown = f"""
Score Breakdown (Ground Truth Evaluation):
------------------------------------------

Standard Metrics (Exact Match):
-------------------------------
Precision: {precision:.4f} - {num_correct}/{num_predicted} predicted queries exactly matched gold
Recall:    {recall:.4f} - {num_correct}/{num_gold} gold queries were exactly matched
F1 Score:  {f1_score:.4f} - Harmonic mean of precision and recall

Flex Metrics (Allows Extra Columns):
------------------------------------
Precision Flex: {precision_flex:.4f} - {num_correct_flex}/{num_predicted} predicted queries matched gold (with extra columns allowed)
Recall Flex:    {recall_flex:.4f} - {num_correct_flex}/{num_gold} gold queries were matched (with extra columns allowed)
F1 Flex:        {f1_flex:.4f} - Harmonic mean of flex precision and recall

Improvement from Flex Matching:
-------------------------------
Precision Gain: {precision_flex - precision:+.4f} ({(precision_flex - precision) * 100:+.1f}%)
Recall Gain:    {recall_flex - recall:+.4f} ({(recall_flex - recall) * 100:+.1f}%)
F1 Gain:        {f1_flex - f1_score:+.4f} ({(f1_flex - f1_score) * 100:+.1f}%)

Execution:
----------
Executable: {num_executable}/{num_predicted} queries
"""
        if execution_errors:
            breakdown += f"Errors: {len(execution_errors)} queries failed to execute\n"
            breakdown += "Error samples:\n"
            for i, error in enumerate(execution_errors[:3], 1):
                error_str = str(error) if not isinstance(error, str) else error
                breakdown += f"  {i}. {error_str[:100]}...\n"
        return breakdown

    def _extract_schema_entities(self, db_dump: str) -> Tuple[set, Dict[str, set]]:
        tables = set()
        table_columns: Dict[str, set] = {}
        create_table_pattern = (
            r"CREATE TABLE (?:IF NOT EXISTS )?[\"']?(\w+)[\"']?\s*\((.*?)\);"
        )
        matches = re.findall(create_table_pattern, db_dump, re.IGNORECASE | re.DOTALL)
        for table_name, columns_str in matches:
            tables.add(table_name.lower())
            columns = set()
            col_pattern = (
                r"[\"']?(\w+)[\"']?\s+(?:INTEGER|TEXT|REAL|BLOB|NUMERIC|VARCHAR|CHAR|INT"
                r"|FLOAT|DOUBLE|BOOLEAN|DATE|DATETIME|TIMESTAMP)"
            )
            col_matches = re.findall(col_pattern, columns_str, re.IGNORECASE)
            for col in col_matches:
                columns.add(col.lower())
            table_columns[table_name.lower()] = columns
        return tables, table_columns

    def _extract_sql_entities(self, sql_query: str) -> Tuple[set, set]:
        sql_lower = sql_query.lower()
        tables = set()
        from_pattern = r"from\s+[\"']?(\w+)[\"']?(?:\s+(?:as\s+)?[\"']?\w+[\"']?)?"
        join_pattern = r"join\s+[\"']?(\w+)[\"']?(?:\s+(?:as\s+)?[\"']?\w+[\"']?)?"
        for match in re.finditer(from_pattern, sql_lower):
            tables.add(match.group(1))
        for match in re.finditer(join_pattern, sql_lower):
            tables.add(match.group(1))

        columns = set()
        col_pattern = (
            r"\b(\w+)\s*[=<>!]|\bselect\s+(?:distinct\s+)?(.+?)\s+from"
            r"|\bgroup\s+by\s+(.+?)(?:\s+having|\s+order|\s+limit|$)"
            r"|\border\s+by\s+(.+?)(?:\s+limit|$)"
        )
        for match in re.finditer(col_pattern, sql_lower, re.IGNORECASE):
            for group in match.groups():
                if group:
                    parts = re.split(r"[,\s]+", group.strip())
                    for part in parts:
                        if "(" in part or ")" in part:
                            continue
                        if "." in part:
                            part = part.split(".")[1]
                        part = re.sub(r"[^\w]", "", part)
                        if part and part not in (
                            "as", "distinct", "from", "where", "and", "or"
                        ):
                            columns.add(part)
        return tables, columns

    def _calculate_schema_correctness(
        self, sql_query: str, db_dump: str
    ) -> Tuple[float, List[str]]:
        issues: List[str] = []
        try:
            valid_tables, valid_columns = self._extract_schema_entities(db_dump)
            sql_tables, sql_columns = self._extract_sql_entities(sql_query)

            invalid_tables = sql_tables - valid_tables
            if invalid_tables:
                issues.append(f"Invalid tables: {', '.join(invalid_tables)}")
            table_correctness = (
                len(sql_tables - invalid_tables) / len(sql_tables) if sql_tables else 0.0
            )

            all_valid_columns: set = set()
            for cols in valid_columns.values():
                all_valid_columns.update(cols)

            sql_keywords = {
                "select", "from", "where", "join", "on", "group", "order", "by",
                "having", "limit", "count", "sum", "avg", "max", "min", "desc",
                "asc", "distinct", "as", "and", "or", "not", "in", "like",
                "between", "case", "when", "then", "else", "end",
            }
            invalid_columns = (sql_columns - all_valid_columns) - sql_keywords
            if invalid_columns:
                issues.append(
                    f"Potentially invalid columns: {', '.join(list(invalid_columns)[:5])}"
                )

            if sql_columns:
                sql_columns_filtered = sql_columns - sql_keywords
                column_correctness = (
                    len(sql_columns_filtered - invalid_columns) / len(sql_columns_filtered)
                    if sql_columns_filtered
                    else 1.0
                )
            else:
                column_correctness = 1.0

            return 0.6 * table_correctness + 0.4 * column_correctness, issues

        except Exception as e:
            logger.warning(f"Error calculating schema correctness: {e}")
            issues.append(f"Schema validation error: {str(e)}")
            return 0.5, issues

    def _calculate_semantic_similarity(self, question: str, interpretation: str) -> float:
        try:
            q_emb = self.semantic_model.encode(question, convert_to_tensor=True)
            i_emb = self.semantic_model.encode(interpretation, convert_to_tensor=True)
            return float(util.cos_sim(q_emb, i_emb)[0][0])
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.5

    def calculate_interpretation_quality(
        self,
        question: str,
        interpretation: str,
        sql_query: str,
        db_dump: str,
        db_file: str,
    ) -> InterpretationQualityScore:
        semantic_similarity = self._calculate_semantic_similarity(question, interpretation)
        schema_correctness, issues = self._calculate_schema_correctness(sql_query, db_dump)

        execution_success = False
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            cursor.fetchall()
            conn.close()
            execution_success = True
        except Exception as e:
            issues.append(f"Execution error: {str(e)[:100]}")

        overall_score = (
            semantic_similarity * 4.0
            + schema_correctness * 3.0
            + (1.0 if execution_success else 0.0) * 3.0
        )

        return InterpretationQualityScore(
            interpretation=interpretation,
            sql_query=sql_query,
            semantic_similarity=semantic_similarity,
            schema_correctness=schema_correctness,
            execution_success=execution_success,
            overall_score=overall_score,
            issues=issues,
        )

    def _generate_query_explanation(self, sql_query: str, db_dump: str) -> str:
        try:
            prompt = f"""Given this SQL query and database schema, explain in one clear sentence what this query retrieves.

Database Schema:
{db_dump}

SQL Query:
{sql_query}

Provide a natural language explanation that describes what data this query returns. Be specific about:
- Which table(s) are being queried
- What filtering conditions are applied
- What columns/data are being returned

Respond with ONLY the explanation sentence, no preamble."""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate query explanation: {e}")
            return f"SQL query {sql_query[:50]}..."

    def _generate_production_score_breakdown(
        self, quality_scores: List[InterpretationQualityScore]
    ) -> str:
        if not quality_scores:
            return "No quality scores available."

        avg_overall = sum(s.overall_score for s in quality_scores) / len(quality_scores)
        avg_semantic = sum(s.semantic_similarity for s in quality_scores) / len(quality_scores)
        avg_schema = sum(s.schema_correctness for s in quality_scores) / len(quality_scores)
        num_successful = sum(1 for s in quality_scores if s.execution_success)
        num_with_issues = sum(1 for s in quality_scores if s.issues)

        breakdown = f"""
Quality Score Summary:
---------------------
Total Interpretations: {len(quality_scores)}
Average Quality Score: {avg_overall:.2f}/10.0

Metric Averages:
- Semantic Alignment: {avg_semantic:.2%}
- Schema Correctness: {avg_schema:.2%}
- Execution Success: {num_successful}/{len(quality_scores)} queries

Score Distribution:
- Excellent (9.0-10.0): {sum(1 for s in quality_scores if s.overall_score >= 9.0)}
- Good (7.0-8.9): {sum(1 for s in quality_scores if 7.0 <= s.overall_score < 9.0)}
- Fair (5.0-6.9): {sum(1 for s in quality_scores if 5.0 <= s.overall_score < 7.0)}
- Poor (0.0-4.9): {sum(1 for s in quality_scores if s.overall_score < 5.0)}

Issues Found: {num_with_issues}/{len(quality_scores)} interpretations have issues
"""
        if quality_scores:
            top_score = quality_scores[0]
            breakdown += f"\nBest Interpretation:\n"
            breakdown += f"  Score: {top_score.overall_score:.2f}/10.0\n"
            breakdown += (
                f"  Interpretation: {top_score.interpretation[:100]}"
                f"{'...' if len(top_score.interpretation) > 100 else ''}\n"
            )
        return breakdown
