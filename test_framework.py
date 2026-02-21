#!/usr/bin/env python3
"""
Evaluation framework for the agentic Text-to-SQL pipeline.

Runs the UnifiedSQLAgent on a chosen dataset and split, producing routing
accuracy and SQL generation metrics (precision, recall, F1 — standard and flex).

Supports four datasets: ambrosia, ambiqt, bird, spider.

Usage:
    # Full test run (default: Ambrosia test split)
    python test_framework.py

    # Evaluate a different dataset
    python test_framework.py --dataset ambiqt

    # Quick test with 50 examples, no LoRA
    python test_framework.py --limit 50 --use-lora-validation False

    # Resume an interrupted run
    python test_framework.py --resume --output-path outputs/framework_results.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import pathlib
import time
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
import openai

# Add parent directory for imports
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.unified_agent import UnifiedSQLAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Server URLs
DEFAULT_MODEL_URL       = "http://localhost:8000/v1"
DEFAULT_LORA_MODEL_URL  = "http://localhost:8002/v1"

# Model names
DEFAULT_MODEL_NAME        = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_LORA_ADAPTER_NAME = "llama-grpo-curriculum"

# Model paths (relative to project root)
DEFAULT_DEBERTA_MODEL = "models/deberta-v3-base_diverse_20251115_160058"
DEFAULT_LORA_ADAPTER  = "models/llama_grpo_curriculum/curriculum_20251123_055234/stage4_mixed/model"

# Dataset CSV paths
DATASET_CONFIGS = {
    'ambrosia': {
        'csv_path':     'data/ambrosia/ambrosia_with_unanswerable_validated.csv',
        'rag_csv_path': 'data/ambrosia/ambrosia_with_unanswerable_validated.csv',
    },
    'ambiqt': {
        'csv_path':     'data/AmbiQT/ambiqt_ambrosia_format.csv',
        'rag_csv_path': 'data/AmbiQT/ambiqt_ambrosia_format.csv',
    },
    'bird': {
        'csv_path':     'data/bird/bird_minidev_ambrosia_format.csv',
        'rag_csv_path': 'data/bird/bird_train_ambrosia_format.csv',
    },
    'spider': {
        'csv_path':     'data/spider/spider_ambrosia_format.csv',
        'rag_csv_path': 'data/spider/spider_ambrosia_format.csv',
    },
}

# Evaluation defaults
DEFAULT_RAG_K                      = 5
DEFAULT_RAG_K_OTHER                = 0
DEFAULT_TEMPERATURE                = 0.0
DEFAULT_LORA_TEMPERATURE           = 0.8
DEFAULT_HYBRID_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_BATCH_SIZE                 = 8
DEFAULT_SEED                       = 42
DEFAULT_OUTPUT_PATH                = "outputs/framework_results.json"


class ConnectionError(Exception):
    """Custom exception for connection errors that should stop evaluation."""
    pass


class FrameworkMetrics:
    """Container for framework evaluation metrics."""

    def __init__(self):
        self.total = 0

        # Routing metrics
        self.routing_correct = 0
        self.routing_confusion = defaultdict(lambda: defaultdict(int))

        # Category-specific counts
        # U: Only track accuracy (no SQL metrics - GT=U means unanswerable)
        # AU/AA: Track accuracy and F1 metrics
        self.category_stats = {
            'U': {'total': 0, 'correct': 0},
            'AU': {'total': 0, 'correct': 0, 'avg_f1': 0.0, 'f1_sum': 0.0},
            'AA': {'total': 0, 'correct': 0, 'avg_f1': 0.0, 'f1_sum': 0.0}
        }

        # SQL generation metrics (for AU/AA) - standard exact match
        self.sql_metrics = {
            'total_examples': 0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_f1': 0.0,
            'full_coverage_count': 0,
            'single_coverage_count': 0,  # At least one interpretation found
            'precision_sum': 0.0,
            'recall_sum': 0.0,
            'f1_sum': 0.0
        }

        # SQL generation flex metrics (allows extra columns)
        self.sql_metrics_flex = {
            'avg_precision_flex': 0.0,
            'avg_recall_flex': 0.0,
            'avg_f1_flex': 0.0,
            'full_coverage_flex_count': 0,
            'single_coverage_flex_count': 0,
            'precision_flex_sum': 0.0,
            'recall_flex_sum': 0.0,
            'f1_flex_sum': 0.0
        }

        # Correction metrics
        self.correction_metrics = {
            'correction_applied_count': 0,
            'total_f1_improvement': 0.0,
            'avg_f1_improvement': 0.0
        }

        # Ambiguity type breakdown
        self.by_ambiguity_type = defaultdict(lambda: {
            'count': 0,
            'precision_sum': 0.0,
            'recall_sum': 0.0,
            'f1_sum': 0.0,
            'full_coverage_count': 0,
            'single_coverage_count': 0,
            'precision_flex_sum': 0.0,
            'recall_flex_sum': 0.0,
            'f1_flex_sum': 0.0,
            'full_coverage_count_flex': 0,
            'single_coverage_count_flex': 0
        })

        # Reclassification tracking
        self.reclassified_to_u = 0

        # Detailed results
        self.detailed_results = []

    def add_result(self, result: Dict):
        """Add a single result to the metrics."""
        self.total += 1

        gt_category = result['gt_category']
        pred_category = result['pred_category']

        # Routing accuracy
        routing_correct = (pred_category == gt_category)
        if routing_correct:
            self.routing_correct += 1
        self.routing_confusion[gt_category][pred_category] += 1

        # Category-specific stats
        if gt_category in self.category_stats:
            self.category_stats[gt_category]['total'] += 1
            if routing_correct:
                self.category_stats[gt_category]['correct'] += 1

        # SQL metrics for AU/AA
        if gt_category in ['AU', 'AA']:
            # If answerable question was misclassified as U, count as 0 across the board
            if pred_category == 'U':
                # Misclassified answerable as unanswerable - count as complete failure
                f1 = 0.0
                precision = 0.0
                recall = 0.0
                one_found = False
                f1_flex = 0.0
                precision_flex = 0.0
                recall_flex = 0.0
                one_found_flex = False
            elif 'sql_evaluation' in result:
                sql_eval = result['sql_evaluation']

                # Standard metrics
                f1 = sql_eval.get('f1_score', 0.0)
                precision = sql_eval.get('precision', 0.0)
                recall = sql_eval.get('recall', 0.0)
                one_found = recall > 0  # At least one interpretation found

                # Flex metrics
                f1_flex = sql_eval.get('f1_flex', 0.0)
                precision_flex = sql_eval.get('precision_flex', 0.0)
                recall_flex = sql_eval.get('recall_flex', 0.0)
                one_found_flex = recall_flex > 0
            else:
                # No SQL evaluation available - treat as 0
                f1 = 0.0
                precision = 0.0
                recall = 0.0
                one_found = False
                f1_flex = 0.0
                precision_flex = 0.0
                recall_flex = 0.0
                one_found_flex = False

            # Update aggregated metrics
            self.category_stats[gt_category]['f1_sum'] += f1
            self.sql_metrics['total_examples'] += 1
            self.sql_metrics['precision_sum'] += precision
            self.sql_metrics['recall_sum'] += recall
            self.sql_metrics['f1_sum'] += f1

            if recall >= 1.0:  # all_found when recall is perfect (all gold interpretations found)
                self.sql_metrics['full_coverage_count'] += 1

            if one_found:
                self.sql_metrics['single_coverage_count'] += 1

            # Flex metrics
            self.sql_metrics_flex['precision_flex_sum'] += precision_flex
            self.sql_metrics_flex['recall_flex_sum'] += recall_flex
            self.sql_metrics_flex['f1_flex_sum'] += f1_flex

            if recall_flex >= 1.0:  # all_found_flex when recall is perfect
                self.sql_metrics_flex['full_coverage_flex_count'] += 1

            if one_found_flex:
                self.sql_metrics_flex['single_coverage_flex_count'] += 1

            # Track correction metrics (only if we have sql_eval)
            if pred_category != 'U' and 'sql_evaluation' in result:
                sql_eval = result['sql_evaluation']
                if sql_eval.get('correction_applied', False):
                    self.correction_metrics['correction_applied_count'] += 1
                    f1_improvement = sql_eval.get('f1_improvement', 0.0)
                    self.correction_metrics['total_f1_improvement'] += f1_improvement

            # Track by ambiguity type
            ambiguity_type = result.get('ambiguity_type', 'None')
            stats = self.by_ambiguity_type[ambiguity_type]
            stats['count'] += 1
            stats['precision_sum'] += precision
            stats['recall_sum'] += recall
            stats['f1_sum'] += f1
            if recall >= 1.0:  # all_found when recall is perfect (all gold interpretations found)
                stats['full_coverage_count'] += 1
            if one_found:
                stats['single_coverage_count'] += 1

            # Track flex metrics by ambiguity type
            stats['precision_flex_sum'] += precision_flex
            stats['recall_flex_sum'] += recall_flex
            stats['f1_flex_sum'] += f1_flex
            if recall_flex >= 1.0:
                stats['full_coverage_count_flex'] += 1
            if one_found_flex:
                stats['single_coverage_count_flex'] += 1

        # Track reclassifications
        if result.get('reclassified_to_u', False):
            self.reclassified_to_u += 1

        # Store detailed result
        self.detailed_results.append(result)

    def calculate_averages(self):
        """Calculate average metrics."""
        # SQL averages (standard)
        if self.sql_metrics['total_examples'] > 0:
            n = self.sql_metrics['total_examples']
            self.sql_metrics['avg_precision'] = self.sql_metrics['precision_sum'] / n
            self.sql_metrics['avg_recall'] = self.sql_metrics['recall_sum'] / n
            self.sql_metrics['avg_f1'] = self.sql_metrics['f1_sum'] / n
            self.sql_metrics['full_coverage_rate'] = self.sql_metrics['full_coverage_count'] / n
            self.sql_metrics['single_coverage_rate'] = self.sql_metrics['single_coverage_count'] / n

            # SQL flex averages
            self.sql_metrics_flex['avg_precision_flex'] = self.sql_metrics_flex['precision_flex_sum'] / n
            self.sql_metrics_flex['avg_recall_flex'] = self.sql_metrics_flex['recall_flex_sum'] / n
            self.sql_metrics_flex['avg_f1_flex'] = self.sql_metrics_flex['f1_flex_sum'] / n
            self.sql_metrics_flex['full_coverage_flex_rate'] = self.sql_metrics_flex['full_coverage_flex_count'] / n
            self.sql_metrics_flex['single_coverage_flex_rate'] = self.sql_metrics_flex['single_coverage_flex_count'] / n

        # Correction metrics
        if self.correction_metrics['correction_applied_count'] > 0:
            self.correction_metrics['avg_f1_improvement'] = (
                self.correction_metrics['total_f1_improvement'] /
                self.correction_metrics['correction_applied_count']
            )

        # Category-specific F1 averages
        for category in ['AU', 'AA']:
            if self.category_stats[category]['total'] > 0:
                self.category_stats[category]['avg_f1'] = (
                    self.category_stats[category]['f1_sum'] /
                    self.category_stats[category]['total']
                )

        # Ambiguity type averages
        for ambig_type, stats in self.by_ambiguity_type.items():
            if stats['count'] > 0:
                n = stats['count']
                stats['avg_precision'] = stats['precision_sum'] / n
                stats['avg_recall'] = stats['recall_sum'] / n
                stats['avg_f1'] = stats['f1_sum'] / n
                stats['full_coverage_rate'] = stats['full_coverage_count'] / n
                stats['single_coverage_rate'] = stats['single_coverage_count'] / n
                # Flex metrics
                stats['avg_precision_flex'] = stats['precision_flex_sum'] / n
                stats['avg_recall_flex'] = stats['recall_flex_sum'] / n
                stats['avg_f1_flex'] = stats['f1_flex_sum'] / n
                stats['full_coverage_rate_flex'] = stats['full_coverage_count_flex'] / n
                stats['single_coverage_rate_flex'] = stats['single_coverage_count_flex'] / n

    def print_report(self):
        """Print comprehensive evaluation report."""
        print("\n" + "="*80)
        print("AGENTIC AI FRAMEWORK EVALUATION")
        print("="*80)

        print(f"\nTotal Examples: {self.total}")

        # Overall routing accuracy
        routing_acc = self.routing_correct / self.total if self.total > 0 else 0

        print(f"\n{'='*80}")
        print("ROUTING ACCURACY")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {self.routing_correct}/{self.total} = {routing_acc:.2%}")

        # Per-category breakdown
        print(f"\n{'='*80}")
        print("PER-CATEGORY PERFORMANCE")
        print(f"{'='*80}")

        for cat in ['U', 'AU', 'AA']:
            stats = self.category_stats[cat]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"\n{cat} (Total: {stats['total']})")
                print(f"  Routing Accuracy: {stats['correct']}/{stats['total']} = {acc:.2%}")

                if cat in ['AU', 'AA'] and stats['avg_f1'] > 0:
                    print(f"  Average F1 Score: {stats['avg_f1']:.4f}")

        # SQL generation metrics
        print(f"\n{'='*80}")
        print("SQL GENERATION METRICS (AU + AA)")
        print(f"{'='*80}")
        if self.sql_metrics['total_examples'] > 0:
            print(f"Total Examples: {self.sql_metrics['total_examples']}")
            print(f"\nStandard Metrics (Exact Match):")
            print(f"  Average Precision: {self.sql_metrics['avg_precision']:.4f}")
            print(f"  Average Recall: {self.sql_metrics['avg_recall']:.4f}")
            print(f"  Average F1 Score: {self.sql_metrics['avg_f1']:.4f}")
            print(f"  Full Coverage Rate: {self.sql_metrics.get('full_coverage_rate', 0):.4f} ({self.sql_metrics['full_coverage_count']}/{self.sql_metrics['total_examples']})")
            print(f"  Single Coverage Rate: {self.sql_metrics.get('single_coverage_rate', 0):.4f} ({self.sql_metrics['single_coverage_count']}/{self.sql_metrics['total_examples']})")

            print(f"\nFlex Metrics (Allows Extra Columns):")
            print(f"  Average Precision Flex: {self.sql_metrics_flex['avg_precision_flex']:.4f} (+{self.sql_metrics_flex['avg_precision_flex'] - self.sql_metrics['avg_precision']:+.4f})")
            print(f"  Average Recall Flex: {self.sql_metrics_flex['avg_recall_flex']:.4f} (+{self.sql_metrics_flex['avg_recall_flex'] - self.sql_metrics['avg_recall']:+.4f})")
            print(f"  Average F1 Flex: {self.sql_metrics_flex['avg_f1_flex']:.4f} (+{self.sql_metrics_flex['avg_f1_flex'] - self.sql_metrics['avg_f1']:+.4f})")
            print(f"  Full Coverage Flex Rate: {self.sql_metrics_flex.get('full_coverage_flex_rate', 0):.4f} ({self.sql_metrics_flex['full_coverage_flex_count']}/{self.sql_metrics['total_examples']})")
            print(f"  Single Coverage Flex Rate: {self.sql_metrics_flex.get('single_coverage_flex_rate', 0):.4f} ({self.sql_metrics_flex['single_coverage_flex_count']}/{self.sql_metrics['total_examples']})")
        else:
            print("No SQL generation examples evaluated")

        # Correction metrics
        if self.correction_metrics['correction_applied_count'] > 0:
            print(f"\n{'='*80}")
            print("SQL CORRECTION METRICS")
            print(f"{'='*80}")
            print(f"Corrections Applied: {self.correction_metrics['correction_applied_count']}/{self.sql_metrics['total_examples']}")
            print(f"Average F1 Improvement: {self.correction_metrics.get('avg_f1_improvement', 0):+.4f}")

        # Ambiguity type breakdown
        if self.by_ambiguity_type:
            print(f"\n{'='*80}")
            print("METRICS BY AMBIGUITY TYPE")
            print(f"{'='*80}")
            for ambig_type in sorted(self.by_ambiguity_type.keys()):
                stats = self.by_ambiguity_type[ambig_type]
                print(f"\n{ambig_type} (Count: {stats['count']})")
                print(f"  Avg Precision: {stats.get('avg_precision', 0):.4f}")
                print(f"  Avg Recall: {stats.get('avg_recall', 0):.4f}")
                print(f"  Avg F1: {stats.get('avg_f1', 0):.4f}")
                print(f"  Full Coverage: {stats.get('full_coverage_rate', 0):.4f} ({stats['full_coverage_count']}/{stats['count']})")
                print(f"  Single Coverage: {stats.get('single_coverage_rate', 0):.4f} ({stats['single_coverage_count']}/{stats['count']})")

        # Reclassification stats
        if self.reclassified_to_u > 0:
            print(f"\n{'='*80}")
            print(f"Reclassified to U: {self.reclassified_to_u} examples")

        # Confusion matrix
        print(f"\n{'='*80}")
        print("CONFUSION MATRIX")
        print(f"{'='*80}")
        self._print_confusion_matrix(self.routing_confusion)

        print(f"\n{'='*80}\n")

    def _print_confusion_matrix(self, confusion_matrix):
        """Print confusion matrix."""
        categories = ['U', 'AU', 'AA']
        print(f"\n           Predicted →")
        print(f"Ground Truth ↓  {'   '.join(categories)}")
        for gt in categories:
            counts = [confusion_matrix[gt][pred] for pred in categories]
            print(f"{gt:>12}     {counts[0]:>3}   {counts[1]:>3}   {counts[2]:>3}")


def load_test_data(
    csv_path: str,
    split: str,
    limit: Optional[int] = None,
    exclude_unanswerable: bool = False
) -> List[Dict]:
    """Load test data from CSV."""
    examples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row.get('split') != split:
                continue

            # Determine question_category
            question_type = row.get('question_type', '').strip()
            is_ambiguous = row.get('is_ambiguous', '').strip().upper()
            question_category = row.get('question_category', '').strip()

            if not question_category:
                if question_type == 'unanswerable':
                    question_category = 'U'
                elif is_ambiguous == 'TRUE':
                    question_category = 'AA'
                elif is_ambiguous == 'FALSE':
                    question_category = 'AU'
                else:
                    continue

            # Skip unanswerable questions if requested
            if exclude_unanswerable and question_category == 'U':
                continue

            # Get db_dump_processed
            db_dump_processed = row.get('db_dump_processed', '').strip()
            if not db_dump_processed:
                db_dump_processed = row.get('db_dump', '').strip()

            # Get db_file
            db_file_full = row.get('db_file_full', '').strip()
            if not db_file_full:
                continue  # Skip examples without database file

            # Get gold SQL queries
            gold_queries = row.get('gold_queries', '').strip()
            gold_sql_queries = []
            if gold_queries:
                gold_sql_queries = [
                    ' '.join(q.split())
                    for q in gold_queries.split('\n\n')
                    if q.strip()
                ]

            examples.append({
                'id': f"{split}_{idx}",
                'question': row['question'],
                'db_dump': db_dump_processed,
                'db_file': db_file_full,
                'gt_category': question_category,
                'question_type': row.get('question_type', 'unanswerable' if question_category == 'U' else 'answerable'),
                'ambiguity_type': row.get('ambig_type', 'None'),
                'gold_sql_queries': gold_sql_queries,
                'interpretations': row.get('nl_interpretations', '').strip().split('\n') if row.get('nl_interpretations') else []
            })

            if limit and len(examples) >= limit:
                break

    return examples


def evaluate_single_example(
    example: Dict,
    agent: UnifiedSQLAgent,
    verbose: bool = False,
    generate_explanations: bool = False,
    agent_type: str = 'integrated'
) -> Optional[Dict]:
    """
    Evaluate a single example through the framework.

    Pipeline:
    - Integrated mode: uses the route_question tool (DeBERTa) then generate_sql tool
      - For U: Returns early with unanswerable result (after validation)
      - For AA/AU: Generates SQL queries with appropriate retrieval strategy
    - No-router mode: Treats all questions as AA to enable LoRA validation
      - Relies on reclassification to U if SQL generation fails

    Args:
        example: Question example with db_dump, db_file, gt_category
        agent: UnifiedSQLAgent instance
        verbose: Print detailed logging
        generate_explanations: Generate natural language explanations
        agent_type: 'integrated' or 'no-router'
    """
    try:
        question = example['question']
        db_dump = example['db_dump']
        db_file = example['db_file']
        gt_category = example['gt_category']

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"Ground Truth: {gt_category}")

        # Step 1: generate_sql tool
        # Integrated mode: Let it classify automatically via route_question (category=None)
        # No-router mode: Always treat as AA to enable LoRA (category='AA')
        sql_result = agent.generate_sql(
            question=question,
            db_dump=db_dump,
            db_file=db_file,
            category=None if agent_type == 'integrated' else 'AA'
        )

        if verbose:
            logger.info(f"Classification: {sql_result.category}")
            if sql_result.category != 'U':
                logger.info(f"SQL Generation: {len(sql_result.sql_queries)} queries generated")
                logger.info(f"Execution: {sql_result.num_valid_queries}/{len(sql_result.sql_queries)} successful")

        result = {
            'id': example['id'],
            'question': question,
            'db_dump': db_dump,
            'db_file': db_file,
            'gt_category': gt_category,
            'question_type': example.get('question_type', 'answerable' if gt_category != 'U' else 'unanswerable'),
            'ambiguity_type': 'null' if gt_category == 'AU' else example.get('ambiguity_type', 'None'),
            'pred_category': sql_result.category,
            'routing_confidence': None,  # Not available with integrated classification
            'routing_validated': sql_result.category == 'U',  # U predictions are validated
            'routing_overridden': False,
            'reclassified_to_u': sql_result.reclassified_to_u
        }

        # Step 2: Handle based on classification result
        if sql_result.category == 'U':
            # U question classified and validated by integrated DeBERTa
            # Generate explanation only if requested (no SQL generated)
            if generate_explanations:
                u_explanation = agent.generate_u_explanation(
                    question=question,
                    db_dump=db_dump,
                    routing_reasoning=sql_result.reasoning,
                    sql_generation_attempted=True,  # U validation always attempts SQL
                    sql_generation_failed=True
                )
            else:
                u_explanation = None

            result['u_explanation'] = {
                'explanation': u_explanation,
                'validated': True,  # U predictions are always validated
                'sql_attempted': True,
                'sql_failed': True
            }

            # Add sql_generation section for U questions (empty but with prompts_used for tracking)
            result['sql_generation'] = {
                'num_queries': 0,
                'num_valid': 0,
                'num_lora_queries': 0,
                'interpretations': [],
                'sql_queries': [],
                'lora_interpretations': None,
                'missed_interpretations': None,
                'execution_errors': [],
                'prompts_used': sql_result.prompts_used if sql_result.prompts_used else {}
            }

            if verbose and u_explanation:
                logger.info(f"U Explanation generated: {u_explanation[:100]}...")

            # Still evaluate U predictions (will get F1=0 since no SQL queries generated)
            # Filter out "UNANSWERABLE" from gold queries (can't execute as SQL)
            gold_queries = example.get('gold_sql_queries', [])
            gold_queries_filtered = [q for q in gold_queries if q.strip().upper() != 'UNANSWERABLE']

            sql_evaluation = agent.evaluate_sql_result(
                predicted_queries=[],  # No SQL queries for U questions
                gold_queries=gold_queries_filtered,
                db_file=db_file,
                category='U'
            )

            # Convert evaluation result to dict (same format as AA/AU questions)
            result['sql_evaluation'] = {
                # Standard metrics (exact match) - should all be 0 for U predictions
                'precision': sql_evaluation.precision,
                'recall': sql_evaluation.recall,
                'f1_score': sql_evaluation.f1_score,
                'all_found': sql_evaluation.all_found,

                # Flex metrics (allows extra columns) - should all be 0 for U predictions
                'precision_flex': sql_evaluation.precision_flex,
                'recall_flex': sql_evaluation.recall_flex,
                'f1_flex': sql_evaluation.f1_flex,
                'all_found_flex': sql_evaluation.all_found_flex,

                # No production explanation for U questions (already in u_explanation)
                'explanation': None,
                'score_breakdown': None,

                # Research fields
                'ground_truth_breakdown': sql_evaluation.score_breakdown,
                'gold_sql_queries': gold_queries_filtered,

                # No correction for U questions
                'correction_applied': False,
                'initial_f1': None,
                'f1_improvement': None,

                # No quality scores for U questions (no SQL generated)
                'quality_scores': []
            }

        else:
            # SQL was already generated for AU/AA questions
            if verbose:
                logger.info(f"SQL queries ready for evaluation")

            # Check if reclassified to U
            if sql_result.reclassified_to_u:
                result['pred_category'] = 'U'
                result['reclassified_to_u'] = True
                if verbose:
                    logger.info("Reclassified to U (no valid SQL could be generated)")

            # Evaluate SQL queries
            sql_evaluation = agent.evaluate_sql_result(
                predicted_queries=sql_result.sql_queries,
                gold_queries=example.get('gold_sql_queries', []),
                db_file=db_file,
                category=sql_result.category
            )

            # Generate production-ready explanation with quality scores (only if requested)
            if generate_explanations:
                production_explanation, production_breakdown, quality_scores = agent.generate_production_explanation(
                    question=question,
                    interpretations=sql_result.interpretations,
                    sql_queries=sql_result.sql_queries,
                    db_dump=db_dump,
                    db_file=db_file
                )
            else:
                production_explanation = None
                production_breakdown = None
                quality_scores = []

            # Track correction metrics if correction was applied
            correction_applied = sql_result.original_queries is not None
            initial_f1 = 0.0
            f1_improvement = 0.0

            if correction_applied:
                # Evaluate original queries to get initial metrics
                initial_evaluation = agent.evaluate_sql_result(
                    predicted_queries=sql_result.original_queries,
                    gold_queries=example.get('gold_sql_queries', []),
                    db_file=db_file,
                    category=sql_result.category
                )
                initial_f1 = initial_evaluation.f1_score
                f1_improvement = sql_evaluation.f1_score - initial_f1

                if verbose:
                    logger.info(f"Correction applied: Initial F1={initial_f1:.3f}, Final F1={sql_evaluation.f1_score:.3f}, Improvement={f1_improvement:+.3f}")

            result['sql_generation'] = {
                'num_queries': len(sql_result.sql_queries),
                'num_valid': sql_result.num_valid_queries,
                'num_lora_queries': sql_result.num_lora_sql_queries,
                'interpretations': sql_result.interpretations,
                'sql_queries': sql_result.sql_queries,  # Add generated SQL queries
                'lora_interpretations': sql_result.lora_interpretations,
                'missed_interpretations': sql_result.missed_interpretations,
                'execution_errors': sql_result.execution_errors,
                'prompts_used': sql_result.prompts_used  # Actual prompts sent to LLM
            }

            result['sql_evaluation'] = {
                # Standard metrics (exact match)
                'precision': sql_evaluation.precision,
                'recall': sql_evaluation.recall,
                'f1_score': sql_evaluation.f1_score,
                'all_found': sql_evaluation.all_found,

                # Flex metrics (allows extra columns)
                'precision_flex': sql_evaluation.precision_flex,
                'recall_flex': sql_evaluation.recall_flex,
                'f1_flex': sql_evaluation.f1_flex,
                'all_found_flex': sql_evaluation.all_found_flex,

                # Production fields (no ground truth)
                'explanation': production_explanation,  # Production explanation with quality scores
                'score_breakdown': production_breakdown,  # Production score breakdown (no ground truth)

                # Research fields (requires ground truth)
                'ground_truth_breakdown': sql_evaluation.score_breakdown,  # Keep for research/debugging
                'gold_sql_queries': example.get('gold_sql_queries', []),  # Add gold SQL queries

                # Other metadata
                'correction_applied': correction_applied,
                'initial_f1': initial_f1 if correction_applied else None,
                'f1_improvement': f1_improvement if correction_applied else None,

                # Quality scores
                'quality_scores': [
                    {
                        'interpretation': qs.interpretation,
                        'overall_score': qs.overall_score,
                        'semantic_similarity': qs.semantic_similarity,
                        'schema_correctness': qs.schema_correctness,
                        'execution_success': qs.execution_success,
                        'issues': qs.issues
                    }
                    for qs in quality_scores
                ]
            }

            if verbose:
                logger.info(f"Evaluation: P={sql_evaluation.precision:.3f}, R={sql_evaluation.recall:.3f}, F1={sql_evaluation.f1_score:.3f}")

        return result

    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
            openai.APIConnectionError, openai.APITimeoutError,
            openai.InternalServerError) as e:
        # Re-raise connection/server errors to stop evaluation
        raise ConnectionError(f"Server error evaluating example {example.get('id', 'unknown')}: {e}") from e
    except Exception as e:
        logger.error(f"Error evaluating example {example.get('id', 'unknown')}: {e}", exc_info=True)
        return None


def evaluate_framework(
    examples: List[Dict],
    agent: UnifiedSQLAgent,
    output_path: str,
    batch_size: int = 1,
    existing_results: Optional[Dict[str, Dict]] = None,
    verbose: bool = False,
    generate_explanations: bool = False,
    agent_type: str = 'integrated'
) -> FrameworkMetrics:
    """
    Evaluate framework with concurrent processing and resume support.

    Args:
        examples: List of test examples
        agent: UnifiedSQLAgent instance
        output_path: Path to save results
        batch_size: Number of concurrent requests
        existing_results: Already completed results (for resume)
        verbose: Print detailed information
        generate_explanations: Generate natural language explanations
        agent_type: 'integrated' or 'no-router'

    Returns:
        FrameworkMetrics with results
    """
    # Initialize metrics from existing results if resuming
    if existing_results is None:
        existing_results = {}

    metrics = FrameworkMetrics()

    # Restore existing results
    for result_dict in existing_results.values():
        metrics.add_result(result_dict)

    # Filter out already-completed examples
    completed_ids = set(existing_results.keys())
    remaining_examples = [ex for ex in examples if ex['id'] not in completed_ids]

    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} completed, {len(remaining_examples)} remaining")

    if not remaining_examples:
        logger.info("All examples already completed!")
        return metrics

    def evaluate_one(example: Dict) -> Optional[Dict]:
        """Evaluate single example with error handling."""
        return evaluate_single_example(
            example, agent, verbose=verbose,
            generate_explanations=generate_explanations, agent_type=agent_type
        )

    # Concurrent or sequential evaluation
    if batch_size > 1:
        logger.info(f"Using concurrent evaluation with batch_size={batch_size}")
    else:
        logger.info("Using sequential evaluation")

    try:
        if batch_size > 1:
            # Concurrent evaluation with batching (chunking)
            # Process examples in chunks to avoid overwhelming the system
            import math
            num_batches = math.ceil(len(remaining_examples) / batch_size)

            with tqdm(total=len(remaining_examples), desc="Evaluating", unit="ex") as pbar:
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Process in chunks
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(remaining_examples))
                        batch_examples = remaining_examples[start_idx:end_idx]

                        # Submit this batch
                        futures = {
                            executor.submit(evaluate_one, example): example
                            for example in batch_examples
                        }

                        # Wait for this batch to complete
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                if result is not None:
                                    metrics.add_result(result)
                            except ConnectionError:
                                # Re-raise to stop evaluation
                                raise
                            pbar.update(1)

                        # Save after each batch
                        save_intermediate_results(output_path, metrics)
                        logger.debug(f"Completed batch {batch_idx + 1}/{num_batches}, total: {metrics.total}")

        else:
            # Sequential evaluation
            for example in tqdm(remaining_examples, desc="Evaluating", unit="ex"):
                result = evaluate_one(example)
                if result is not None:
                    metrics.add_result(result)

                # Save after each example
                save_intermediate_results(output_path, metrics)

    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("❌ INTERRUPTED BY USER")
        print(f"Completed {metrics.total}/{len(examples)} examples")
        print("="*80)

        save_intermediate_results(output_path, metrics)
        print(f"Saved progress. Resume with: --resume --output_path {output_path}")
        os._exit(1)

    except ConnectionError as e:
        print("\n" + "="*80)
        print("❌ CONNECTION ERROR - STOPPING EVALUATION")
        print(f"Error: {e}")
        print(f"Completed {metrics.total}/{len(examples)} examples")
        print("="*80)

        save_intermediate_results(output_path, metrics)
        print(f"Saved progress. Resume with: --resume --output_path {output_path}")
        os._exit(1)

    return metrics


def load_existing_results(output_path: str) -> Dict[str, Dict]:
    """Load existing results from output file for resume."""
    if not os.path.exists(output_path):
        return {}

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)

        results = data.get('detailed_results', [])
        logger.info(f"Loaded {len(results)} existing results from {output_path}")

        return {r['id']: r for r in results}
    except Exception as e:
        logger.warning(f"Could not load existing results: {e}")
        return {}


def save_intermediate_results(output_path: str, metrics: FrameworkMetrics):
    """Save intermediate results."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Calculate averages
        metrics.calculate_averages()

        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_examples': metrics.total,
                'routing_accuracy': metrics.routing_correct / metrics.total if metrics.total > 0 else 0,
                'category_stats': metrics.category_stats,
                'sql_metrics': metrics.sql_metrics,
                'sql_metrics_flex': metrics.sql_metrics_flex,
                'by_ambiguity_type': {k: dict(v) for k, v in metrics.by_ambiguity_type.items()},
                'reclassified_to_u': metrics.reclassified_to_u,
                'correction_metrics': metrics.correction_metrics,
                'confusion_matrix': dict(metrics.routing_confusion)
            },
            'detailed_results': metrics.detailed_results
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to save intermediate results: {e}")


def save_final_results(
    output_path: str,
    metrics: FrameworkMetrics,
    config: Dict,
    elapsed_time: float
):
    """Save final evaluation results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate averages
    metrics.calculate_averages()

    output = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'elapsed_time': elapsed_time,
        'summary': {
            'total_examples': metrics.total,
            'routing_accuracy': metrics.routing_correct / metrics.total if metrics.total > 0 else 0,
            'category_stats': metrics.category_stats,
            'sql_metrics': metrics.sql_metrics,
            'sql_metrics_flex': metrics.sql_metrics_flex,
            'by_ambiguity_type': {k: dict(v) for k, v in metrics.by_ambiguity_type.items()},
            'reclassified_to_u': metrics.reclassified_to_u,
            'correction_metrics': metrics.correction_metrics,
            'confusion_matrix': dict(metrics.routing_confusion)
        },
        'detailed_results': metrics.detailed_results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Test Framework for Text-to-SQL Datasets")

    # Dataset selection (NEW!)
    parser.add_argument("--dataset", type=str, default="ambrosia",
                        choices=["ambrosia", "ambiqt", "bird", "spider"],
                        help="Dataset to evaluate (ambrosia, ambiqt, bird, spider)")

    # Data arguments
    parser.add_argument("--csv-path", default=None,
                        help="Path to evaluation CSV (auto-set based on --dataset if not provided)")
    parser.add_argument("--rag-csv-path", default=None,
                        help="Path to RAG training CSV (auto-set based on --dataset if not provided)")
    parser.add_argument("--split", default="test", choices=["test", "dev", "train", "validation"],
                        help="Which split to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--exclude-unanswerable", action="store_true",
                        help="Exclude unanswerable (U) questions from evaluation")

    # Model arguments
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                        help="URL of the vLLM server for SQL generation (Qwen)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Model name for SQL generation")
    parser.add_argument("--lora-model-url", default=DEFAULT_LORA_MODEL_URL,
                        help="Optional URL of separate vLLM server for LoRA model (e.g., http://localhost:8002/v1). "
                             "If not provided, LoRA model will be loaded locally (requires more GPU memory).")
    parser.add_argument("--deberta-model", default=DEFAULT_DEBERTA_MODEL,
                        help="Path to DeBERTa model")
    parser.add_argument("--lora-adapter", default=DEFAULT_LORA_ADAPTER,
                        help="Path to LoRA adapter (for local loading or vLLM server reference)")
    parser.add_argument("--lora-adapter-name", default=DEFAULT_LORA_ADAPTER_NAME,
                        help="Name of the LoRA adapter on the vLLM server (e.g., 'ambrosia_lora', 'ambiqt_lora')")

    # Agent configuration
    parser.add_argument("--rag-k", type=int, default=DEFAULT_RAG_K,
                        help="Number of RAG examples from same category")
    parser.add_argument("--rag-k-other", type=int, default=DEFAULT_RAG_K_OTHER,
                        help="Number of RAG examples from other category")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="LLM temperature")
    parser.add_argument("--lora-temperature", type=float, default=DEFAULT_LORA_TEMPERATURE,
                        help="Optional separate temperature for LoRA validation (defaults to --temperature if not specified)")
    parser.add_argument("--use-sql-correction", action="store_true", default=True,
                        help="Enable SQL self-correction")
    parser.add_argument("--use-lora-validation", type=lambda x: x.lower() == 'true',
                        default=True, metavar='True|False',
                        help="Enable LoRA model for interpretation validation (default: True)")
    parser.add_argument("--hybrid-confidence-threshold", type=float, default=DEFAULT_HYBRID_CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for attachment/scope type-specific retrieval (default: 0.7)")
    parser.add_argument("--agent-type", default="integrated", choices=["integrated", "no-router"],
                        help="Agent mode: 'integrated' uses DeBERTa classification and routing, "
                             "'no-router' treats all questions as AA to enable LoRA (default: integrated)")

    # Evaluation arguments
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH,
                        help="Output file path for results")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of concurrent requests per batch (default=8, 1=sequential)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--generate-explanations", action="store_true", default=False,
                        help="Generate natural language explanations for SQL queries (default: False)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")

    args = parser.parse_args()

    # Auto-configure dataset paths if not explicitly provided
    config = DATASET_CONFIGS[args.dataset]
    if args.csv_path is None:
        args.csv_path = config['csv_path']
    if args.rag_csv_path is None:
        args.rag_csv_path = config['rag_csv_path']

    logger.info(f"=" * 80)
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info(f"CSV Path: {args.csv_path}")
    logger.info(f"RAG CSV Path: {args.rag_csv_path}")
    logger.info(f"=" * 80)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load test set
    logger.info(f"Loading {args.split} split from {args.csv_path}")
    examples = load_test_data(args.csv_path, args.split, args.limit, args.exclude_unanswerable)
    if args.exclude_unanswerable:
        logger.info(f"Loaded {len(examples)} examples (excluding unanswerable)")
    else:
        logger.info(f"Loaded {len(examples)} examples")

    # Load existing results if resuming
    existing_results = {}
    if args.resume:
        logger.info("Resume mode enabled")
        existing_results = load_existing_results(args.output_path)

    # Initialize unified agent
    logger.info(f"Initializing UnifiedSQLAgent (mode: {args.agent_type})...")
    agent = UnifiedSQLAgent(
        model_url=args.model_url,
        model_name=args.model_name,
        lora_model_url=args.lora_model_url,
        lora_adapter_name=args.lora_adapter_name,
        lora_adapter_path=args.lora_adapter,
        csv_path=args.rag_csv_path,
        rag_k=args.rag_k,
        rag_k_other=args.rag_k_other,
        temperature=args.temperature,
        lora_temperature=args.lora_temperature,
        use_sql_correction=args.use_sql_correction,
        use_lora_validation=args.use_lora_validation,
        hybrid_confidence_threshold=args.hybrid_confidence_threshold,
        seed=args.seed,
        deberta_model_path=args.deberta_model,
        validate_u_predictions=(args.agent_type == 'integrated'),
    )
    logger.info("UnifiedSQLAgent initialized successfully!")

    # Run evaluation
    logger.info(f"\nStarting evaluation on {len(examples)} examples...")
    start_time = time.time()

    config = vars(args)

    # Add prompts to config for reproducibility
    from rag.evaluation_ambrosia_prompts_authors import (
        SYSTEM_MESSAGE_SQL_GENERATION,
        USER_PROMPT_SQL_GENERATION_TEMPLATE,
        USER_PROMPT_SQL_CORRECTION_TEMPLATE
    )

    config['prompts'] = {
        'system_message': SYSTEM_MESSAGE_SQL_GENERATION,
        'sql_generation_user_template': USER_PROMPT_SQL_GENERATION_TEMPLATE,
        'sql_correction_user_template': USER_PROMPT_SQL_CORRECTION_TEMPLATE
    }

    metrics = evaluate_framework(
        examples=examples,
        agent=agent,
        output_path=args.output_path,
        batch_size=args.batch_size,
        existing_results=existing_results,
        verbose=args.verbose,
        generate_explanations=args.generate_explanations,
        agent_type=args.agent_type,
    )

    elapsed_time = time.time() - start_time

    # Print report
    metrics.print_report()

    logger.info(f"Evaluation complete! Processed {metrics.total} examples in {elapsed_time:.1f}s ({metrics.total/elapsed_time:.2f} ex/s)")

    # Save final results
    config['elapsed_time'] = elapsed_time
    save_final_results(args.output_path, metrics, config, elapsed_time)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nRouting Accuracy: {metrics.routing_correct}/{metrics.total} = {metrics.routing_correct/metrics.total:.2%}")
    if metrics.sql_metrics['total_examples'] > 0:
        print(f"SQL Generation F1: {metrics.sql_metrics['avg_f1']:.4f}")
        print(f"Full Coverage Rate: {metrics.sql_metrics['full_coverage_count']}/{metrics.sql_metrics['total_examples']} = {metrics.sql_metrics['full_coverage_count']/metrics.sql_metrics['total_examples']:.2%}")
    print(f"\nResults saved to {args.output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
