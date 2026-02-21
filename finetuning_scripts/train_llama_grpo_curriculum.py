#!/usr/bin/env python3
"""
======================================================================================
CURRICULUM GRPO TRAINING FOR AMBIGUITY TYPES
======================================================================================

Train a Llama model with GRPO using curriculum learning over different ambiguity types:
- Stage 1: scope ambiguity
- Stage 2: attachment ambiguity
- Stage 3: vague ambiguity
- Stage 4: mixed (all types together)

Basic usage:
    python finetuning_scripts/train_llama_grpo_curriculum.py

With custom settings:
    python finetuning_scripts/train_llama_grpo_curriculum.py   
    --reward_model_type openai   \
    --reward_model_name gpt-5.2-2025-12-11  \
    --output_dir output_path    \
    --vllm_gpu_memory_utilization 0.25 \
    --ambiguous_only \
    --balanced_sampling


With custom settings, continuing from a previous checkpoint:
    python finetuning_scripts/train_llama_grpo_curriculum.py   \
    --reward_model_type openai   \
    --reward_model_name gpt-5.2-2025-12-11  \
    --output_dir models/llama_grpo_curriculum/curriculum_gpt-5_2-2025-12-11_20260214_053301   \
    --skip_stages scope,attachment,vague   \
    --resume_from_checkpoint models/llama_grpo_curriculum/curriculum_gpt-5_2-2025-12-11_20260214_053301/stage3_vague/model  \
    --vllm_gpu_memory_utilization 0.25 \
    --ambiguous_only \
    --balanced_sampling

Key arguments:
    --data_path             Path to CSV data file
    --ambiguous_only        Only train on ambiguous questions
    --epochs_per_stage      Number of epochs per curriculum stage (default: 3)
    --curriculum_order      Order of ambiguity types (default: scope,attachment,vague,mixed)

Reward model backends (--reward_model_type):
    claude   Use Anthropic Claude via the Anthropic SDK (default)
             --anthropic_api_key  OR set ANTHROPIC_API_KEY env var
             --reward_model_name  e.g. claude-sonnet-4-5-20250929

    openai   Use OpenAI models via the OpenAI SDK
             --openai_api_key  OR set OPENAI_API_KEY env var
             --reward_model_name  e.g. gpt-4o

    vllm     Use any model served by a vLLM server (OpenAI-compatible endpoint)
             --reward_base_url  e.g. http://localhost:8000/v1
             --reward_model_name  model name as registered in the vLLM server
             --openai_api_key  if the server requires authentication (optional)

Examples:
    # Claude (default)
    python train_llama_grpo_curriculum.py

    # GPT-4o as reward model
    python train_llama_grpo_curriculum.py \
        --reward_model_type openai \
        --reward_model_name gpt-4o

    # Local vLLM server (e.g. Qwen-72B or Llama-3.1-70B)
    python train_llama_grpo_curriculum.py \
        --reward_model_type vllm \
        --reward_base_url http://localhost:8000/v1 \
        --reward_model_name Qwen/Qwen2.5-72B-Instruct

======================================================================================
"""

import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import csv
import json
import random
import argparse
import re
import gc
import time
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

try:
    import anthropic
except ImportError:
    print("Warning: anthropic package not installed. Install with: pip install anthropic")
    anthropic = None

try:
    import openai
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")
    openai = None

import logging
for name in ["vllm", "vllm.engine", "vllm.api_server", "uvicorn", "uvicorn.error", "uvicorn.access"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False


def remove_insert_statements(db_dump: str) -> str:
    """Remove INSERT INTO statements from db_dump to reduce token usage."""
    lines = db_dump.split('\n')
    filtered_lines = [line for line in lines if not line.strip().upper().startswith('INSERT INTO')]
    return '\n'.join(filtered_lines)


def load_interpretation_data(csv_path: str, split: str = 'train', ambiguous_only: bool = True,
                            ambiguity_types: List[str] = None,
                            max_samples: int = None) -> List[Dict]:
    """
    Load interpretation generation data from CSV, optionally filtered by ambiguity type.

    Args:
        csv_path: Path to CSV file
        split: Data split to load
        ambiguous_only: Only load ambiguous questions
        ambiguity_types: List of ambiguity types to include (e.g., ['scope', 'attachment', 'vague'])
                        If None or empty, include all types.
        max_samples: Maximum samples to return (for balanced sampling)

    Returns:
        List of dictionaries with 'db_dump', 'question', 'interpretations', 'ambig_type' fields
    """
    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split', '').strip() != split:
                continue

            question = row.get('question', '').strip()
            is_ambiguous = row.get('is_ambiguous', '').strip().upper() == 'TRUE'
            question_type = row.get('question_type', '').strip()

            if not question:
                continue

            if ambiguous_only and not is_ambiguous:
                continue

            nl_interpretations = row.get('nl_interpretations', '').strip()
            if not nl_interpretations:
                continue

            interpretations = [interp.strip() for interp in nl_interpretations.split('\n') if interp.strip()]

            db_dump = row.get('db_dump_processed', '').strip()
            if not db_dump:
                db_dump = ""

            question_category = row.get('question_category', '').strip()
            if not question_category:
                if question_type == 'unanswerable':
                    question_category = 'U'
                elif is_ambiguous:
                    question_category = 'AA'
                else:
                    question_category = 'AU'

            ambig_type = row.get('ambig_type', '').strip().lower()
            if not ambig_type:
                ambig_type = 'unknown'

            # Filter by ambiguity type if specified
            if ambiguity_types and ambig_type not in ambiguity_types:
                continue

            data.append({
                'db_dump': db_dump,
                'question': question,
                'interpretations': interpretations,
                'num_interpretations': len(interpretations),
                'question_category': question_category,
                'ambig_type': ambig_type
            })

    # Apply max_samples limit with shuffling
    if max_samples and len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    return data


def balanced_sample_by_type(data_by_type: Dict[str, List[Dict]], samples_per_type: int) -> Dict[str, List[Dict]]:
    """
    Balance sampling across ambiguity types.

    Args:
        data_by_type: Dictionary mapping ambiguity type to list of examples
        samples_per_type: Target number of samples per type

    Returns:
        Dictionary with balanced samples per type
    """
    balanced = {}
    for ambig_type, items in data_by_type.items():
        if ambig_type in ['scope', 'attachment', 'vague']:
            n_samples = min(samples_per_type, len(items))
            balanced[ambig_type] = random.sample(items, n_samples)
        else:
            # Keep other types as-is (e.g., 'unknown')
            balanced[ambig_type] = items
    return balanced


def create_grpo_dataset(data: List[Dict], tokenizer, num_samples: int = None) -> Dataset:
    """Create a dataset in the format expected by TRL's GRPOTrainer."""
    data = data.copy()
    random.shuffle(data)

    if num_samples:
        data = data[:num_samples]

    dataset_dict = {
        "prompt": [],
        "ground_truth_interpretations": [],
        "num_ground_truth": [],
        "db_dump": [],
        "question": [],
        "ambig_type": [],
    }

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
- Avoid synonym variations - focus on semantic differences

Output format: Return ONLY a JSON array of strings. No explanations, no prose, no markdown code blocks.
Example: ["interpretation 1", "interpretation 2", "interpretation 3"]"""

    for item in data:
        db_dump_clean = remove_insert_statements(item['db_dump'])

        user_message = f"""Given:

Database Schema:
{db_dump_clean}

Ambiguous Question: {item['question']}

Task: List every distinct way the question could be understood.

Each interpretation should:
1. **Be grounded in the provided schema**
2. Represent a different valid meaning (not synonym variations)
3. Be specific enough to translate into SQL
5. Provide the CORRECT information to the question asked

Return ONLY a valid JSON array of strings. No other text, no explanations, no markdown formatting."""

        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        dataset_dict["prompt"].append(prompt)
        dataset_dict["ground_truth_interpretations"].append(item['interpretations'])
        dataset_dict["num_ground_truth"].append(len(item['interpretations']))
        dataset_dict["db_dump"].append(item['db_dump'])
        dataset_dict["question"].append(item['question'])
        dataset_dict["ambig_type"].append(item['ambig_type'])

    return Dataset.from_dict(dataset_dict)


_REWARD_SYSTEM_PROMPT = """You are an expert evaluator of database question interpretations. Your task is to score how well generated interpretations match ground-truth (GT) interpretations.

INPUT:
1. Database schema
2. Ambiguous question
3. Generated interpretations (candidate set)
4. Ground-truth interpretations (GT set with N items)

UNDERSTANDING GT INTERPRETATIONS:
The GT interpretations represent semantically distinct meanings due to linguistic ambiguity:

1. SCOPE AMBIGUITY: Quantifier scope ("each", "every", "all") is unclear
   - Example: "Show me the toys every cat has."
   - Collective interpretation (property shared by ALL entities together): "List the toys that all cats have in common."
   - Distributive interpretation (property of EACH entity individually): "For each cat, list their toys."

2. ATTACHMENT AMBIGUITY: "X and Y [modifier]" - modifier attachment is unclear
   - Example: "Give me dogs and cats who are orange. Show them in one table."
   - Broad scope (modifier applies to BOTH X and Y): "Select all dogs who are orange AND cats who are orange."
   - Narrow scope (modifier applies ONLY to Y): "Select ALL dogs (no color filter) AND cats who are orange."

3. VAGUENESS: Underspecified terms have multiple possible referents in the schema
   - Example 1: "What technology does Rio know?"
     - Interpretation 1: "What software does Rio know?"
     - Interpretation 2: "What programming languages does Rio know?"
     - Interpretation 3: "What software and programming languages does Rio know?"

   - Example 2: "What was the focus of the conference?"
     - Interpretation 1: "What was the topic of the conference?"
     - Interpretation 2: "What was the theme track of the conference?"
     - Interpretation 3: "What was the theme track and topic of the conference?"

   IMPORTANT: The GT interpretations specify WHICH schema element(s) the vague term maps to.
   Generated interpretations MUST match these specific schema mappings, not just paraphrase the original vague term.

EVALUATION CRITERIA:

VALIDITY CHECK (for each generated interpretation):
- Schema-grounded: references actual tables/columns from the schema
- Executable: specific enough to translate to SQL unambiguously
- Semantically distinct: would produce different SQL query or result set from other interpretations

MATCHING CHECK:
- A generated interpretation MATCHES a GT interpretation if both would lead to the same SQL query
- Paraphrases/synonyms that yield identical SQL are the SAME interpretation (not distinct)

SCORING FRAMEWORK:

Note: N = number of GT interpretations
Note: Recall is MOST important. Precision matters less - a few extras are acceptable.

Step 1 - RECALL: How many of the N GT meanings were captured?
- Count exact matches (interpretations that capture the EXACT same semantic meaning as GT)
- This is the PRIMARY scoring factor

Step 2 - DISTINCTNESS: Are generated interpretations truly distinct from each other?
- Duplicates: multiple generated interpretations that map to the same SQL
- This is a MODERATE penalty

Step 3 - VALIDITY: Are interpretations schema-grounded and executable?
- Invalid: references non-existent tables/columns or too vague to translate to SQL
- This is a MODERATE penalty

Step 4 - PRECISION: How many extras beyond GT?
- One extra is acceptable - MINOR penalty
- A few extras (2-3) suggest lack of focus - MODERATE penalty
- Extreme extras (4+) suggest poor understanding - HEAVY penalty

SCORING RUBRIC (0.0 to 1.0):

0.95-1.00: Perfect
- Captured all N GT meanings
- All valid and distinct
- 0-1 extra interpretations
- Example: GT=2, Generated=2-3 (all valid, all distinct, all GT captured)

0.80-0.90: Excellent
- Captured all N GT meanings (perfect recall!)
- Minor issues: 1-2 duplicates OR extras
- Example: GT=2, Generated=3-4 but captured both GT meanings

0.65-0.75: Good
- Captured N-1 GT meanings (missing just one)
- 1-2 duplicates or extras, but mostly valid
- Example: GT=3, Generated=4, captured 2 GT meanings

0.45-0.60: Fair
- Captured roughly half of GT meanings
  - For N=2: captured 1
  - For N=3: captured 1-2
- May have 2-3 duplicates, extras, or validity issues

0.25-0.40: Poor
- Captured few GT meanings (0-1 when N=2-3)
- Multiple issues: duplicates, invalids, or extreme over-generation (7+)

0.00-0.20: Very Poor
- Captured 0 GT meanings OR
- All/mostly invalid interpretations OR
- Fundamental misunderstanding

CRITICAL OUTPUT FORMAT:
- **RETURN ONLY A SINGLE NUMBER**
- No explanation, no text, no additional characters - just the number.

Examples of valid output: 0.85  or  1.0  or  0.42"""


def _parse_score(generated_text: str) -> float:
    """Parse a 0.0-1.0 score from model output using multiple fallback strategies."""
    score = None

    # Strategy 1: first token
    try:
        score = float(generated_text.split()[0])
        score = max(0.0, min(1.0, score))
        return score
    except (ValueError, IndexError):
        pass

    # Strategy 2: any decimal in [0, 1] range (scan from end)
    numbers = re.findall(r'\b(0?\.\d+|1\.0+|0|1)\b', generated_text)
    for num_str in reversed(numbers):
        try:
            candidate = float(num_str)
            if 0.0 <= candidate <= 1.0:
                return candidate
        except ValueError:
            continue

    # Strategy 3: last line
    last_line = generated_text.strip().split('\n')[-1]
    numbers = re.findall(r'\b(0?\.\d+|1\.0+|0|1)\b', last_line)
    if numbers:
        try:
            return max(0.0, min(1.0, float(numbers[-1])))
        except ValueError:
            pass

    print(f"Warning: Could not parse reward score from: {generated_text[:200]}...")
    return 0.0


class ClaudeRewardModel:
    """Reward model backed by Anthropic Claude (claude-* models)."""

    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929", api_key: str = None):
        if anthropic is None:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        print(f"Reward model initialised: Claude ({self.model_name})")

    def get_reward(self, db_dump: str, question: str, generated_interpretations: str,
                   ground_truth_interpretations: List[str]) -> float:
        db_dump = remove_insert_statements(db_dump)
        user_message = _build_reward_user_message(db_dump, question, generated_interpretations,
                                                   ground_truth_interpretations)
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=250,
                temperature=0.0,
                system=_REWARD_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            return _parse_score(response.content[0].text.strip())
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return 0.0


class OpenAIRewardModel:
    """Reward model backed by OpenAI API (gpt-* models)."""

    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        print(f"Reward model initialised: OpenAI ({self.model_name})")

    def get_reward(self, db_dump: str, question: str, generated_interpretations: str,
                   ground_truth_interpretations: List[str]) -> float:
        db_dump = remove_insert_statements(db_dump)
        user_message = _build_reward_user_message(db_dump, question, generated_interpretations,
                                                   ground_truth_interpretations)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_completion_tokens=250,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _REWARD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            )
            return _parse_score(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return 0.0


class VLLMRewardModel:
    """Reward model backed by a vLLM server via its OpenAI-compatible endpoint.

    Works with any model served by vLLM, e.g.:
        vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
        vllm serve meta-llama/Llama-3.1-70B-Instruct --port 8000
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "token-abc123"):
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("VLLM_API_KEY", "token-abc123"),
        )
        self.model_name = model_name
        print(f"Reward model initialised: vLLM ({self.model_name}) at {base_url}")

    def get_reward(self, db_dump: str, question: str, generated_interpretations: str,
                   ground_truth_interpretations: List[str]) -> float:
        db_dump = remove_insert_statements(db_dump)
        user_message = _build_reward_user_message(db_dump, question, generated_interpretations,
                                                   ground_truth_interpretations)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=250,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _REWARD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            )
            return _parse_score(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error calling vLLM server: {e}")
            return 0.0


def _build_reward_user_message(db_dump: str, question: str, generated_interpretations: str,
                                ground_truth_interpretations: List[str]) -> str:
    """Build the user message for reward model evaluation."""
    gt_lines = chr(10).join(f'{i+1}. {interp}' for i, interp in enumerate(ground_truth_interpretations))
    return (
        f"Database Schema:\n{db_dump}\n\n"
        f"Ambiguous Question: {question}\n\n"
        f"Ground Truth Interpretations ({len(ground_truth_interpretations)} valid interpretations):\n"
        f"{gt_lines}\n\n"
        f"Generated Interpretations:\n{generated_interpretations}\n\n"
        f"Respond with ONLY a single number between 0.0 and 1.0."
    )


_reward_model = None


def initialize_reward_model(
    reward_model_type: str = "claude",
    model_name: str = None,
    api_key: str = None,
    base_url: str = "http://localhost:8000/v1",
    vllm_api_key: str = None,
):
    """Initialise the global reward model.

    Args:
        reward_model_type: One of "claude", "openai", or "vllm".
        model_name:        Override the default model name for the chosen backend.
        api_key:           API key for Claude or OpenAI (or set via env var).
        base_url:          Base URL for the vLLM server (only used when type="vllm").
        vllm_api_key:      API key / token for the vLLM server (only used when type="vllm").
    """
    global _reward_model
    if reward_model_type == "claude":
        _reward_model = ClaudeRewardModel(
            model_name=model_name or "claude-sonnet-4-5-20250929",
            api_key=api_key,
        )
    elif reward_model_type == "openai":
        _reward_model = OpenAIRewardModel(
            model_name=model_name or "gpt-4o",
            api_key=api_key,
        )
    elif reward_model_type == "vllm":
        if not model_name:
            raise ValueError("--reward_model_name is required when --reward_model_type=vllm")
        _reward_model = VLLMRewardModel(
            model_name=model_name,
            base_url=base_url,
            api_key=vllm_api_key,
        )
    else:
        raise ValueError(f"Unknown reward_model_type '{reward_model_type}'. "
                         f"Choose from: claude, openai, vllm")


def quality_reward_func(prompts, completions, ground_truth_interpretations,
                       db_dump, question, **kwargs) -> List[float]:
    """Main reward function using Claude to evaluate interpretation quality."""
    global _reward_model

    if _reward_model is None:
        raise RuntimeError("Reward model not initialized.")

    rewards = []
    parsing_errors = 0

    for i in range(len(completions)):
        generated_text = completions[i][0]['content'].strip()

        # Print header
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{len(completions)}")
        print(f"{'='*80}")
        print(f"Question: {question[i]}")
        print(f"\nGround Truth Interpretations ({len(ground_truth_interpretations[i])}):")
        for idx, gt in enumerate(ground_truth_interpretations[i], 1):
            print(f"  {idx}. {gt}")

        print(f"\nGenerated Interpretations:")
        print(generated_text)

        # Check if it's valid JSON before sending to Claude
        # Remove markdown code blocks if present
        text_to_parse = generated_text
        if text_to_parse.startswith('```'):
            lines = text_to_parse.split('\n')
            text_to_parse = '\n'.join(lines[1:-1]) if len(lines) > 2 else text_to_parse
            text_to_parse = text_to_parse.strip()

        try:
            parsed = json.loads(text_to_parse)
            if not isinstance(parsed, list):
                raise ValueError("Output is not a JSON array")

            # Valid JSON - send to Claude for evaluation
            reward = _reward_model.get_reward(
                db_dump=db_dump[i],
                question=question[i],
                generated_interpretations=generated_text,
                ground_truth_interpretations=ground_truth_interpretations[i]
            )
            print(f"\n→ Quality Score: {reward:.3f} (from reward model evaluation)")
            rewards.append(reward)

        except (json.JSONDecodeError, ValueError) as e:
            # Parsing error - give fixed score of 0.25
            parsing_errors += 1
            print(f"\n→ PARSING ERROR: {str(e)[:100]}")
            print(f"→ Quality Score: 0.25 (default for parsing errors)")
            rewards.append(0.25)

        print(f"{'='*80}\n")

    # Print summary for this batch
    if rewards:
        valid_rewards = [r for r in rewards if r != 0.25]
        if valid_rewards:
            print(f"\n[Quality Rewards Summary]")
            print(f"  Valid completions: min={min(valid_rewards):.3f}, max={max(valid_rewards):.3f}, mean={np.mean(valid_rewards):.3f}")
        print(f"  Overall (including parsing errors): min={min(rewards):.3f}, max={max(rewards):.3f}, mean={np.mean(rewards):.3f}")
        if parsing_errors > 0:
            print(f"  Parsing errors: {parsing_errors}/{len(completions)} (assigned score 0.25)")

    return rewards


def format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if interpretations are properly formatted as JSON array."""
    import json
    rewards = []

    for completion in completions:
        generated_text = completion[0]['content'].strip()

        if generated_text.startswith('```'):
            lines = generated_text.split('\n')
            generated_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else generated_text
            generated_text = generated_text.strip()

        try:
            parsed = json.loads(generated_text)
            if isinstance(parsed, list) and len(parsed) >= 2:
                if all(isinstance(item, str) for item in parsed):
                    substantial = [item for item in parsed if len(item) >= 50]
                    if len(substantial) >= 2:
                        rewards.append(0.1)
                    else:
                        rewards.append(0.05)
                else:
                    rewards.append(0.01)
            else:
                rewards.append(0.0)
        except (json.JSONDecodeError, ValueError):
            rewards.append(0.0)

    return rewards


class CurriculumStageCallback(TrainerCallback):
    """Callback to track progress within a curriculum stage."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.epoch_rewards = []
        self.current_epoch_rewards = []
        self.current_epoch_quality_rewards = []
        self.last_completed_epoch = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        reward = logs.get('reward', None)
        if reward is None:
            reward = logs.get('train/reward', None) or logs.get('train_reward', None)
        if reward is not None:
            self.current_epoch_rewards.append(reward)

        # Track quality rewards separately
        quality_reward = logs.get('rewards/quality_reward_func/mean', None)
        if quality_reward is not None:
            self.current_epoch_quality_rewards.append(quality_reward)

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        if current_epoch <= self.last_completed_epoch:
            return
        self.last_completed_epoch = current_epoch

        if self.current_epoch_rewards:
            mean_reward = np.mean(self.current_epoch_rewards)
            self.epoch_rewards.append(mean_reward)
            print(f"\n[{self.stage_name}] Epoch {current_epoch} average reward: {mean_reward:.4f}")

        if self.current_epoch_quality_rewards:
            min_quality = min(self.current_epoch_quality_rewards)
            max_quality = max(self.current_epoch_quality_rewards)
            mean_quality = np.mean(self.current_epoch_quality_rewards)
            print(f"[{self.stage_name}] Epoch {current_epoch} quality rewards - min: {min_quality:.4f}, max: {max_quality:.4f}, mean: {mean_quality:.4f}")

        self.current_epoch_rewards = []
        self.current_epoch_quality_rewards = []

    def on_train_end(self, args, state, control, **kwargs):
        if self.epoch_rewards:
            print(f"\n[{self.stage_name}] Stage complete. Best reward: {max(self.epoch_rewards):.4f}")


def train_curriculum_stage(
    stage_name: str,
    train_data: List[Dict],
    model,
    tokenizer,
    base_output_dir: str,
    args,
    stage_num: int,
    epochs_override: int = None,
) -> None:
    """Train a single curriculum stage."""

    print("\n" + "="*80)
    print(f"CURRICULUM STAGE {stage_num}: {stage_name.upper()}")
    print("="*80)
    print(f"Training samples: {len(train_data)}")

    # Re-seed before dataset shuffle so each stage is reproducible
    random.seed(args.seed + stage_num)
    np.random.seed(args.seed + stage_num)

    # Create dataset for this stage
    train_dataset = create_grpo_dataset(train_data, tokenizer, args.num_train_samples)
    print(f"Dataset size: {len(train_dataset)}")

    # Stage output directory
    stage_output_dir = os.path.join(base_output_dir, f"stage{stage_num}_{stage_name}")
    os.makedirs(stage_output_dir, exist_ok=True)

    # Use epochs_override if provided, otherwise use args.epochs_per_stage
    num_epochs = epochs_override if epochs_override is not None else args.epochs_per_stage

    # Configure GRPO for this stage
    training_args = GRPOConfig(
        output_dir=stage_output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type='cosine',
        logging_steps=args.logging_steps,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=num_epochs,
        save_strategy='epoch',
        save_total_limit=2,
        max_grad_norm=args.max_grad_norm,
        log_on_each_node=False,
        report_to=["tensorboard"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        temperature=0.8,
        load_best_model_at_end=False,
        metric_for_best_model="reward",
        greater_is_better=True,
    )

    # Setup callback
    stage_callback = CurriculumStageCallback(stage_name)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, quality_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[stage_callback],
    )

    # Train this stage
    trainer.train()

    # Save stage checkpoint
    stage_model_dir = os.path.join(stage_output_dir, "model")
    print(f"\nSaving stage {stage_num} model to {stage_model_dir}...")
    trainer.save_model(stage_model_dir)

    return stage_callback.epoch_rewards, trainer


def main():
    parser = argparse.ArgumentParser(description='Curriculum GRPO Training for Ambiguity Types')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='data/ambrosia/ambrosia_with_unanswerable_validated.csv',
                        help='Path to CSV file with splits')
    parser.add_argument('--ambiguous_only', action='store_true', default=False,
                        help='Only use ambiguous questions')

    # Curriculum arguments
    parser.add_argument('--curriculum_order', type=str, default='scope,attachment,vague,mixed',
                        help='Comma-separated order of ambiguity types for curriculum')
    parser.add_argument('--epochs_per_stage', type=int, default=3,
                        help='Number of epochs per curriculum stage')
    parser.add_argument('--scope_epochs', type=int, default=2,
                        help='Number of epochs for scope stage (default: 2)')
    parser.add_argument('--attachment_epochs', type=int, default=2,
                        help='Number of epochs for attachment stage (default: 2)')
    parser.add_argument('--vague_epochs', type=int, default=3,
                        help='Number of epochs for vague stage (overrides --epochs_per_stage if set)')
    parser.add_argument('--mixed_epochs', type=int, default=3,
                        help='Number of epochs for mixed stage (overrides --epochs_per_stage if set)')
    parser.add_argument('--skip_stages', type=str, default='',
                        help='Comma-separated stages to skip (e.g., "mixed")')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to LoRA checkpoint to resume from (e.g., curriculum_xxx/stage2_attachment/model)')
    parser.add_argument('--balanced_sampling', action='store_true', default=False,
                        help='Balance samples across ambiguity types')
    parser.add_argument('--samples_per_type', type=int, default=100,
                        help='Number of samples per ambiguity type when using balanced sampling')

    # Model arguments
    parser.add_argument('--model_name', type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Trainable model name (HuggingFace hub or local path)')

    # Reward model arguments
    parser.add_argument('--reward_model_type', type=str, default='claude',
                        choices=['claude', 'openai', 'vllm'],
                        help='Reward model backend: claude | openai | vllm (default: claude)')
    parser.add_argument('--reward_model_name', type=str, default=None,
                        help='Reward model name/id. Defaults: claude=claude-sonnet-4-5-20250929, '
                             'openai=gpt-4o. Required for vllm.')
    parser.add_argument('--reward_base_url', type=str, default='http://localhost:8000/v1',
                        help='Base URL for vLLM server (only used with --reward_model_type=vllm)')
    parser.add_argument('--anthropic_api_key', type=str, default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--vllm_api_key', type=str, default=None,
                        help='API key/token for the vLLM server (or set VLLM_API_KEY env var)')

    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')

    # Training arguments
    parser.add_argument('--output_dir', type=str,
                        default='models/llama_grpo_curriculum',
                        help='Output directory')
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Number of training samples per stage (None = all)')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_generations', type=int, default=4,
                        help='Number of samples per prompt (GRPO group size)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--max_prompt_length', type=int, default=4096,
                        help='Maximum prompt length')
    parser.add_argument('--max_completion_length', type=int, default=1024,
                        help='Maximum completion length')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.25,
                        help='Fraction of GPU memory vLLM may use for KV cache (default: 0.25). '
                             'Lower this if you get OOM errors between curriculum stages.')
    parser.add_argument('--max_grad_norm', type=float, default=0.25,
                        help='Maximum gradient norm')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio')
    parser.add_argument('--logging_steps', type=int, default=5,
                        help='Log every N steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Parse curriculum order
    curriculum_stages = [s.strip().lower() for s in args.curriculum_order.split(',')]
    skip_stages = set(s.strip().lower() for s in args.skip_stages.split(',') if s.strip())
    curriculum_stages = [s for s in curriculum_stages if s not in skip_stages]

    # Set random seeds before any data loading or sampling
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup output directory
    # If resuming from checkpoint and output_dir looks like a curriculum dir, use it as-is
    if args.resume_from_checkpoint and 'curriculum_' in os.path.basename(args.output_dir):
        # Use the provided directory directly (resuming into existing run)
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # Create new timestamped directory that includes a sanitized reward model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reward_model_slug = (args.reward_model_name or args.reward_model_type).replace("/", "-").replace(".", "_")
        args.output_dir = os.path.join(args.output_dir, f"curriculum_{reward_model_slug}_{timestamp}")
        os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("CURRICULUM GRPO TRAINING")
    print("="*80)
    print(f"Trainable model: {args.model_name}")
    reward_label = args.reward_model_name or {"claude": "claude-sonnet-4-5-20250929", "openai": "gpt-4o"}.get(args.reward_model_type, "?")
    print(f"Reward model: {args.reward_model_type} / {reward_label}")
    if args.reward_model_type == "vllm":
        print(f"vLLM server: {args.reward_base_url}")
    print(f"Curriculum stages: {' -> '.join(curriculum_stages)}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    if args.balanced_sampling:
        print(f"Balanced sampling: {args.samples_per_type} samples per type")
    print(f"Output directory: {args.output_dir}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print("="*80)

    # Load all data once
    print("\nLoading data...")
    all_data = load_interpretation_data(
        args.data_path,
        split='train',
        ambiguous_only=args.ambiguous_only,
        ambiguity_types=None  # Load all types
    )

    # Group data by ambiguity type
    data_by_type = {}
    for item in all_data:
        ambig_type = item['ambig_type']
        if ambig_type not in data_by_type:
            data_by_type[ambig_type] = []
        data_by_type[ambig_type].append(item)

    print(f"\nData distribution by ambiguity type (before balancing):")
    for ambig_type, items in sorted(data_by_type.items()):
        print(f"  {ambig_type}: {len(items)} examples")

    # Apply balanced sampling if requested
    if args.balanced_sampling:
        print(f"\nApplying balanced sampling: {args.samples_per_type} samples per type")
        data_by_type = balanced_sample_by_type(data_by_type, args.samples_per_type)
        # Update all_data to reflect balanced sampling
        all_data = []
        for items in data_by_type.values():
            all_data.extend(items)
        print(f"\nData distribution after balancing:")
        for ambig_type, items in sorted(data_by_type.items()):
            print(f"  {ambig_type}: {len(items)} examples")
        print(f"Total samples after balancing: {len(all_data)}")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Initialize reward model
    print(f"\nInitializing reward model (type={args.reward_model_type})...")
    # Pick the right API key depending on backend
    _api_key = args.anthropic_api_key if args.reward_model_type == "claude" else args.openai_api_key
    initialize_reward_model(
        reward_model_type=args.reward_model_type,
        model_name=args.reward_model_name,
        api_key=_api_key,
        base_url=args.reward_base_url,
        vllm_api_key=args.vllm_api_key,
    )

    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA or load from checkpoint
    if args.resume_from_checkpoint:
        print(f"\nLoading LoRA from checkpoint: {args.resume_from_checkpoint}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        print("LoRA checkpoint loaded successfully")

        # Ensure LoRA parameters are trainable
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
    else:
        print("\nApplying fresh LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Track results across stages
    all_stage_results = {}

    # Run curriculum training
    all_stages = ['scope', 'attachment', 'vague', 'mixed']
    for stage_name in curriculum_stages:
        # Get original stage number (1-indexed)
        stage_num = all_stages.index(stage_name) + 1

        # Get data for this stage
        if stage_name == 'mixed':
            # Use all data for mixed stage
            stage_data = all_data
        else:
            # Use only data matching this ambiguity type
            stage_data = data_by_type.get(stage_name, [])

        if not stage_data:
            print(f"\nWarning: No data found for stage '{stage_name}', skipping...")
            continue

        # Determine epochs for this stage
        stage_epoch_overrides = {
            'scope': args.scope_epochs,
            'attachment': args.attachment_epochs,
            'vague': args.vague_epochs,
            'mixed': args.mixed_epochs,
        }
        override = stage_epoch_overrides.get(stage_name)
        epochs_for_stage = override if override is not None else args.epochs_per_stage
        if epochs_for_stage != args.epochs_per_stage:
            print(f"Using {epochs_for_stage} epoch(s) for {stage_name} stage (overriding default {args.epochs_per_stage})")

        # Train this stage
        stage_rewards, trainer = train_curriculum_stage(
            stage_name=stage_name,
            train_data=stage_data,
            model=model,
            tokenizer=tokenizer,
            base_output_dir=args.output_dir,
            args=args,
            stage_num=stage_num,
            epochs_override=epochs_for_stage,
        )

        all_stage_results[stage_name] = {
            'num_samples': len(stage_data),
            'epoch_rewards': stage_rewards,
            'best_reward': max(stage_rewards) if stage_rewards else 0.0,
        }

        # Cleanup vLLM and GPU memory before next stage
        print(f"\n[Cleanup] Releasing GPU memory before next stage...")

        # Shut down the colocated vLLM engine fully before releasing anything else.
        # In vllm_mode="colocate" the LLM object lives on trainer.llm (TRL >= 0.16).
        # We must call destroy_model_parallel / distributed teardown so vLLM releases
        # its CUDA allocations, then delete every reference before emptying the cache.
        if hasattr(trainer, 'llm'):
            try:
                llm = trainer.llm
                # Step 1: shutdown the engine core (frees KV-cache blocks)
                if hasattr(llm, 'llm_engine'):
                    engine = llm.llm_engine
                    if hasattr(engine, 'engine_core'):
                        try:
                            engine.engine_core.shutdown()
                        except Exception:
                            pass
                    if hasattr(engine, 'shutdown'):
                        try:
                            engine.shutdown()
                        except Exception:
                            pass
                # Step 2: destroy vLLM's model parallel process groups if present
                try:
                    from vllm.distributed.parallel_state import destroy_model_parallel
                    destroy_model_parallel()
                except Exception:
                    pass
                # Step 3: delete the LLM object so its __del__ runs
                del llm
                del trainer.llm
            except Exception as e:
                print(f"[Cleanup] vLLM shutdown warning (non-fatal): {e}")

        # Delete the trainer (releases optimizer states, gradient buffers, etc.)
        del trainer

        # Force garbage collection to run __del__ methods
        gc.collect()

        # Clear CUDA allocator cache — this is the critical step that returns
        # pages to the CUDA driver so the next vLLM LLM() init can see free memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Brief pause to let the driver reclaim memory
        time.sleep(5)

        # Second gc + cache clear cycle for any deferred cleanups
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"[Cleanup] GPU memory released — {free/1024**3:.1f}/{total/1024**3:.1f} GiB free")
        else:
            print(f"[Cleanup] GPU memory released")

    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    print(f"\nSaving final model to {final_dir}...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save curriculum results
    results_path = os.path.join(args.output_dir, "curriculum_results.json")
    results = {
        'curriculum_order': curriculum_stages,
        'epochs_per_stage': args.epochs_per_stage,
        'stages': all_stage_results,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save config
    config_path = os.path.join(args.output_dir, "training_config.json")
    config = vars(args).copy()
    config.pop("anthropic_api_key", None)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*80)
    print(f"\nStage Results:")
    for stage_name, results in all_stage_results.items():
        print(f"  {stage_name}: {results['num_samples']} samples, best reward: {results['best_reward']:.4f}")
    print(f"\nFinal model saved to: {final_dir}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
