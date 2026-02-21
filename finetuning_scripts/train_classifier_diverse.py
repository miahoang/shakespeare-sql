#!/usr/bin/env python3
"""
Script to finetune DeBERTa with diverse dataset mixture.
Train: up to 10000 Ambrosia + up to 200 AmbiQT + up to 200 Spider
Val:   up to 1000 Ambrosia + up to 100 AmbiQT + up to 100 Spider
Test:  Full Ambrosia test set

This tests whether dataset diversity improves generalization.
"""

import os
import csv
import json
import random
import argparse
from collections import Counter
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# Sample size limits per split
AMBROSIA_TRAIN_MAX = 10000
AMBIQT_TRAIN_MAX = 200
SPIDER_TRAIN_MAX = 200

AMBROSIA_VAL_MAX = 1000
AMBIQT_VAL_MAX = 100
SPIDER_VAL_MAX = 100

# Label mapping
LABEL_MAP = {
    'U': 0,   # Unanswerable
    'AA': 1,  # Answerable Ambiguous
    'AU': 2   # Answerable Unambiguous
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def remove_insert_statements(db_dump: str) -> str:
    """
    Remove all INSERT INTO statements from db_dump to reduce sequence length.
    """
    lines = db_dump.split('\n')
    filtered_lines = [line for line in lines if not line.strip().upper().startswith('INSERT INTO')]
    return '\n'.join(filtered_lines)


class QuestionDataset(Dataset):
    """Dataset for question classification."""

    def __init__(self, db_dump_processed: List[str], questions: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.db_dump_processed = db_dump_processed
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        db_dump = self.db_dump_processed[idx]
        question = self.questions[idx]
        label = self.labels[idx]

        # Remove INSERT statements to reduce sequence length
        db_dump = remove_insert_statements(db_dump)

        # Concatenate question and db_dump_processed as two segments
        # Format: [CLS] db_dump_processed [SEP] question [SEP]
        encoding = self.tokenizer(
            db_dump,
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_first',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_ambrosia_sample(csv_path: str, split_filter: str, max_samples: int, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Load random sample from Ambrosia dataset.

    Returns:
        (db_dump_processed, questions, labels)
    """
    db_dumps = []
    questions = []
    labels = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get('split', '').strip()
            if split != split_filter:
                continue

            question = row.get('question', '').strip()
            if not question:
                continue

            db_dump_processed = row.get('db_dump_processed', '').strip()
            if not db_dump_processed:
                db_dump_processed = ""

            # Determine label
            question_type = row.get('question_type', '').strip()
            is_ambiguous = row.get('is_ambiguous', '').strip().upper()

            if question_type == 'unanswerable':
                label = 'U'
            elif is_ambiguous == 'TRUE':
                label = 'AA'
            elif is_ambiguous == 'FALSE':
                label = 'AU'
            else:
                continue

            db_dumps.append(db_dump_processed)
            questions.append(question)
            labels.append(label)

    # Randomly sample
    if max_samples and len(questions) > max_samples:
        random.seed(seed)
        indices = list(range(len(questions)))
        random.shuffle(indices)
        sampled_indices = indices[:max_samples]

        db_dumps = [db_dumps[i] for i in sampled_indices]
        questions = [questions[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]

    return (db_dumps, questions, labels)


def load_ambiqt_sample(csv_path: str, split_filter: str, max_samples: int, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Load random sample from AmbiQT dataset (all AA).

    Returns:
        (db_dump_processed, questions, labels)
    """
    db_dumps = []
    questions = []
    labels = []

    if not os.path.exists(csv_path):
        print(f"Warning: AmbiQT data file not found at {csv_path}")
        return ([], [], [])

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get('split', '').strip()
            if split != split_filter:
                continue

            question = row.get('question', '').strip()
            if not question:
                continue

            db_dump_processed = row.get('db_dump_processed', '').strip()
            if not db_dump_processed:
                db_dump_processed = ""

            # All AmbiQT questions are AA
            db_dumps.append(db_dump_processed)
            questions.append(question)
            labels.append('AA')

    # Randomly sample
    if max_samples and len(questions) > max_samples:
        random.seed(seed)
        indices = list(range(len(questions)))
        random.shuffle(indices)
        sampled_indices = indices[:max_samples]

        db_dumps = [db_dumps[i] for i in sampled_indices]
        questions = [questions[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]

    return (db_dumps, questions, labels)


def load_spider_sample(csv_path: str, split_filter: str, max_samples: int, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Load random sample from Spider dataset (all AU).

    Returns:
        (db_dump_processed, questions, labels)
    """
    db_dumps = []
    questions = []
    labels = []

    if not os.path.exists(csv_path):
        print(f"Warning: Spider data file not found at {csv_path}")
        return ([], [], [])

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get('split', '').strip()
            if split != split_filter:
                continue

            question = row.get('question', '').strip()
            if not question:
                continue

            db_dump_processed = row.get('db_dump_processed', '').strip()
            if not db_dump_processed:
                db_dump_processed = ""

            # All Spider questions are AU
            db_dumps.append(db_dump_processed)
            questions.append(question)
            labels.append('AU')

    # Randomly sample
    if max_samples and len(questions) > max_samples:
        random.seed(seed)
        indices = list(range(len(questions)))
        random.shuffle(indices)
        sampled_indices = indices[:max_samples]

        db_dumps = [db_dumps[i] for i in sampled_indices]
        questions = [questions[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]

    return (db_dumps, questions, labels)


def load_diverse_data(
    ambrosia_path: str,
    ambiqt_path: str,
    spider_path: str,
    seed: int = 42
) -> Dict[str, Tuple[List[str], List[str], List[int]]]:
    """
    Load diverse dataset mixture.

    Returns:
        {split_name: (db_dump_processed, questions, label_ids)}
    """
    split_data = {}

    print("\n" + "="*80)
    print("Loading diverse dataset mixture...")
    print("="*80)

    # Training set: 
    print("\nLoading TRAINING data:")
    print("-" * 80)

    ambrosia_train_db, ambrosia_train_q, ambrosia_train_l = load_ambrosia_sample(
        ambrosia_path, 'train', max_samples=AMBROSIA_TRAIN_MAX, seed=seed
    )
    print(f"  Ambrosia train: {len(ambrosia_train_q)} examples")

    ambiqt_train_db, ambiqt_train_q, ambiqt_train_l = load_ambiqt_sample(
        ambiqt_path, 'train', max_samples=AMBIQT_TRAIN_MAX, seed=seed
    )
    print(f"  AmbiQT train:   {len(ambiqt_train_q)} examples")

    spider_train_db, spider_train_q, spider_train_l = load_spider_sample(
        spider_path, 'train', max_samples=SPIDER_TRAIN_MAX, seed=seed
    )
    print(f"  Spider train:   {len(spider_train_q)} examples")

    # Combine training data
    train_db = ambrosia_train_db + ambiqt_train_db + spider_train_db
    train_q = ambrosia_train_q + ambiqt_train_q + spider_train_q
    train_l = ambrosia_train_l + ambiqt_train_l + spider_train_l

    # Validation set: 
    print("\nLoading VALIDATION data:")
    print("-" * 80)

    ambrosia_val_db, ambrosia_val_q, ambrosia_val_l = load_ambrosia_sample(
        ambrosia_path, 'validation', max_samples=AMBROSIA_VAL_MAX, seed=seed
    )
    print(f"  Ambrosia val:   {len(ambrosia_val_q)} examples")

    # AmbiQT only has 'train' and 'test' splits, not 'validation'
    # Use 'test' split for validation (we're only testing on Ambrosia test set)
    ambiqt_val_db, ambiqt_val_q, ambiqt_val_l = load_ambiqt_sample(
        ambiqt_path, 'test', max_samples=AMBIQT_VAL_MAX, seed=seed
    )
    print(f"  AmbiQT val:     {len(ambiqt_val_q)} examples")

    spider_val_db, spider_val_q, spider_val_l = load_spider_sample(
        spider_path, 'validation', max_samples=SPIDER_VAL_MAX, seed=seed
    )
    print(f"  Spider val:     {len(spider_val_q)} examples")

    # Combine validation data
    val_db = ambrosia_val_db + ambiqt_val_db + spider_val_db
    val_q = ambrosia_val_q + ambiqt_val_q + spider_val_q
    val_l = ambrosia_val_l + ambiqt_val_l + spider_val_l

    # Test set: Full Ambrosia test set
    print("\nLoading TEST data:")
    print("-" * 80)

    ambrosia_test_db, ambrosia_test_q, ambrosia_test_l = load_ambrosia_sample(
        ambrosia_path, 'test', max_samples=None, seed=seed  # Full test set
    )
    print(f"  Ambrosia test:  {len(ambrosia_test_q)} examples (full set)")

    # Convert labels to IDs
    train_label_ids = [LABEL_MAP[label] for label in train_l]
    val_label_ids = [LABEL_MAP[label] for label in val_l]
    test_label_ids = [LABEL_MAP[label] for label in ambrosia_test_l]

    split_data['train'] = (train_db, train_q, train_label_ids)
    split_data['validation'] = (val_db, val_q, val_label_ids)
    split_data['test'] = (ambrosia_test_db, ambrosia_test_q, test_label_ids)

    return split_data


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }


def print_classification_report(trainer: Trainer, test_dataset: QuestionDataset, output_dir: str, split_name: str = "test"):
    """Generate and print detailed classification report."""
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    # Convert to label names
    pred_names = [ID_TO_LABEL[label_id] for label_id in pred_labels]
    true_names = [ID_TO_LABEL[label_id] for label_id in true_labels]

    # Print overall classification report
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION REPORT - {split_name.upper()} SET (OVERALL)")
    print(f"{'='*80}")
    report = classification_report(true_names, pred_names)
    print(report)

    # Print per-question-type breakdown
    print(f"\n{'='*80}")
    print(f"PER-QUESTION-TYPE BREAKDOWN - {split_name.upper()} SET")
    print(f"{'='*80}")

    # Group by true label (question type)
    for question_type in ['U', 'AA', 'AU']:
        if question_type not in ID_TO_LABEL.values():
            continue

        # Get indices where true label matches this question type
        indices = [i for i, label in enumerate(true_names) if label == question_type]

        if len(indices) == 0:
            continue

        # Filter predictions and true labels for this question type
        type_true = [true_names[i] for i in indices]
        type_pred = [pred_names[i] for i in indices]

        # Calculate accuracy for this type
        type_accuracy = accuracy_score(type_true, type_pred)
        type_total = len(indices)
        type_correct = sum(1 for t, p in zip(type_true, type_pred) if t == p)

        print(f"\nQuestion Type: {question_type}")
        print(f"  Total: {type_total}")
        print(f"  Correct: {type_correct}")
        print(f"  Accuracy: {type_accuracy:.4f}")

        # Show confusion for this type
        type_counter = Counter(type_pred)
        print(f"  Predictions breakdown:")
        for pred_label in ['U', 'AA', 'AU']:
            count = type_counter.get(pred_label, 0)
            pct = (count / type_total * 100) if type_total > 0 else 0
            print(f"    Predicted as {pred_label}: {count} ({pct:.1f}%)")

    # Save report to file
    report_path = os.path.join(output_dir, f'classification_report_{split_name}.txt')
    with open(report_path, 'w') as f:
        f.write(f"CLASSIFICATION REPORT - {split_name.upper()} SET (OVERALL)\n")
        f.write("="*80 + "\n")
        f.write(report)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write(f"PER-QUESTION-TYPE BREAKDOWN - {split_name.upper()} SET\n")
        f.write("="*80 + "\n")

        for question_type in ['U', 'AA', 'AU']:
            if question_type not in ID_TO_LABEL.values():
                continue

            indices = [i for i, label in enumerate(true_names) if label == question_type]
            if len(indices) == 0:
                continue

            type_true = [true_names[i] for i in indices]
            type_pred = [pred_names[i] for i in indices]
            type_accuracy = accuracy_score(type_true, type_pred)
            type_total = len(indices)
            type_correct = sum(1 for t, p in zip(type_true, type_pred) if t == p)

            f.write(f"\nQuestion Type: {question_type}\n")
            f.write(f"  Total: {type_total}\n")
            f.write(f"  Correct: {type_correct}\n")
            f.write(f"  Accuracy: {type_accuracy:.4f}\n")

            type_counter = Counter(type_pred)
            f.write(f"  Predictions breakdown:\n")
            for pred_label in ['U', 'AA', 'AU']:
                count = type_counter.get(pred_label, 0)
                pct = (count / type_total * 100) if type_total > 0 else 0
                f.write(f"    Predicted as {pred_label}: {count} ({pct:.1f}%)\n")

    print(f"\nClassification report saved to: {report_path}")

    # Save predictions
    predictions_path = os.path.join(output_dir, f'predictions_{split_name}.json')
    predictions_data = {
        'predictions': pred_names,
        'true_labels': true_names,
        'metrics': {
            'accuracy': accuracy_score(true_labels, pred_labels),
        },
        'per_question_type': {}
    }

    # Add per-type metrics to JSON
    for question_type in ['U', 'AA', 'AU']:
        if question_type not in ID_TO_LABEL.values():
            continue

        indices = [i for i, label in enumerate(true_names) if label == question_type]
        if len(indices) == 0:
            continue

        type_true = [true_names[i] for i in indices]
        type_pred = [pred_names[i] for i in indices]
        type_accuracy = accuracy_score(type_true, type_pred)
        type_total = len(indices)
        type_correct = sum(1 for t, p in zip(type_true, type_pred) if t == p)
        type_counter = Counter(type_pred)

        predictions_data['per_question_type'][question_type] = {
            'total': type_total,
            'correct': type_correct,
            'accuracy': type_accuracy,
            'predictions_breakdown': dict(type_counter)
        }

    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)

    print(f"Predictions saved to: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(description='Finetune DeBERTa with diverse dataset mixture')

    # Data arguments
    parser.add_argument('--ambrosia_path', type=str,
                        default='data/ambrosia/ambrosia_with_unanswerable_validated.csv',
                        help='Path to Ambrosia CSV file')
    parser.add_argument('--ambiqt_path', type=str,
                        default='data/AmbiQT/ambiqt_ambrosia_format.csv',
                        help='Path to AmbiQT CSV file')
    parser.add_argument('--spider_path', type=str,
                        default='data/spider_data/reformatted/spider_ambrosia_format.csv',
                        help='Path to Spider CSV file')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                        help='Pretrained model name')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')

    # Training arguments
    parser.add_argument('--config_path', type=str,
                        default='finetuning/models/deberta-v3-base_20251029_221116/training_config.json',
                        help='Path to config JSON file with hyperparameters')
    parser.add_argument('--output_dir', type=str,
                        default='finetuning/models/deberta_diverse',
                        help='Output directory for model')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size (overrides config if specified)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config if specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Load hyperparameters from config file
    print(f"Loading hyperparameters from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Extract params from config
    if 'hyperparameters' in config:
        config_params = config['hyperparameters']
    elif 'params' in config:
        config_params = config['params']
    else:
        config_params = {}

    # Use config values as defaults, allow command-line overrides
    if args.batch_size is None:
        args.batch_size = config_params.get('batch_size', 8)
    if args.learning_rate is None:
        args.learning_rate = config_params.get('learning_rate', 3e-5)

    warmup_steps = config_params.get('warmup_steps', 500)
    weight_decay = config_params.get('weight_decay', 0.01)
    drop_out = config_params.get('dropout', 0.1)
    max_grad_norm = config_params.get('max_grad_norm', 1.0)
    lr_reduce_factor = config_params.get('lr_reduce_factor', 0.5)
    lr_reduce_patience = config_params.get('lr_reduce_patience', 2)

    print(f"\nHyperparameters loaded from config:")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  dropout: {drop_out}")
    print(f"  max_grad_norm: {max_grad_norm}")

    # Generate output directory with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    args.output_dir = os.path.join('finetuning/models', f"{model_short_name}_diverse_{timestamp}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    print("="*80)
    print("DeBERTa Question Classifier - Diverse Dataset Training")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Train: up to {AMBROSIA_TRAIN_MAX} Ambrosia + up to {AMBIQT_TRAIN_MAX} AmbiQT + up to {SPIDER_TRAIN_MAX} Spider")
    print(f"Val:   up to {AMBROSIA_VAL_MAX} Ambrosia + up to {AMBIQT_VAL_MAX} AmbiQT + up to {SPIDER_VAL_MAX} Spider")
    print(f"Test:  Full Ambrosia test set")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*80)

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        hidden_dropout_prob=drop_out,
        attention_probs_dropout_prob=drop_out
    )

    # Load diverse dataset
    split_data = load_diverse_data(
        args.ambrosia_path,
        args.ambiqt_path,
        args.spider_path,
        seed=args.seed
    )

    # Print split statistics
    print(f"\n{'='*80}")
    print("FINAL DATASET STATISTICS")
    print(f"{'='*80}")
    for split, (db_dumps, questions, labels) in sorted(split_data.items()):
        print(f"\n{split}:")
        print(f"  Total: {len(questions)} questions")
        label_dist = Counter(labels)
        for label_id in sorted(label_dist.keys()):
            label_name = ID_TO_LABEL[label_id]
            count = label_dist[label_id]
            pct = (count / len(labels)) * 100
            print(f"  {label_name}: {count} ({pct:.1f}%)")

    # Rename splits for clarity
    train_db_dumps, train_questions, train_labels = split_data['train']
    val_db_dumps, val_questions, val_labels = split_data['validation']
    test_db_dumps, test_questions, test_labels = split_data['test']

    # Create datasets
    train_dataset = QuestionDataset(train_db_dumps, train_questions, train_labels, tokenizer, args.max_length)
    val_dataset = QuestionDataset(val_db_dumps, val_questions, val_labels, tokenizer, args.max_length)
    test_dataset = QuestionDataset(test_db_dumps, test_questions, test_labels, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type='reduce_lr_on_plateau',
        lr_scheduler_kwargs={
            'mode': 'max',
            'factor': lr_reduce_factor,
            'patience': lr_reduce_patience,
            'min_lr': 1e-7,
        },
        optim='adamw_torch',
        weight_decay=weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_strategy='epoch',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_weighted',
        greater_is_better=True,
        seed=args.seed,
        data_seed=args.seed,  # Ensures DataLoader shuffling is reproducible
        report_to='none',
        save_total_limit=2,
        max_grad_norm=max_grad_norm,
    )

    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    trainer.train()

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print("\nTest set results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            print(f"  {metric_name}: {value:.4f}")

    # Generate classification reports
    print_classification_report(trainer, val_dataset, args.output_dir, 'validation')
    print_classification_report(trainer, test_dataset, args.output_dir, 'test')

    # Save the final model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training configuration
    config_save = {
        'args': vars(args),
        'loaded_from_config': args.config_path,
        'dataset_composition': {
            'train': f'up to {AMBROSIA_TRAIN_MAX} Ambrosia + up to {AMBIQT_TRAIN_MAX} AmbiQT + up to {SPIDER_TRAIN_MAX} Spider',
            'validation': f'up to {AMBROSIA_VAL_MAX} Ambrosia + up to {AMBIQT_VAL_MAX} AmbiQT + up to {SPIDER_VAL_MAX} Spider',
            'test': 'Full Ambrosia test set'
        },
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'dropout': drop_out,
            'max_grad_norm': max_grad_norm,
            'lr_reduce_factor': lr_reduce_factor,
            'lr_reduce_patience': lr_reduce_patience,
        }
    }
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)

    print(f"Training configuration saved to: {config_path}")

    print("\n" + "="*80)
    print("All done! ✓")
    print("="*80)
    print(f"\nModel saved to: {args.output_dir}")
    print("\nDataset diversity experiment complete!")
    print("This model was trained on a diverse mix to test generalization.")

if __name__ == '__main__':
    main()
