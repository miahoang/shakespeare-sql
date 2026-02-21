#!/usr/bin/env python3
"""
LDA-based Ambiguity Type Classifier for Ambrosia Dataset

Uses Linear Discriminant Analysis to find linear combinations of embedding
dimensions that maximize separation between ambiguity types.

Performance: 94.42% accuracy (vs 77.29% with max-similarity)
"""

import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class AmbiguityTypeClassifierLDA:
    """
    LDA-based classifier for ambiguity types.

    Uses Linear Discriminant Analysis on sentence embeddings to find
    optimal linear combinations that separate scope/attachment/vague.
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        training_csv_path: str = 'data/ambrosia/ambrosia_with_unanswerable_validated.csv'
    ):
        """
        Initialize and train LDA classifier.

        Args:
            model_name: Name of sentence transformer model
            training_csv_path: Path to training CSV
        """
        self.model = SentenceTransformer(model_name)
        self.lda = LinearDiscriminantAnalysis()
        self.scaler = StandardScaler()  # Optional: normalize embeddings

        # Train LDA on training data
        self._train_lda(training_csv_path)

    def _train_lda(self, csv_path: str):
        """Train LDA classifier on training data."""
        # Load training data
        df = pd.read_csv(csv_path)
        train_df = df[(df['question_type'] == 'ambig') &
                      (df['ambig_type'] != 'unanswerable') &
                      (df['split'] == 'train')].copy()

        print(f"Training LDA on {len(train_df)} examples...")

        # Embed questions
        questions = train_df['question'].tolist()
        labels = train_df['ambig_type'].tolist()

        embeddings = self.model.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Train LDA
        self.lda.fit(embeddings, labels)

        print(f"LDA trained successfully")
        print(f"Number of components: {self.lda.n_components}")

        # Show per-class performance on training data
        train_preds = self.lda.predict(embeddings)
        train_acc = np.mean([pred == true for pred, true in zip(train_preds, labels)])
        print(f"Training accuracy: {train_acc:.2%}")

    def predict(
        self,
        question: str,
        db_schema: Optional[str] = None,
        top_k: int = 1,
        return_scores: bool = False
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Predict ambiguity type(s) for a question.

        Args:
            question: Input question to classify
            db_schema: Optional database schema (currently not used by LDA)
            top_k: Number of top predictions to return
            return_scores: If True, return (type, probability) tuples

        Returns:
            List of predicted ambiguity types, optionally with probabilities
        """
        # Encode question
        embedding = self.model.encode(
            [question],
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Get probabilities from LDA
        probabilities = self.lda.predict_proba(embedding)[0]

        # Get class names
        classes = self.lda.classes_

        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1][:top_k]

        if return_scores:
            return [(classes[i], probabilities[i]) for i in sorted_indices]
        else:
            return [classes[i] for i in sorted_indices]

    def predict_with_confidence(
        self,
        question: str,
        db_schema: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Predict ambiguity type with confidence score.

        Args:
            question: Input question to classify
            db_schema: Optional database schema (not used)

        Returns:
            (predicted_type, confidence) where confidence is the probability
        """
        predictions = self.predict(question, db_schema, top_k=1, return_scores=True)
        return predictions[0]  # (type, probability)

    def predict_batch(
        self,
        questions: List[str],
        db_schemas: Optional[List[str]] = None,
        top_k: int = 1,
        return_scores: bool = False
    ) -> List[List[str]] | List[List[Tuple[str, float]]]:
        """
        Predict ambiguity types for multiple questions (batch processing).

        Args:
            questions: List of input questions
            db_schemas: Optional list of database schemas (not used by LDA)
            top_k: Number of top predictions per question
            return_scores: If True, return (type, probability) tuples

        Returns:
            List of predictions for each question
        """
        # Encode all questions
        embeddings = self.model.encode(
            questions,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Get probabilities
        probabilities = self.lda.predict_proba(embeddings)
        classes = self.lda.classes_

        results = []
        for probs in probabilities:
            sorted_indices = np.argsort(probs)[::-1][:top_k]

            if return_scores:
                results.append([(classes[i], probs[i]) for i in sorted_indices])
            else:
                results.append([classes[i] for i in sorted_indices])

        return results

    def explain_prediction(
        self,
        question: str,
        db_schema: Optional[str] = None
    ) -> Dict:
        """
        Provide detailed explanation for ambiguity type prediction.

        Args:
            question: Input question
            db_schema: Optional database schema

        Returns:
            Dictionary with prediction details and explanations
        """
        # Get predictions with probabilities
        predictions = self.predict(
            question,
            db_schema,
            top_k=3,
            return_scores=True
        )

        explanation = {
            'question': question,
            'predictions': []
        }

        for type_name, prob in predictions:
            explanation['predictions'].append({
                'type': type_name,
                'probability': float(prob),
                'confidence': 'high' if prob > 0.7 else 'medium' if prob > 0.4 else 'low'
            })

        return explanation


def evaluate_classifier_lda(
    csv_path: str = 'data/ambrosia/ambrosia_with_unanswerable_validated.csv',
    sample_size: Optional[int] = None
) -> Dict:
    """
    Evaluate LDA classifier accuracy on test data.

    Args:
        csv_path: Path to Ambrosia CSV file with labeled examples
        sample_size: Number of examples to evaluate (None = all)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Filter to ambiguous examples only (use test/validation split)
    df = df[df['question_type'] == 'ambig'].copy()
    df = df[df['ambig_type'] != 'unanswerable'].copy()
    df = df[df['split'] != 'train'].copy()

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f"Evaluating on {len(df)} examples from Ambrosia dataset...")

    # Initialize classifier (trains on training data)
    classifier = AmbiguityTypeClassifierLDA()

    # Get predictions
    questions = df['question'].tolist()
    true_labels = df['ambig_type'].tolist()

    # Predict top-1 and top-2
    print("Predicting...")
    pred_top1 = classifier.predict_batch(questions, top_k=1)
    pred_top2 = classifier.predict_batch(questions, top_k=2)

    # Calculate metrics
    top1_correct = sum(
        1 for pred, true in zip(pred_top1, true_labels)
        if pred[0] == true
    )

    top2_correct = sum(
        1 for pred, true in zip(pred_top2, true_labels)
        if true in pred
    )

    # Per-type breakdown
    type_stats = {}
    for ambig_type in df['ambig_type'].unique():
        type_mask = df['ambig_type'] == ambig_type
        type_df = df[type_mask]
        type_indices = type_df.index.tolist()

        type_preds = [pred_top1[i] for i, idx in enumerate(df.index) if idx in type_indices]
        type_true = type_df['ambig_type'].tolist()

        type_correct = sum(1 for pred, true in zip(type_preds, type_true) if pred[0] == true)

        type_stats[ambig_type] = {
            'count': len(type_df),
            'accuracy': type_correct / len(type_df) if len(type_df) > 0 else 0
        }

    results = {
        'total_examples': len(df),
        'top1_accuracy': top1_correct / len(df),
        'top2_accuracy': top2_correct / len(df),
        'per_type_accuracy': type_stats
    }

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("LDA-BASED AMBIGUITY TYPE CLASSIFIER")
    print("=" * 80)

    # Evaluate on full test set
    results = evaluate_classifier_lda(sample_size=None)

    print(f"\nOverall Accuracy: {results['top1_accuracy']:.2%}")
    print(f"Top-2 Accuracy: {results['top2_accuracy']:.2%}")

    print("\nPer-type accuracy:")
    for ambig_type, stats in sorted(results['per_type_accuracy'].items()):
        print(f"  {ambig_type}: {stats['accuracy']:.2%} ({stats['count']} examples)")

    # Test on example questions
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)

    classifier = AmbiguityTypeClassifierLDA()

    test_questions = [
        ("Tell me the systems every support specialist works with.", "scope"),
        ("Show all banquet halls and conference rooms with a 200 person capacity.", "attachment"),
        ("What compensation is offered to software developers?", "vague"),
    ]

    for question, expected in test_questions:
        explanation = classifier.explain_prediction(question)
        predicted = explanation['predictions'][0]

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted['type']} (probability: {predicted['probability']:.2%})")
        print(f"✓ Correct!" if predicted['type'] == expected else "✗ Incorrect")
