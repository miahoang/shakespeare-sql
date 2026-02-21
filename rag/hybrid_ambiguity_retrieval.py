#!/usr/bin/env python3
"""
Hybrid Ambiguity-Aware Retrieval using Semantic + Structural Features

This module combines:
1. Semantic similarity (sentence embeddings of question + schema)
2. Structural similarity (POS tags, dependency parse, n-grams)
3. LDA classifier to predict ambiguity type
4. Filtered retrieval to get examples with same ambiguity type

Key insight: Ambiguity types have structural patterns:
- Attachment: PP-attachment patterns (IN/TO + NP), e.g., "for the X"
- Scope: Quantifiers (DT "each"/"all"/"every") + scope markers
- Vague: Multiple synonym candidates (more semantic than structural)

Usage:
    retriever = HybridAmbiguityRetriever()

    # Predict ambiguity type via LDA classifier (94%+ accuracy)
    predicted_type, confidence = retriever.predict_ambiguity_type(question, db_dump)

    # Retrieve examples with same ambiguity type
    examples = retriever.retrieve_by_ambiguity_type(
        question, db_dump, ambig_type=predicted_type, k=5
    )
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.rag_vectordb import VectorDatabase
from rag.ambiguity_type_classifier_lda import AmbiguityTypeClassifierLDA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridAmbiguityRetriever:
    """
    Hybrid retrieval system using LDA classifier and structural features.

    Architecture:
    1. LDA Classifier: Predicts ambiguity type from question embeddings
       - Trained on Linear Discriminant Analysis
       - 94%+ accuracy on attachment/scope/vague classification
    2. Structural Features: POS tags, dependency patterns, n-grams
       - Used for type-specific retrieval (finding similar examples)
       - Weighted combination of semantic + structural similarity
    3. Type-specific Retrieval: Get examples with the predicted ambiguity type
    """

    def __init__(
        self,
        vectordb: Optional[VectorDatabase] = None,
        vectordb_cache_dir: str = "data/vectordb_cache",
        csv_path: str = "data/ambrosia/ambrosia_with_unanswerable_validated.csv",
        semantic_weight: float = 0.7,
        structural_weight: float = 0.3,
        use_pos: bool = True,
        use_dependency: bool = True,
        use_ngrams: bool = True
    ):
        """
        Initialize hybrid retriever with LDA classifier.

        Args:
            vectordb: Existing VectorDatabase instance (or create new one)
            vectordb_cache_dir: Cache directory for vector database
            csv_path: Path to training data CSV
            semantic_weight: Weight for semantic similarity (0-1)
            structural_weight: Weight for structural similarity (0-1)
            use_pos: Include POS tag features for structural retrieval
            use_dependency: Include dependency parse features for structural retrieval
            use_ngrams: Include character n-gram features for structural retrieval
        """
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.use_pos = use_pos
        self.use_dependency = use_dependency
        self.use_ngrams = use_ngrams

        # Validate weights
        total_weight = semantic_weight + structural_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0 (sum={total_weight}), normalizing...")
            self.semantic_weight = semantic_weight / total_weight
            self.structural_weight = structural_weight / total_weight

        # Initialize or use existing vectordb
        if vectordb is None:
            logger.info("Initializing VectorDatabase...")
            self.vectordb = VectorDatabase(cache_dir=vectordb_cache_dir)
            self.vectordb.build_from_csv(csv_path, split='train', force_rebuild=False)
        else:
            self.vectordb = vectordb

        # Initialize LDA classifier (required)
        logger.info("Initializing LDA classifier for ambiguity type prediction...")
        self.lda_classifier = AmbiguityTypeClassifierLDA(
            training_csv_path=csv_path
        )
        logger.info("LDA classifier initialized successfully")

        # Load spaCy for POS tagging and dependency parsing (still needed for structural retrieval)
        if use_pos or use_dependency:
            logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                logger.info("Falling back to basic features only")
                self.use_pos = False
                self.use_dependency = False
                self.nlp = None
        else:
            self.nlp = None

        # Initialize structural feature extractors (for retrieval, not classification)
        logger.info("Building structural feature extractors...")
        self._build_structural_features()

        logger.info("HybridAmbiguityRetriever initialized")

    def _build_structural_features(self):
        """Pre-compute structural features for all training examples"""
        self.pos_features = []
        self.dep_features = []
        self.structural_feature_matrix = None

        if not self.vectordb.examples:
            logger.warning("No examples in vectordb")
            return

        questions = [ex['question'] for ex in self.vectordb.examples]

        # Extract POS features
        if self.use_pos and self.nlp:
            logger.info("Extracting POS features...")
            self.pos_features = [self._extract_pos_features(q) for q in questions]

        # Extract dependency features
        if self.use_dependency and self.nlp:
            logger.info("Extracting dependency features...")
            self.dep_features = [self._extract_dependency_features(q) for q in questions]

        # Build combined structural feature strings
        structural_texts = []
        for i, question in enumerate(questions):
            features = []

            # Add POS features
            if self.pos_features:
                features.append(self.pos_features[i])

            # Add dependency features
            if self.dep_features:
                features.append(self.dep_features[i])

            # Add original question for n-grams
            if self.use_ngrams:
                features.append(question)

            structural_texts.append(' '.join(features))

        # Create TF-IDF matrix for structural features
        if structural_texts:
            logger.info("Building TF-IDF matrix for structural features...")
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                lowercase=True,
                analyzer='char_wb'  # Character n-grams within word boundaries
            )
            self.structural_feature_matrix = self.tfidf.fit_transform(structural_texts)
            logger.info(f"Structural feature matrix shape: {self.structural_feature_matrix.shape}")

    def _extract_pos_features(self, question: str) -> str:
        """
        Extract POS tag sequence features.

        Returns string representation of POS patterns, e.g.:
        "DET NOUN VERB DET NOUN" for "each student takes a course"

        Also extracts special patterns:
        - Quantifiers: each, all, every, some
        - Prepositions: for, with, by, in
        - Comparative/superlative: more, most, less, least
        """
        if not self.nlp:
            return ""

        doc = self.nlp(question.lower())

        # Basic POS sequence
        pos_seq = ' '.join([token.pos_ for token in doc])

        # Special pattern markers for ambiguity types
        patterns = []

        # Quantifiers (for scope ambiguity)
        quantifiers = [token.text for token in doc if token.text in ['each', 'every', 'all', 'some', 'any']]
        if quantifiers:
            patterns.append(f"QUANT_{' '.join(quantifiers)}")

        # Prepositions (for attachment ambiguity)
        prepositions = [token.text for token in doc if token.pos_ == 'ADP']
        if prepositions:
            patterns.append(f"PREP_{' '.join(prepositions)}")

        # Conjunction patterns (for scope/attachment)
        conjunctions = [token.text for token in doc if token.pos_ == 'CCONJ']
        if conjunctions:
            patterns.append(f"CONJ_{' '.join(conjunctions)}")

        return f"{pos_seq} {' '.join(patterns)}"

    def _extract_dependency_features(self, question: str) -> str:
        """
        Extract dependency parse features.

        Returns string of dependency relation patterns, useful for:
        - PP-attachment: prep -> pobj chains
        - Modification: amod, advmod patterns
        - Quantifier scope: det -> nsubj chains
        """
        if not self.nlp:
            return ""

        doc = self.nlp(question.lower())

        # Dependency relation sequence
        dep_seq = ' '.join([token.dep_ for token in doc])

        # Special dependency patterns
        patterns = []

        # PP-attachment patterns (for attachment ambiguity)
        for token in doc:
            if token.dep_ == 'prep':
                # Get the preposition and its object
                pobj = [child.text for child in token.children if child.dep_ == 'pobj']
                if pobj:
                    patterns.append(f"PP_{token.text}_{pobj[0]}")

        # Quantifier scope patterns
        for token in doc:
            if token.text in ['each', 'every', 'all'] and token.dep_ == 'det':
                # Get the head noun
                if token.head:
                    patterns.append(f"QUANT_SCOPE_{token.text}_{token.head.text}")

        return f"{dep_seq} {' '.join(patterns)}"

    def _compute_structural_similarity(self, question: str) -> np.ndarray:
        """
        Compute structural similarity between query question and all examples.

        Returns:
            Array of similarity scores (length = number of examples)
        """
        if self.structural_feature_matrix is None:
            # No structural features available
            return np.ones(len(self.vectordb.examples))

        # Extract structural features for query
        features = []

        if self.use_pos and self.nlp:
            features.append(self._extract_pos_features(question))

        if self.use_dependency and self.nlp:
            features.append(self._extract_dependency_features(question))

        if self.use_ngrams:
            features.append(question)

        query_text = ' '.join(features)

        # Transform to TF-IDF vector
        query_vector = self.tfidf.transform([query_text])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.structural_feature_matrix)[0]

        return similarities

    def predict_ambiguity_type(
        self,
        question: str,
        db_dump: str,
        k_neighbors: int = 20,
        min_confidence: float = 0.3
    ) -> Tuple[Optional[str], float]:
        """
        Predict ambiguity type using LDA classifier.

        Args:
            question: Query question
            db_dump: Database schema (not used by LDA)
            k_neighbors: Not used (kept for API compatibility)
            min_confidence: Minimum confidence threshold (return None if below)

        Returns:
            (predicted_type, confidence) where confidence is the LDA probability
            Returns (None, 0.0) if confidence is below threshold or LDA fails
        """
        if not self.lda_classifier:
            logger.error("LDA classifier not initialized")
            return None, 0.0

        try:
            # Get top prediction with probability
            predictions = self.lda_classifier.predict(
                question,
                db_schema=db_dump,
                top_k=1,
                return_scores=True
            )

            if predictions:
                predicted_type, confidence = predictions[0]

                # Check minimum confidence
                if confidence < min_confidence:
                    logger.debug(f"LDA confidence {confidence:.3f} below threshold {min_confidence}")
                    return None, confidence

                logger.debug(f"LDA predicted: {predicted_type} (conf: {confidence:.3f})")
                return predicted_type, confidence
            else:
                logger.warning("LDA returned no predictions")
                return None, 0.0

        except Exception as e:
            logger.error(f"LDA prediction failed: {e}")
            return None, 0.0

    def retrieve_by_ambiguity_type(
        self,
        question: str,
        db_dump: str,
        ambig_type: Optional[str] = None,
        k: int = 5,
        auto_predict: bool = True,
        fallback_to_semantic: bool = True
    ) -> List[Dict]:
        """
        Retrieve examples with specific ambiguity type.

        Args:
            question: Query question
            db_dump: Database schema
            ambig_type: Target ambiguity type (or None to auto-predict)
            k: Number of examples to retrieve
            auto_predict: Automatically predict type if ambig_type is None
            fallback_to_semantic: Fall back to pure semantic retrieval if no type match

        Returns:
            List of examples with the specified ambiguity type
        """
        # Auto-predict ambiguity type if needed
        if ambig_type is None and auto_predict:
            ambig_type, confidence = self.predict_ambiguity_type(question, db_dump)
            if ambig_type:
                logger.info(f"Predicted ambiguity type: {ambig_type} (confidence: {confidence:.3f})")
            else:
                logger.info("Could not predict ambiguity type with sufficient confidence")

        if ambig_type is None:
            # Fall back to regular semantic retrieval
            if fallback_to_semantic:
                logger.info("Falling back to semantic retrieval (no ambiguity type)")
                return self.vectordb.retrieve_similar(question, db_dump, k=k)
            else:
                return []

        # Retrieve more candidates than needed
        candidates = self.vectordb.retrieve_similar(
            question=question,
            db_dump=db_dump,
            k=min(len(self.vectordb.examples), k * 10),  # Get 10x more for filtering
            exclude_exact_match=True
        )

        # Get structural similarities for re-ranking
        structural_similarities = self._compute_structural_similarity(question)

        # Filter by ambiguity type and re-rank with hybrid similarity
        filtered = []
        for result in candidates:
            if result.get('ambig_type') == ambig_type:
                # Find index for structural similarity
                example_idx = None
                for idx, ex in enumerate(self.vectordb.examples):
                    if ex['question'] == result['question']:
                        example_idx = idx
                        break

                if example_idx is not None:
                    # Compute hybrid similarity
                    semantic_sim = result['similarity']
                    structural_sim = structural_similarities[example_idx]
                    hybrid_sim = (
                        self.semantic_weight * semantic_sim +
                        self.structural_weight * structural_sim
                    )
                    result['hybrid_similarity'] = hybrid_sim
                    result['structural_similarity'] = structural_sim
                else:
                    result['hybrid_similarity'] = result['similarity']
                    result['structural_similarity'] = 0.0

                filtered.append(result)

        # Sort by hybrid similarity
        filtered.sort(key=lambda x: x['hybrid_similarity'], reverse=True)

        # Return top k
        results = filtered[:k]

        # Fall back to semantic if not enough results
        if len(results) < k and fallback_to_semantic:
            logger.warning(f"Only found {len(results)} examples with type '{ambig_type}', "
                          f"falling back to semantic retrieval for remaining {k - len(results)}")
            # Get additional semantic results
            additional = self.vectordb.retrieve_similar(question, db_dump, k=k - len(results))
            results.extend(additional)

        return results

    def analyze_ambiguity_distribution(
        self,
        question: str,
        db_dump: str,
        k_neighbors: int = 50
    ) -> Dict[str, float]:
        """
        Analyze the distribution of ambiguity types in nearest neighbors.
        Useful for debugging and understanding predictions.

        Returns:
            Dictionary mapping ambiguity types to their weighted vote proportions
        """
        # Get neighbors
        semantic_results = self.vectordb.retrieve_similar(
            question=question,
            db_dump=db_dump,
            k=k_neighbors,
            exclude_exact_match=True
        )

        # Get structural similarity
        structural_similarities = self._compute_structural_similarity(question)

        # Collect weighted votes
        vote_weights = Counter()
        total_weight = 0.0

        for result in semantic_results:
            # Find example index
            example_idx = None
            for idx, ex in enumerate(self.vectordb.examples):
                if ex['question'] == result['question']:
                    example_idx = idx
                    break

            if example_idx is None:
                continue

            # Compute hybrid similarity
            semantic_sim = result['similarity']
            structural_sim = structural_similarities[example_idx]
            combined_sim = (
                self.semantic_weight * semantic_sim +
                self.structural_weight * structural_sim
            )

            # Get ambiguity type
            ambig_type = result.get('ambig_type', 'None')
            vote_weights[ambig_type] += combined_sim
            total_weight += combined_sim

        # Normalize to proportions
        if total_weight > 0:
            distribution = {k: v / total_weight for k, v in vote_weights.items()}
        else:
            distribution = {}

        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))


def main():
    """Test hybrid retrieval"""
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid ambiguity retrieval")
    parser.add_argument("--question", required=True, help="Test question")
    parser.add_argument("--db-dump", help="Database schema (optional for testing)")
    parser.add_argument("--k", type=int, default=5, help="Number of examples to retrieve")
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument("--structural-weight", type=float, default=0.3)

    args = parser.parse_args()

    # Simple test schema if not provided
    db_dump = args.db_dump or "CREATE TABLE students (id INT, name TEXT); CREATE TABLE courses (id INT, title TEXT);"

    # Initialize retriever
    logger.info("Initializing hybrid retriever...")
    retriever = HybridAmbiguityRetriever(
        semantic_weight=args.semantic_weight,
        structural_weight=args.structural_weight
    )

    # Predict ambiguity type
    logger.info(f"\nQuestion: {args.question}")
    predicted_type, confidence = retriever.predict_ambiguity_type(args.question, db_dump)
    logger.info(f"Predicted type: {predicted_type} (confidence: {confidence:.3f})")

    # Show distribution
    distribution = retriever.analyze_ambiguity_distribution(args.question, db_dump)
    logger.info(f"\nAmbiguity type distribution in neighbors:")
    for ambig_type, proportion in distribution.items():
        logger.info(f"  {ambig_type}: {proportion:.3f}")

    # Retrieve examples
    logger.info(f"\nRetrieving {args.k} examples with type '{predicted_type}'...")
    examples = retriever.retrieve_by_ambiguity_type(
        question=args.question,
        db_dump=db_dump,
        ambig_type=predicted_type,
        k=args.k
    )

    logger.info(f"\nRetrieved {len(examples)} examples:")
    for i, ex in enumerate(examples):
        logger.info(f"\n{i+1}. {ex['question']}")
        logger.info(f"   Type: {ex.get('ambig_type', 'None')}")
        logger.info(f"   Semantic sim: {ex.get('similarity', 0):.3f}")
        logger.info(f"   Structural sim: {ex.get('structural_similarity', 0):.3f}")
        logger.info(f"   Hybrid sim: {ex.get('hybrid_similarity', 0):.3f}")


if __name__ == '__main__':
    main()
