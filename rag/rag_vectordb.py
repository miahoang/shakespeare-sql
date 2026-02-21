#!/usr/bin/env python3
"""
Vector database for retrieval-augmented generation (RAG).

This module provides functionality to:
1. Embed training examples using sentence transformers
2. Store embeddings in FAISS for efficient similarity search
3. Retrieve top-k most similar examples for few-shot prompting

Usage:
    # Build vector database from training data
    vectordb = VectorDatabase(cache_dir='data/vectordb_cache')
    vectordb.build_from_csv(
        csv_path='data/ambrosia/ambrosia_with_unanswerable_validated.csv',
        split='train'
    )

    # Retrieve similar examples
    similar_examples = vectordb.retrieve_similar(
        question="What compensation is offered to developers?",
        db_dump=db_schema,
        k=3
    )
"""

import os
import csv
import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")


class VectorDatabase:
    """
    Vector database for retrieval-augmented generation.

    Embeddings are created by concatenating question + simplified db_schema.
    This allows retrieval based on semantic similarity of both the question
    and the database structure.
    """

    def __init__(
        self,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        cache_dir: str = 'data/vectordb_cache',
        use_cache: bool = True
    ):
        """
        Initialize vector database.

        Args:
            embedding_model: HuggingFace model name for embeddings
            cache_dir: Directory to cache embeddings and index
            use_cache: Whether to use cached embeddings if available
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")

        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Initialize model
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Storage
        self.examples: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _simplify_db_dump(self, db_dump: str) -> str:
        """
        Simplify database dump by removing INSERT statements and extra whitespace.
        This reduces noise in embeddings.
        """
        lines = db_dump.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Keep CREATE TABLE, CREATE INDEX, and other schema statements
            if line.upper().startswith(('CREATE', 'FOREIGN KEY', 'PRIMARY KEY', ');')):
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)

    def _create_embedding_text(self, question: str, db_dump: str) -> str:
        """
        Create text for embedding by combining question and simplified schema.

        Format: "Question: {question} Schema: {simplified_schema}"
        """
        simplified_schema = self._simplify_db_dump(db_dump)
        return f"Question: {question}\nSchema: {simplified_schema}"

    def _get_cache_path(self, csv_path: str, split: str) -> Tuple[str, str, str]:
        """Get paths for cached embeddings, examples, and index."""
        cache_name = f"{os.path.basename(csv_path).replace('.csv', '')}_{split}"
        embeddings_path = os.path.join(self.cache_dir, f"{cache_name}_embeddings.npy")
        examples_path = os.path.join(self.cache_dir, f"{cache_name}_examples.pkl")
        index_path = os.path.join(self.cache_dir, f"{cache_name}_index.faiss")
        return embeddings_path, examples_path, index_path

    def build_from_csv(
        self,
        csv_path: str,
        split: str = 'train',
        ambiguous_only: bool = False,
        force_rebuild: bool = False
    ) -> None:
        """
        Build vector database from CSV file.

        IMPORTANT: This method ALWAYS uses the 'train' split to avoid data leakage.
        The split parameter is ignored and forced to 'train'.

        Args:
            csv_path: Path to CSV file
            split: IGNORED - always uses 'train' to prevent data leakage
            ambiguous_only: Only include ambiguous examples
            force_rebuild: Force rebuild even if cache exists
        """
        # Always force split to 'train' to avoid data leakage
        if split != 'train':
            print(f"WARNING: Requested split='{split}' but forcing to 'train' to avoid data leakage")
        split = 'train'

        embeddings_path, examples_path, index_path = self._get_cache_path(csv_path, split)

        # Try to load from cache
        if self.use_cache and not force_rebuild:
            if all(os.path.exists(p) for p in [embeddings_path, examples_path, index_path]):
                print(f"Loading cached vector database from {self.cache_dir}")
                self.embeddings = np.load(embeddings_path)
                with open(examples_path, 'rb') as f:
                    self.examples = pickle.load(f)
                self.index = faiss.read_index(index_path)
                print(f"Loaded {len(self.examples)} examples from cache (train split only)")
                return

        # Load data from CSV
        print(f"Building vector database from {csv_path} (train split only)")
        examples = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split - always use 'train'
                if row.get('split', '').strip() != 'train':
                    continue

                # Get question
                question = row.get('question', '').strip()
                if not question:
                    continue

                # Get question metadata
                is_ambiguous = row.get('is_ambiguous', '').strip().upper()
                question_type = row.get('question_type', '').strip()

                # Determine question_category (U, AA, AU)
                question_category = row.get('question_category', '').strip()
                if not question_category:
                    if question_type == 'unanswerable':
                        question_category = 'U'
                    elif is_ambiguous == 'TRUE':
                        question_category = 'AA'
                    elif is_ambiguous == 'FALSE':
                        question_category = 'AU'
                    else:
                        continue  # Skip if we can't determine category

                # Skip non-ambiguous if required by filter
                if ambiguous_only and question_category != 'AA':
                    continue

                # Get interpretations
                nl_interpretations = row.get('nl_interpretations', '').strip()
                if not nl_interpretations:
                    interpretations = []
                else:
                    interpretations = [interp.strip() for interp in nl_interpretations.split('\n') if interp.strip()]

                # Get db_dump
                db_dump = row.get('db_dump_processed', '').strip()
                if not db_dump:
                    db_dump = row.get('db_dump', '').strip()

                if not db_dump:
                    continue

                # Get SQL queries
                # First try ambig_queries (for AA questions with multiple interpretations)
                # Then fallback to gold_queries (for AU questions like BIRD/Spider)
                ambig_queries_str = row.get('ambig_queries', '').strip()
                gold_queries_str = row.get('gold_queries', '').strip()

                gold_queries = []

                # Try ambig_queries first (Python list literal or JSON)
                if ambig_queries_str:
                    try:
                        # Use ast.literal_eval for Python list literals with single quotes
                        import ast
                        gold_queries = ast.literal_eval(ambig_queries_str)
                    except (ValueError, SyntaxError):
                        # Fallback: try JSON parsing
                        try:
                            gold_queries = json.loads(ambig_queries_str)
                        except json.JSONDecodeError:
                            gold_queries = []

                # Fallback to gold_queries column (for BIRD/Spider unambiguous questions)
                if not gold_queries and gold_queries_str:
                    # gold_queries is typically a single SQL or newline-separated SQLs
                    if gold_queries_str.startswith('['):
                        # It's a list format
                        try:
                            import ast
                            gold_queries = ast.literal_eval(gold_queries_str)
                        except (ValueError, SyntaxError):
                            try:
                                gold_queries = json.loads(gold_queries_str)
                            except json.JSONDecodeError:
                                gold_queries = [gold_queries_str]
                    else:
                        # Single SQL or newline-separated
                        gold_queries = [q.strip() for q in gold_queries_str.split('\n\n') if q.strip()]
                        if not gold_queries:
                            gold_queries = [gold_queries_str]

                # Only set ambig_type for ambiguous questions (AA)
                # For AU and U, ambig_type should be None
                ambig_type = row.get('ambig_type', '').strip() if question_category == 'AA' else None

                examples.append({
                    'question': question,
                    'db_dump': db_dump,
                    'interpretations': interpretations,
                    'gold_queries': gold_queries,
                    'ambig_type': ambig_type,
                    'question_type': question_type,
                    'question_category': question_category,
                    'is_ambiguous': is_ambiguous,  # Store original is_ambiguous field
                })

        print(f"Loaded {len(examples)} examples from CSV")

        if len(examples) == 0:
            raise ValueError(f"No examples found in {csv_path} for split={split}")

        # Create embeddings
        print("Creating embeddings...")
        texts = [self._create_embedding_text(ex['question'], ex['db_dump']) for ex in examples]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Build FAISS index
        print("Building FAISS index...")
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine similarity for normalized vectors
        index.add(embeddings.astype(np.float32))

        # Store
        self.examples = examples
        self.embeddings = embeddings
        self.index = index

        # Save to cache
        if self.use_cache:
            print(f"Saving to cache: {self.cache_dir}")
            np.save(embeddings_path, embeddings)
            with open(examples_path, 'wb') as f:
                pickle.dump(examples, f)
            faiss.write_index(index, index_path)

        print(f"Vector database ready with {len(examples)} examples")

    def retrieve_similar(
        self,
        question: str,
        db_dump: str,
        k: int = 3,
        exclude_exact_match: bool = True
    ) -> List[Dict]:
        """
        Retrieve k most similar examples based on semantic similarity.

        Retrieves examples from ALL categories (U, AA, AU) - no filtering by category.

        Args:
            question: Query question
            db_dump: Query database schema
            k: Number of examples to retrieve
            exclude_exact_match: Exclude exact matches (useful when retrieving from same dataset)

        Returns:
            List of example dictionaries with similarity scores and metadata (including question_category)
        """
        if self.index is None or len(self.examples) == 0:
            raise ValueError("Vector database not built. Call build_from_csv() first.")

        # Handle k=0 case - return empty list
        if k <= 0:
            return []

        # Create query embedding
        query_text = self._create_embedding_text(question, db_dump)
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False  # Disable progress bar for individual queries
        )

        # Search for top k+1 (in case we need to exclude exact match)
        search_k = min(k + 1 if exclude_exact_match else k, len(self.examples))
        distances, indices = self.index.search(query_embedding.astype(np.float32), search_k)

        # Collect results (no category filtering)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            example = self.examples[idx].copy()
            example['similarity'] = float(dist)

            # Check for exact match
            if exclude_exact_match and example['question'] == question:
                continue

            results.append(example)

            if len(results) >= k:
                break

        return results

    def retrieve_similar_with_category_distribution(
        self,
        question: str,
        db_dump: str,
        question_category: str,
        k_same_category: int = 3,
        k_other_category: int = 1,
        exclude_exact_match: bool = True
    ) -> List[Dict]:
        """
        Retrieve examples with specified category distribution:
        - k_same_category examples from the same category as the query
        - k_other_category examples from the OTHER non-U category

        Since the training data has no U (Unanswerable) examples, this method only
        retrieves from AA and AU categories.

        For example, if question_category='AA' and k_same_category=3, k_other_category=1:
        - Retrieve 3 examples from AA category
        - Retrieve 1 example from AU category
        Total: 4 examples

        Args:
            question: Query question
            db_dump: Query database schema
            question_category: Category of the query question ('U', 'AA', or 'AU')
            k_same_category: Number of examples to retrieve from same category
            k_other_category: Number of examples to retrieve from the other non-U category
            exclude_exact_match: Exclude exact matches (useful when retrieving from same dataset)

        Returns:
            List of example dictionaries with similarity scores and metadata, ordered by:
            1. Examples from same category (sorted by similarity)
            2. Examples from other category (sorted by similarity)
        """
        if self.index is None or len(self.examples) == 0:
            raise ValueError("Vector database not built. Call build_from_csv() first.")

        if question_category not in ['U', 'AA', 'AU']:
            raise ValueError(f"Invalid question_category: {question_category}. Must be 'U', 'AA', or 'AU'")

        # Handle k=0 case - return empty list
        if k_same_category <= 0 and k_other_category <= 0:
            return []

        # Create query embedding
        query_text = self._create_embedding_text(question, db_dump)
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False  # Disable progress bar for individual queries
        )

        # Search for many candidates to account for category filtering
        search_k = len(self.examples)
        distances, indices = self.index.search(query_embedding.astype(np.float32), search_k)

        # Determine other category (only consider AA and AU, not U)
        if question_category == 'AA':
            other_category = 'AU'
        elif question_category == 'AU':
            other_category = 'AA'
        else:  # U
            # For U questions, get examples from both AA and AU
            other_category = None

        # Collect results by category
        same_category_results = []
        other_category_results = []

        for dist, idx in zip(distances[0], indices[0]):
            example = self.examples[idx].copy()
            example['similarity'] = float(dist)

            # Get example category
            example_category = example.get('question_category')

            # Check for exact match
            if exclude_exact_match and example['question'] == question:
                continue

            # Categorize the result
            if example_category == question_category:
                if len(same_category_results) < k_same_category:
                    same_category_results.append(example)
            elif other_category is None:
                # For U questions, get from both AA and AU
                if example_category in ['AA', 'AU']:
                    if len(other_category_results) < k_other_category:
                        other_category_results.append(example)
            elif example_category == other_category:
                if len(other_category_results) < k_other_category:
                    other_category_results.append(example)

            # Check if we have enough examples
            have_enough_same = len(same_category_results) >= k_same_category
            have_enough_other = len(other_category_results) >= k_other_category

            if have_enough_same and have_enough_other:
                break

        # Combine results: same category first, then other category
        results = same_category_results.copy()
        results.extend(other_category_results)

        return results

    def retrieve_by_ambiguity_type(
        self,
        question: str,
        db_dump: str,
        predicted_ambig_type: str,
        k_same_type: int = 5,
        k_mixed: int = 0,
        exclude_exact_match: bool = True
    ) -> List[Dict]:
        """
        Retrieve examples filtered by ambiguity type for better coverage.

        Strategy:
        - Retrieve k_same_type examples with the SAME ambiguity type as predicted
        - Optionally retrieve k_mixed examples from any ambiguity type

        Args:
            question: Query question
            db_dump: Query database schema
            predicted_ambig_type: Predicted ambiguity type ('scope', 'attachment', or 'vague')
            k_same_type: Number of examples to retrieve with same ambiguity type
            k_mixed: Number of additional examples from any type (default 0)
            exclude_exact_match: Exclude exact matches

        Returns:
            List of example dictionaries with similarity scores, ordered by similarity
        """
        if self.index is None or len(self.examples) == 0:
            raise ValueError("Vector database not built. Call build_from_csv() first.")

        # Handle k=0 case
        if k_same_type <= 0 and k_mixed <= 0:
            return []

        # Create query embedding
        query_text = self._create_embedding_text(question, db_dump)
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Search through all examples
        search_k = len(self.examples)
        distances, indices = self.index.search(query_embedding.astype(np.float32), search_k)

        # Collect results
        same_type_results = []
        mixed_results = []

        for dist, idx in zip(distances[0], indices[0]):
            example = self.examples[idx].copy()
            example['similarity'] = float(dist)

            # Check for exact match
            if exclude_exact_match and example['question'] == question:
                continue

            # Get example ambiguity type
            example_ambig_type = example.get('ambig_type')

            # Categorize the result
            if example_ambig_type == predicted_ambig_type:
                if len(same_type_results) < k_same_type:
                    same_type_results.append(example)
            else:
                if len(mixed_results) < k_mixed:
                    mixed_results.append(example)

            # Check if we have enough
            if len(same_type_results) >= k_same_type and len(mixed_results) >= k_mixed:
                break

        # Combine results: same type first, then mixed
        results = same_type_results.copy()
        results.extend(mixed_results)

        return results

    def format_examples_for_prompt(
        self,
        examples: List[Dict],
        include_sql: bool = False,
        classification_probs: dict = None,
        question_category: str = None,
        k_same_category: int = None,
        k_other_category: int = None
    ) -> str:
        """
        Format retrieved examples for inclusion in prompt.

        Args:
            examples: List of example dictionaries from retrieve_similar()
            include_sql: Whether to include SQL queries in examples
            classification_probs: Dict of classification probabilities {'U': 0.1, 'AA': 0.8, 'AU': 0.1}
            question_category: Predicted category with highest probability ('U', 'AA', or 'AU')
            k_same_category: Number of examples from same category (for display purposes)
            k_other_category: Number of examples from other category (for display purposes)

        Returns:
            Formatted string for prompt
        """
        # Category name mapping for display
        CATEGORY_NAMES = {
            'U': 'Unanswerable',
            'AA': 'Answerable-Ambiguous',
            'AU': 'Answerable-Unambiguous'
        }

        if not examples:
            return ""

        formatted_parts = []

        # Add classification information if provided
        if classification_probs and question_category:
            classification_text = "**Question Classification:**\n\n"
            classification_text += "Classification probabilities:\n"
            for cat in ['U', 'AA', 'AU']:
                prob = classification_probs.get(cat, 0.0)
                cat_name = CATEGORY_NAMES.get(cat, cat)
                classification_text += f"  - {cat_name}: {prob:.4f}\n"

            cat_name = CATEGORY_NAMES.get(question_category, question_category)
            classification_text += f"\nThe question is most likely: **{cat_name}**\n\n"

            # Determine the other category
            if question_category == 'AA':
                other_category = 'AU'
            elif question_category == 'AU':
                other_category = 'AA'
            else:
                other_category = 'AA/AU'

            other_cat_name = CATEGORY_NAMES.get(other_category, other_category)

            if k_same_category and k_other_category:
                classification_text += f"Showing {k_same_category} example(s) from **{cat_name}** category "
                classification_text += f"and {k_other_category} example(s) from **{other_cat_name}** category.\n\n"

            classification_text += "---\n\n"
            formatted_parts.append(classification_text)

        # Format examples
        for i, ex in enumerate(examples, 1):
            # Get example category and spell it out
            ex_category = ex.get('question_category', 'Unknown')
            ex_category_name = CATEGORY_NAMES.get(ex_category, ex_category)

            # Extract schema without INSERT statements (keep only CREATE TABLE)
            db_dump = ex.get('db_dump', '')
            # Remove INSERT statements but keep CREATE TABLE statements
            schema_lines = []
            for line in db_dump.split('\n'):
                if not line.strip().startswith('INSERT INTO'):
                    schema_lines.append(line)
            schema_only = '\n'.join(schema_lines).strip()

            # Include ambiguity type for ambiguous questions
            ambig_type = ex.get('ambig_type', '')
            if ambig_type and ex_category == 'AA':
                part = f"""**Example {i}** (Category: {ex_category_name}, Ambiguity Type: {ambig_type}):

Database Schema:
{schema_only}

Question: {ex['question']}"""
            else:
                part = f"""**Example {i}** (Category: {ex_category_name}):

Database Schema:
{schema_only}

Question: {ex['question']}"""

            # Get SQL queries if available (now always a list from ambig_queries)
            sql_queries = []
            if include_sql and ex.get('gold_queries'):
                sql_queries = ex['gold_queries'] if isinstance(ex['gold_queries'], list) else []

            # Format SQL queries only (no interpretations) - matches AmbiQT format
            if include_sql and sql_queries:
                num_queries = len(sql_queries)
                if num_queries > 1:
                    part += f"\n\n**This question has {num_queries} valid SQL queries:**"
                else:
                    part += f"\n\n**This question has 1 valid SQL query:**"

                for j, sql in enumerate(sql_queries, 1):
                    part += f"\n\nSQL {j}: {sql}"

            formatted_parts.append(part)

        return "\n\n".join(formatted_parts)
