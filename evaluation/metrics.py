from collections import Counter
import sqlite3

from .exceptions import DublicatesError, MetricError, GoldQueryExecutionError, PredQueryExecutionError


# ============================================================================
# Value Normalization Functions (for enhanced flex evaluation)
# ============================================================================

def normalize_value(value):
    """
    Normalize a single value for flexible comparison.

    Handles:
    - Number formatting (1.0 == 1, "1" == 1)
    - String case (case-insensitive)
    - Whitespace trimming
    - NULL equivalents

    Args:
        value: Any value from SQL result

    Returns:
        Normalized value for comparison
    """
    if value is None:
        return None

    # Try as number first
    if isinstance(value, (int, float)):
        # Normalize floats: 1.0 becomes 1
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    # String normalization
    if isinstance(value, str):
        # Strip whitespace
        normalized = value.strip()

        # Handle common NULL representations
        if normalized.lower() in ('null', 'none', ''):
            return None

        # Try to parse as number for mixed type comparisons (e.g., "1" vs 1)
        try:
            num = float(normalized)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            # Keep as string, but case-insensitive
            return normalized.lower()

    return value


def normalize_row(row):
    """
    Normalize a row (tuple of values) for comparison.

    Args:
        row: Tuple or list of values

    Returns:
        Tuple of normalized values
    """
    return tuple(normalize_value(v) for v in row)


def extract_number(value):
    """
    Extract numeric value from various formats.

    Handles:
    - Plain numbers
    - Percentages ("50%")
    - Formatted numbers ("1,234.56")

    Args:
        value: Any value

    Returns:
        Float if parseable, None otherwise
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Remove common formatting characters
        cleaned = value.strip().replace('%', '').replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


# ============================================================================
# Semantic Equivalence Detection (without LLM)
# ============================================================================

def is_percentage_equivalent(pred_results, gold_results):
    """
    Check if results differ only by percentage formatting.

    Examples:
    - 0.5 vs 50 (one multiplied by 100)
    - "50%" vs 50 or 0.5

    Args:
        pred_results: Predicted query results
        gold_results: Gold query results

    Returns:
        True if equivalent with percentage conversion
    """
    if len(pred_results) != len(gold_results):
        return False

    for pred_row, gold_row in zip(pred_results, gold_results):
        if len(pred_row) != len(gold_row):
            return False

        for pred_val, gold_val in zip(pred_row, gold_row):
            # Try to extract numeric values
            pred_num = extract_number(pred_val)
            gold_num = extract_number(gold_val)

            if pred_num is None or gold_num is None:
                # Non-numeric values must match exactly
                if normalize_value(pred_val) != normalize_value(gold_val):
                    return False
                continue

            # Check if one is 100x the other (percentage conversion)
            if gold_num != 0:
                ratio = pred_num / gold_num
                # Allow either same value or 100x multiplier
                if not (0.99 < ratio < 1.01 or 99 < ratio < 101 or 0.0099 < ratio < 0.0101):
                    return False
            elif pred_num != 0:
                return False

    return True


def is_boolean_equivalent(pred_results, gold_results):
    """
    Check if results differ only in boolean representation.

    Examples:
    - 1 vs "YES" vs "true" vs True
    - 0 vs "NO" vs "false" vs False

    Args:
        pred_results: Predicted query results
        gold_results: Gold query results

    Returns:
        True if equivalent with boolean normalization
    """
    bool_maps = {
        1: ['1', 'true', 'yes', 't', 'y', True],
        0: ['0', 'false', 'no', 'f', 'n', False],
    }

    def normalize_bool(val):
        """Normalize boolean value to 0 or 1."""
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, int) and val in (0, 1):
            return val
        if isinstance(val, str):
            val_lower = val.lower().strip()
            for bool_val, representations in bool_maps.items():
                if val_lower in [str(r).lower() for r in representations]:
                    return bool_val
        return val

    if len(pred_results) != len(gold_results):
        return False

    for pred_row, gold_row in zip(pred_results, gold_results):
        if len(pred_row) != len(gold_row):
            return False

        pred_normalized = [normalize_bool(v) for v in pred_row]
        gold_normalized = [normalize_bool(v) for v in gold_row]

        if pred_normalized != gold_normalized:
            return False

    return True


def is_aggregation_equivalent(pred_results, gold_results, tolerance=0.01):
    """
    Check if numeric results are equivalent within tolerance.

    Useful for floating-point aggregations (AVG, SUM with divisions, etc.)

    Args:
        pred_results: Predicted query results
        gold_results: Gold query results
        tolerance: Maximum allowed difference for numeric values

    Returns:
        True if equivalent within tolerance
    """
    if len(pred_results) != len(gold_results):
        return False

    for pred_row, gold_row in zip(pred_results, gold_results):
        if len(pred_row) != len(gold_row):
            return False

        for pred_val, gold_val in zip(pred_row, gold_row):
            # Normalize first
            pred_norm = normalize_value(pred_val)
            gold_norm = normalize_value(gold_val)

            # Both numeric - check tolerance
            if isinstance(pred_norm, (int, float)) and isinstance(gold_norm, (int, float)):
                if abs(pred_norm - gold_norm) > tolerance:
                    return False
            else:
                # Non-numeric must match exactly after normalization
                if pred_norm != gold_norm:
                    return False

    return True


def detect_semantic_equivalence(pred_results, gold_results):
    """
    Detect common patterns of semantic equivalence without LLM.

    Checks multiple heuristic patterns:
    - Percentage formatting differences
    - Boolean representation differences
    - Floating-point aggregation tolerance

    Args:
        pred_results: Predicted query results
        gold_results: Gold query results

    Returns:
        True if semantically equivalent by any heuristic
    """
    # Check 1: Percentage equivalence
    if is_percentage_equivalent(pred_results, gold_results):
        return True

    # Check 2: Boolean equivalence
    if is_boolean_equivalent(pred_results, gold_results):
        return True

    # Check 3: Aggregation tolerance (small floating point differences)
    if is_aggregation_equivalent(pred_results, gold_results, tolerance=0.01):
        return True

    return False

def sort_key(x):
    if x is None:
        return (0, '')  # Treat None as the smallest value
    elif isinstance(x, (int, float)):
        return (1, float(x))  # Handle numerical types uniformly
    else:
        return (2, str(x))  # Convert all other types to string for consistent comparison

def sort_with_different_types(arr):
    sorted_arr = sorted(arr, key=sort_key)
    return sorted_arr

def compare_query_results(predicted_results, gold_results, order_by=False):
    # Handle empty results: both empty = match, one empty = no match
    if not predicted_results and not gold_results:
        return True
    if not predicted_results or not gold_results:
        return False

    if order_by:
        if len(gold_results) != len(predicted_results):
            return False

        if any(len(row) != len(gold_results[0]) for row in gold_results + predicted_results):
            return False

        for gold_row, predicted_row in zip(gold_results, predicted_results):
            if tuple(sort_with_different_types(gold_row)) != tuple(sort_with_different_types(predicted_row)):
                return False
        return True
    else:
        flat_gold = Counter(item for row in gold_results for item in row)
        flat_predicted = Counter(item for row in predicted_results for item in row)

        return flat_gold == flat_predicted


def find_best_column_mapping(predicted_results, gold_results):
    """
    Find the best mapping of predicted columns to gold columns by value similarity.

    Args:
        predicted_results: Predicted query results (rows of tuples)
        gold_results: Gold query results (rows of tuples)

    Returns:
        List of column indices in predicted that match gold columns, or None if no good match
    """
    if not predicted_results or not gold_results:
        return None

    pred_num_cols = len(predicted_results[0])
    gold_num_cols = len(gold_results[0])

    # If predicted has fewer columns than gold, can't match
    if pred_num_cols < gold_num_cols:
        return None

    # Try all possible column combinations
    from itertools import combinations, permutations

    # For each combination of pred_num_cols choose gold_num_cols
    best_mapping = None
    best_score = 0

    # Limit search to prevent combinatorial explosion
    if pred_num_cols > 10 or gold_num_cols > 5:
        return None

    for col_combo in combinations(range(pred_num_cols), gold_num_cols):
        for col_perm in permutations(col_combo):
            # Extract predicted columns according to this mapping
            score = 0
            for pred_row, gold_row in zip(predicted_results, gold_results):
                for pred_col_idx, gold_col_idx in zip(col_perm, range(gold_num_cols)):
                    pred_val = normalize_value(pred_row[pred_col_idx])
                    gold_val = normalize_value(gold_row[gold_col_idx])
                    if pred_val == gold_val:
                        score += 1

            # If this mapping matches all cells, return it
            if score == len(predicted_results) * gold_num_cols:
                return list(col_perm)

            # Track best partial match
            if score > best_score:
                best_score = score
                best_mapping = list(col_perm)

    # Return best mapping if it matches at least 50% of cells
    threshold = len(predicted_results) * gold_num_cols * 0.5
    if best_score >= threshold:
        return best_mapping

    return None


def project_columns(results, column_indices):
    """
    Extract specific columns from query results.

    Args:
        results: Query results (rows of tuples)
        column_indices: List of column indices to extract

    Returns:
        New results with only the specified columns
    """
    projected = []
    for row in results:
        projected_row = tuple(row[i] for i in column_indices)
        projected.append(projected_row)
    return projected


def compare_with_normalization(predicted_results, gold_results, order_by=False):
    """
    Compare results with value normalization applied.

    Args:
        predicted_results: Predicted query results
        gold_results: Gold query results
        order_by: Whether this is an ordered query

    Returns:
        True if results match after normalization
    """
    # Check dimensions
    if len(predicted_results) != len(gold_results):
        return False

    if not predicted_results or not gold_results:
        return len(predicted_results) == len(gold_results) == 0

    if len(predicted_results[0]) != len(gold_results[0]):
        return False

    # Normalize all values
    pred_normalized = [normalize_row(row) for row in predicted_results]
    gold_normalized = [normalize_row(row) for row in gold_results]

    if order_by:
        # For ordered queries, compare row by row
        return pred_normalized == gold_normalized
    else:
        # For unordered queries, compare as sets
        return Counter(pred_normalized) == Counter(gold_normalized)


def subset_matching_with_normalization(predicted_results, gold_results):
    """
    Check if gold data is a subset of predicted data (with normalization).

    Args:
        predicted_results: Predicted query results
        gold_results: Gold query results

    Returns:
        True if all gold data exists in predicted data
    """
    # Normalize all rows
    pred_normalized = [normalize_row(row) for row in predicted_results]
    gold_normalized = [normalize_row(row) for row in gold_results]

    # Flatten to individual cells and use Counter for subset checking
    flat_gold = Counter(item for row in gold_normalized for item in row)
    flat_predicted = Counter(item for row in pred_normalized for item in row)

    # Check if all gold items are present in predicted (allows extras)
    for item, count in flat_gold.items():
        if flat_predicted[item] < count:
            return False

    return True


def compare_query_results_flex(predicted_results, gold_results, order_by=False, interpretation_similarity=None):
    """
    Enhanced flexible comparison that allows extra columns in predicted results.

    This compares only the data present in gold results, allowing predicted
    results to contain additional columns. Useful when the model generates
    more informative queries than the ground truth.

    Strategies (tried in order):
    1. Same dimensions: Try semantic equivalence detection
    2. Same dimensions: Try comparison with normalization
    3. Different columns: Try column projection and matching
    4. Fallback: Subset matching (only if interpretation_similarity > 0.8)

    Args:
        predicted_results: Predicted query results
        gold_results: Gold query results
        order_by: Whether ORDER BY is present in query
        interpretation_similarity: Cosine similarity between question and interpretation (0-1).
                                   If provided, subset matching only used when > 0.8

    Returns:
        True if predicted results match or contain gold results
    """
    # Handle empty results: both empty = match, one empty = no match
    if not predicted_results and not gold_results:
        return True
    if not predicted_results or not gold_results:
        return False

    # Get dimensions
    gold_num_rows = len(gold_results)
    pred_num_rows = len(predicted_results)
    gold_num_cols = len(gold_results[0]) if gold_results else 0
    pred_num_cols = len(predicted_results[0]) if predicted_results else 0

    # Strategy 1: Same dimensions - try semantic equivalence first
    if gold_num_cols == pred_num_cols and gold_num_rows == pred_num_rows:
        # Check for percentage, boolean, or aggregation equivalence
        if detect_semantic_equivalence(predicted_results, gold_results):
            return True

        # Check with normalization
        if compare_with_normalization(predicted_results, gold_results, order_by):
            return True

        # Even with same column count, columns might be in different order
        # Try column mapping to handle reordered columns
        column_mapping = find_best_column_mapping(predicted_results, gold_results)
        if column_mapping is not None and column_mapping != list(range(gold_num_cols)):
            # Found a non-trivial mapping (columns are reordered)
            projected_pred = project_columns(predicted_results, column_mapping)
            if detect_semantic_equivalence(projected_pred, gold_results):
                return True
            if compare_with_normalization(projected_pred, gold_results, order_by):
                return True

    # Strategy 2: Predicted has more columns - try column projection
    if pred_num_cols > gold_num_cols:
        # For ordered queries or when column count is reasonable
        if pred_num_rows == gold_num_rows:
            column_mapping = find_best_column_mapping(predicted_results, gold_results)
            if column_mapping is not None:
                # Extract matching columns from predicted
                projected_pred = project_columns(predicted_results, column_mapping)

                # Try semantic equivalence on projected results
                if detect_semantic_equivalence(projected_pred, gold_results):
                    return True

                # Try normalized comparison on projected results
                if compare_with_normalization(projected_pred, gold_results, order_by):
                    return True

    # Strategy 3: Predicted has fewer columns - check if it's a valid subset
    # (Gold might have requested extra columns we didn't generate)
    if pred_num_cols < gold_num_cols and pred_num_rows == gold_num_rows:
        column_mapping = find_best_column_mapping(gold_results, predicted_results)
        if column_mapping is not None:
            # Extract matching columns from gold
            projected_gold = project_columns(gold_results, column_mapping)

            # Try semantic equivalence
            if detect_semantic_equivalence(predicted_results, projected_gold):
                return True

            # Try normalized comparison
            if compare_with_normalization(predicted_results, projected_gold, order_by):
                return True

    # Strategy 3: Fallback for unordered queries - subset matching
    # Only use if interpretation similarity > 0.7 (high confidence alternative interpretation)
    if not order_by:
        # Only use subset matching when interpretation is highly similar to question
        # This avoids accepting overly broad/incorrect queries
        if interpretation_similarity is not None and interpretation_similarity > 0.7:
            return subset_matching_with_normalization(predicted_results, gold_results)

    # FINAL FALLBACK: Use strict comparison as last resort
    # Flex should match everything strict matches (flex must be >= strict)
    # This ensures flex doesn't miss matches that strict finds
    if compare_query_results(predicted_results, gold_results, order_by):
        return True

    # For ordered queries with unmatched dimensions, or low similarity, fail
    return False

def duplicate_exact(results):
    # Convert each result set into a tuple of tuples to make them hashable
    hashable_results = [tuple(map(tuple, result)) for result in results]
    
    # Use a set to identify duplicates
    seen = set()
    for idx, result in enumerate(hashable_results):
        if result in seen:
            return True, result, idx + 1
        seen.add(result)
    return False, None, None

def count_unique_results(results):
    # Use a set to identify duplicates
    seen = set()
    
    for result in results:
        if isinstance(result, PredQueryExecutionError):
            seen.add(None)
        else:
            hashable_result = tuple(map(tuple, result))
            seen.add(hashable_result)
    
    return len(seen)

def remove_duplicate_results(all_pred_exec_outputs):
    # List of keys to remove
    keys_to_remove = []

    # List of queries already processed
    processed_queries = list(all_pred_exec_outputs.keys())
    
    for i, query1 in enumerate(processed_queries):
        if query1 in keys_to_remove:
            continue  # Skip if this key is already marked for removal
        
        result1 = all_pred_exec_outputs[query1]

        if isinstance(result1, PredQueryExecutionError):
            result1 = None
        
        for query2 in processed_queries[i+1:]:
            if query2 in keys_to_remove:
                continue  # Skip if this key is already marked for removal
            
            result2 = all_pred_exec_outputs[query2]
            if isinstance(result2, PredQueryExecutionError):
                result2 = None
                if result1 is None:
                    keys_to_remove.append(query2)
                continue

            order_by = 'order by' in query1.lower() or 'order by' in query2.lower()
            
            if compare_query_results(result1, result2, order_by):
                # Mark query2 for removal if it's a duplicate of query1
                keys_to_remove.append(query2)
    
    # Remove the marked keys from the dictionary
    for key in keys_to_remove:
        del all_pred_exec_outputs[key]
    
    return all_pred_exec_outputs


def evaluate_predicted_statements(file_name, pred_statements, gold_sql_queries, remove_duplicates_predictions=False, calculate_unique=False, verbose=False, return_pred_exec_outputs=False):
    conn = sqlite3.connect(file_name)
    cursor = conn.cursor()

    all_gold_exec_outputs = {}
    exec_acc_per_gold_queries = {}
    for query in gold_sql_queries:
        try:
            cursor.execute(query)
            all_gold_exec_outputs[query] = cursor.fetchall()
            exec_acc_per_gold_queries[query] = False
        except sqlite3.DatabaseError as e:
            raise GoldQueryExecutionError(query, e)

    has_duplicates, duplicates, duplicate_idx = duplicate_exact(list(all_gold_exec_outputs.values()))
    if has_duplicates:
        print("duplicates", duplicates)
        raise DublicatesError(duplicates, duplicate_idx)

    all_pred_exec_outputs = {}
    num_queries = len(pred_statements)
    pred_statements = list(set(pred_statements))
    execution_errors = []
    for query in pred_statements:
        try:
            cursor.execute(query)
            all_pred_exec_outputs[query] = cursor.fetchall()
        except sqlite3.DatabaseError as e:
            all_pred_exec_outputs[query] = PredQueryExecutionError(query, e)
            execution_errors.append(PredQueryExecutionError(query, e).to_dict())

            error_message = str(e).lower()
            ignore_errors = ["no such table", "no such column", "ambiguous"]

            if verbose and not any(ignore in error_message for ignore in ignore_errors):
                print(f'\nCannot execute {query}\nError: {e}\n{file_name}\n\n')

    if remove_duplicates_predictions and all_pred_exec_outputs:
        all_pred_exec_outputs = remove_duplicate_results(all_pred_exec_outputs)
        pred_statements = list(all_pred_exec_outputs.keys())

    exec_acc_per_pred_queries = {query: False for query in pred_statements} 
    for pred_query, pred_exec_output in all_pred_exec_outputs.items():
        if isinstance(pred_exec_output, PredQueryExecutionError):
            continue
        for gold_query, gold_exec_output in all_gold_exec_outputs.items():
            try:
                if 'order by' in gold_query.lower():
                    is_same = compare_query_results(pred_exec_output, gold_exec_output, order_by=True)
                else:
                    is_same = compare_query_results(pred_exec_output, gold_exec_output, order_by=False)
                if is_same:
                    exec_acc_per_gold_queries[gold_query] = True
                    exec_acc_per_pred_queries[pred_query] = True
            except Exception as e:
                raise MetricError(pred_exec_output, gold_exec_output, pred_query, gold_query, e)
                
    recall = sum(exec_acc_per_gold_queries.values()) / len(gold_sql_queries)
    all_found =  sum(exec_acc_per_gold_queries.values()) == len(gold_sql_queries)
    precision = sum(exec_acc_per_pred_queries.values()) / len(pred_statements) if pred_statements else 0

    # Ensure metrics are bounded [0, 1]
    recall = min(1.0, max(0.0, recall))
    precision = min(1.0, max(0.0, precision))

    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    if calculate_unique:
        if remove_duplicates_predictions:
            all_pred_exec_outputs_wo_duplicates = all_pred_exec_outputs
        else:
            all_pred_exec_outputs_wo_duplicates = remove_duplicate_results(all_pred_exec_outputs)
        unique_results = count_unique_results(list(all_pred_exec_outputs.values()))
        unique_results_filtered = count_unique_results(list(all_pred_exec_outputs_wo_duplicates.values()))
    else:
        unique_results = unique_results_filtered = 0

    metrics = {
                'recall': recall,
                'precision': precision,
                'one_found': recall > 0,
                'f1_score': f1_score,
                'num_queries': num_queries,
                'num_unique_queries': len(pred_statements),
                'unique_results': unique_results,
                'unique_results_filtered': unique_results_filtered,
                'execution_errors': execution_errors,
                'all_found': all_found
            }
    
    if return_pred_exec_outputs:
        metrics['pred_exec_outputs'] = all_pred_exec_outputs

    conn.close()
    return metrics


def evaluate_predicted_statements_flex(file_name, pred_statements, gold_sql_queries, remove_duplicates_predictions=False, calculate_unique=False, verbose=False, return_pred_exec_outputs=False, interpretation_similarities=None):
    """
    Flexible evaluation that allows extra columns in predicted results.

    This version uses compare_query_results_flex() which considers predicted
    results as matching if they contain all the gold data (allowing extra columns).

    Args:
        file_name: Path to SQLite database
        pred_statements: List of predicted SQL queries
        gold_sql_queries: List of gold SQL queries
        remove_duplicates_predictions: Whether to remove duplicate predictions
        calculate_unique: Whether to calculate unique result counts
        verbose: Whether to print verbose error messages
        return_pred_exec_outputs: Whether to return prediction execution outputs
        interpretation_similarities: Optional dict mapping predicted queries to their
                                    interpretation similarity scores (0-1). Used to
                                    control subset matching in flex evaluation.

    Returns:
        dict with keys: recall_flex, precision_flex, f1_flex, all_found_flex,
                        one_found_flex, num_queries, execution_errors, etc.
    """
    conn = sqlite3.connect(file_name)
    cursor = conn.cursor()

    all_gold_exec_outputs = {}
    exec_acc_per_gold_queries = {}
    for query in gold_sql_queries:
        try:
            cursor.execute(query)
            all_gold_exec_outputs[query] = cursor.fetchall()
            exec_acc_per_gold_queries[query] = False
        except sqlite3.DatabaseError as e:
            raise GoldQueryExecutionError(query, e)

    has_duplicates, duplicates, duplicate_idx = duplicate_exact(list(all_gold_exec_outputs.values()))
    if has_duplicates:
        raise DublicatesError(duplicates, duplicate_idx)

    all_pred_exec_outputs = {}
    num_queries = len(pred_statements)
    pred_statements = list(set(pred_statements))
    execution_errors = []
    for query in pred_statements:
        try:
            cursor.execute(query)
            all_pred_exec_outputs[query] = cursor.fetchall()
        except sqlite3.DatabaseError as e:
            all_pred_exec_outputs[query] = PredQueryExecutionError(query, e)
            execution_errors.append(PredQueryExecutionError(query, e).to_dict())

            error_message = str(e).lower()
            ignore_errors = ["no such table", "no such column", "ambiguous"]

            if verbose and not any(ignore in error_message for ignore in ignore_errors):
                print(f'\nCannot execute {query}\nError: {e}\n{file_name}\n\n')

    if remove_duplicates_predictions and all_pred_exec_outputs:
        all_pred_exec_outputs = remove_duplicate_results(all_pred_exec_outputs)
        pred_statements = list(all_pred_exec_outputs.keys())

    exec_acc_per_pred_queries = {query: False for query in pred_statements}
    for pred_query, pred_exec_output in all_pred_exec_outputs.items():
        if isinstance(pred_exec_output, PredQueryExecutionError):
            continue

        # Get interpretation similarity for this predicted query
        similarity = None
        if interpretation_similarities is not None:
            similarity = interpretation_similarities.get(pred_query, None)

        for gold_query, gold_exec_output in all_gold_exec_outputs.items():
            try:
                if 'order by' in gold_query.lower():
                    is_same = compare_query_results_flex(pred_exec_output, gold_exec_output, order_by=True, interpretation_similarity=similarity)
                else:
                    is_same = compare_query_results_flex(pred_exec_output, gold_exec_output, order_by=False, interpretation_similarity=similarity)
                if is_same:
                    exec_acc_per_gold_queries[gold_query] = True
                    exec_acc_per_pred_queries[pred_query] = True
            except Exception as e:
                raise MetricError(pred_exec_output, gold_exec_output, pred_query, gold_query, e)

    recall_flex = sum(exec_acc_per_gold_queries.values()) / len(gold_sql_queries)
    all_found_flex = sum(exec_acc_per_gold_queries.values()) == len(gold_sql_queries)
    precision_flex = sum(exec_acc_per_pred_queries.values()) / len(pred_statements) if pred_statements else 0

    # Ensure metrics are bounded [0, 1]
    recall_flex = min(1.0, max(0.0, recall_flex))
    precision_flex = min(1.0, max(0.0, precision_flex))

    f1_flex = 2 * precision_flex * recall_flex / (precision_flex + recall_flex) if precision_flex + recall_flex > 0 else 0

    if calculate_unique:
        if remove_duplicates_predictions:
            all_pred_exec_outputs_wo_duplicates = all_pred_exec_outputs
        else:
            all_pred_exec_outputs_wo_duplicates = remove_duplicate_results(all_pred_exec_outputs)
        unique_results = count_unique_results(list(all_pred_exec_outputs.values()))
        unique_results_filtered = count_unique_results(list(all_pred_exec_outputs_wo_duplicates.values()))
    else:
        unique_results = unique_results_filtered = 0

    metrics = {
                'recall_flex': recall_flex,
                'precision_flex': precision_flex,
                'one_found_flex': recall_flex > 0,
                'f1_flex': f1_flex,
                'num_queries': num_queries,
                'num_unique_queries': len(pred_statements),
                'unique_results': unique_results,
                'unique_results_filtered': unique_results_filtered,
                'execution_errors': execution_errors,
                'all_found_flex': all_found_flex
            }

    if return_pred_exec_outputs:
        metrics['pred_exec_outputs'] = all_pred_exec_outputs

    conn.close()
    return metrics
