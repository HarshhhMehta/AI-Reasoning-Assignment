"""
evaluation.py
=============
Evaluation pipeline for the AI Reasoning System.

Computes:
- Accuracy, precision, recall, F1
- Confusion matrix
- Parsing failure rate
- Inference timing statistics
- Per-dataset and per-split breakdowns
"""
from __future__ import annotations
import time
import logging
from typing import Optional
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure Python metric helpers (avoids sklearn C-extension DLL policy issues)
# ---------------------------------------------------------------------------

def compute_metrics_pure_python(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float, list[list[int]]]:
    """Calculate precision, recall, F1, and confusion matrix in pure Python."""
    if not y_true:
        return 0.0, 0.0, 0.0, [[0, 0], [0, 0]]
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    cm = [[tn, fp], [fn, tp]]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, cm


from src.representation import Example
from src.reasoning_engine import ReasoningEngine, InferenceResult


# ---------------------------------------------------------------------------
# Per-example evaluation record
# ---------------------------------------------------------------------------

class EvalRecord:
    """Holds the evaluation outcome for a single example."""

    def __init__(self, example: Example, result: InferenceResult) -> None:
        self.example = example
        self.result = result
        self.correct: Optional[bool] = None
        if example.label is not None:
            self.correct = (result.prediction == example.label)

    def __repr__(self) -> str:
        return (f"EvalRecord(id={self.example.id!r}, "
                f"pred={self.result.prediction}, "
                f"gold={self.example.label}, "
                f"correct={self.correct})")


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluates the reasoning engine on a list of examples.

    Parameters
    ----------
    engine : ReasoningEngine
        The reasoning engine to evaluate.
    verbose : bool
        If True, print per-example outcomes.
    """

    def __init__(self, engine: ReasoningEngine, verbose: bool = False) -> None:
        self.engine = engine
        self.verbose = verbose

    def evaluate(self, examples: list[Example],
                 tag: str = "eval") -> dict:
        """Run evaluation and return a metrics dict.

        Parameters
        ----------
        examples : list[Example]
            The examples to evaluate (must have ground-truth labels).
        tag : str
            Label for the report (e.g. 'ruletaker/validation').

        Returns
        -------
        dict with keys:
            accuracy, precision, recall, f1,
            n_total, n_labeled, n_correct, n_parse_fail,
            confusion_matrix, timing_stats, records
        """
        records: list[EvalRecord] = []
        parse_times: list[float] = []
        bc_times: list[float] = []
        fc_times: list[float] = []
        total_times: list[float] = []
        parse_failures = 0

        t_eval_start = time.perf_counter()

        for i, ex in enumerate(examples):
            ir = self.engine.predict_example(ex)
            rec = EvalRecord(ex, ir)
            records.append(rec)

            parse_times.append(ir.parse_time)
            bc_times.append(ir.bc_time)
            fc_times.append(ir.fc_time)
            total_times.append(ir.total_time)

            if ir.query_lit is None:
                parse_failures += 1

            if self.verbose or (i < 5):
                status = '[OK]' if rec.correct else ('[X]' if rec.correct is False else '[?]')
                print(f"  [{i+1}/{len(examples)}] {status} "
                      f"id={ex.id!r} pred={ir.prediction} gold={ex.label} "
                      f"method={ir.method} time={ir.total_time*1000:.1f}ms")

        wall_time = time.perf_counter() - t_eval_start

        # Filter to labeled examples for accuracy
        labeled = [r for r in records if r.example.label is not None]
        correct = [r for r in labeled if r.correct]

        n_total = len(records)
        n_labeled = len(labeled)
        n_correct = len(correct)
        accuracy = 100.0 * n_correct / n_labeled if n_labeled > 0 else 0.0

        # Build y_true and y_pred
        y_true = [1 if r.example.label else 0 for r in labeled]
        y_pred = [1 if r.result.prediction else 0 for r in labeled]

        precision, recall, f1, cm = compute_metrics_pure_python(y_true, y_pred)

        def safe_stats(lst: list[float]) -> dict:
            if not lst:
                return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            return {
                'mean': statistics.mean(lst),
                'min': min(lst),
                'max': max(lst),
                'median': statistics.median(lst),
            }

        timing_stats = {
            'parse': safe_stats(parse_times),
            'backward_chain': safe_stats(bc_times),
            'forward_chain': safe_stats(fc_times),
            'total_per_query': safe_stats(total_times),
            'wall_time_total': wall_time,
        }

        metrics = {
            'tag': tag,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_total': n_total,
            'n_labeled': n_labeled,
            'n_correct': n_correct,
            'n_parse_fail': parse_failures,
            'parse_fail_rate': 100.0 * parse_failures / n_total if n_total > 0 else 0.0,
            'confusion_matrix': cm,
            'timing_stats': timing_stats,
            'records': records,
        }

        self._print_report(metrics)
        return metrics

    def _print_report(self, metrics: dict) -> None:
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {metrics['tag']}")
        print(f"{'='*60}")
        print(f"  Total examples:   {metrics['n_total']}")
        print(f"  Labeled examples: {metrics['n_labeled']}")
        print(f"  Correct:          {metrics['n_correct']}")
        print(f"  Accuracy:         {metrics['accuracy']:.2f}%")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 Score:         {metrics['f1']:.4f}")
        print(f"  Parse failures:   {metrics['n_parse_fail']} "
              f"({metrics['parse_fail_rate']:.1f}%)")
        if metrics['confusion_matrix']:
            print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
        ts = metrics['timing_stats']
        print(f"\n  Timing (per query):")
        print(f"    Parse:         {ts['parse']['mean']*1000:.3f}ms avg")
        print(f"    Backward:      {ts['backward_chain']['mean']*1000:.3f}ms avg")
        print(f"    Forward:       {ts['forward_chain']['mean']*1000:.3f}ms avg")
        print(f"    Total:         {ts['total_per_query']['mean']*1000:.3f}ms avg")
        print(f"    Wall time:     {ts['wall_time_total']:.2f}s total")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Submission CSV generator
# ---------------------------------------------------------------------------

def generate_submission_csv(records: list[EvalRecord],
                            output_path: str,
                            id_field: str = 'id',
                            label_field: str = 'predicted_label') -> None:
    """Write predictions to a CSV submission file.

    Parameters
    ----------
    records : list[EvalRecord]
        Evaluation records containing predictions.
    output_path : str
        Path for the output CSV file.
    id_field : str
        Column name for the example ID.
    label_field : str
        Column name for the predicted label.
    """
    import csv
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_ids: set[str] = set()
    rows = []
    for rec in records:
        ex_id = rec.example.id
        if ex_id in seen_ids:
            logger.warning(f"Duplicate ID '{ex_id}' in submission; keeping first.")
            continue
        seen_ids.add(ex_id)
        rows.append({id_field: ex_id, label_field: rec.result.prediction})

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[id_field, label_field])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[submission.csv] Written {len(rows)} rows to: {output_path}")

    # Validate
    assert len(rows) == len(records) or len(rows) == len(seen_ids), \
        "Row count mismatch after duplicate removal"
    label_vals = set(r[label_field] for r in rows)
    assert label_vals.issubset({True, False}), \
        f"Unexpected label values in submission: {label_vals}"
    print(f"  Validation: OK | Labels present: {label_vals}")
