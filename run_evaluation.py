"""
run_evaluation.py
=================
Main evaluation script for the AI Reasoning System.

Usage:
    python run_evaluation.py [--max N] [--dataset {ruletaker,counterbench,folio,all}]

Runs evaluation on the specified datasets, generates submission.csv,
and prints accuracy + timing results.
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset_loader import load_all_datasets, load_ruletaker, load_counterbench, load_folio
from src.reasoning_engine import ReasoningEngine
from src.evaluation import Evaluator, EvalRecord, generate_submission_csv


def main():
    parser = argparse.ArgumentParser(description='AI Reasoning System Evaluator')
    parser.add_argument('--max', type=int, default=None,
                        help='Max examples per dataset (for quick testing)')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['ruletaker', 'counterbench', 'folio', 'all'],
                        help='Which dataset to evaluate')
    parser.add_argument('--strategy', type=str, default='both',
                        choices=['forward', 'backward', 'both'],
                        help='Inference strategy')
    parser.add_argument('--verbose', action='store_true',
                        help='Print inference traces')
    args = parser.parse_args()

    print("="*60)
    print("AI REASONING SYSTEM — EVALUATION")
    print("="*60)
    print(f"Strategy: {args.strategy} | Max/dataset: {args.max}")

    use_forward = args.strategy in ('forward', 'both')
    use_backward = args.strategy in ('backward', 'both')

    engine = ReasoningEngine(
        use_forward=use_forward,
        use_backward=use_backward,
        verbose=args.verbose
    )
    evaluator = Evaluator(engine, verbose=args.verbose)

    all_records: list[EvalRecord] = []
    all_metrics: list[dict] = []

    # Load and evaluate datasets
    t_load_start = time.perf_counter()

    if args.dataset in ('ruletaker', 'all'):
        splits = load_ruletaker(max_examples=args.max)
        for split_name, examples in splits.items():
            if not examples:
                continue
            tag = f"ruletaker/{split_name}"
            metrics = evaluator.evaluate(examples, tag=tag)
            all_metrics.append(metrics)
            all_records.extend(metrics['records'])

    if args.dataset in ('counterbench', 'all'):
        splits = load_counterbench(max_examples=args.max)
        for split_name, examples in splits.items():
            if not examples:
                continue
            tag = f"counterbench/{split_name}"
            metrics = evaluator.evaluate(examples, tag=tag)
            all_metrics.append(metrics)
            all_records.extend(metrics['records'])

    if args.dataset in ('folio', 'all'):
        splits = load_folio(max_examples=args.max)
        for split_name, examples in splits.items():
            if not examples:
                continue
            tag = f"folio/{split_name}"
            metrics = evaluator.evaluate(examples, tag=tag)
            all_metrics.append(metrics)
            all_records.extend(metrics['records'])

    print(f"\nTotal data load time: {time.perf_counter() - t_load_start:.1f}s")
    print(f"Total records evaluated: {len(all_records)}")

    # Generate submission.csv
    if all_records:
        csv_path = os.path.join('submission', 'submission.csv')
        os.makedirs('submission', exist_ok=True)
        generate_submission_csv(all_records, csv_path)

    # Overall accuracy summary
    print("\n=== OVERALL ACCURACY SUMMARY ===")
    for m in all_metrics:
        print(f"  {m['tag']}: {m['accuracy']:.2f}% "
              f"({m['n_correct']}/{m['n_labeled']}) "
              f"F1={m['f1']:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
