"""
dataset_loader.py
=================
Unified dataset loading layer for RuleTaker, CounterBench, and FOLIO.

Each loader:
1. Loads available splits from Hugging Face (or local cache).
2. Normalises schema to the common Example dataclass.
3. Maps labels to bool (True/False).
4. Validates counts, missing fields, duplicate IDs, and label distributions.

Hugging Face Repositories:
- RuleTaker:    "tasksource/ruletaker"
- FOLIO:        "tasksource/folio" (or "yale-nlp/folio")
- CounterBench: "CounterBench/CounterBench" (individual JSON downloads)
"""
from __future__ import annotations
import os
import json
import logging
import time
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("HuggingFace 'datasets' / 'huggingface_hub' not installed.")

from src.representation import Example


# ---------------------------------------------------------------------------
# Label normalisation helpers
# ---------------------------------------------------------------------------

def _to_bool(raw_label) -> Optional[bool]:
    """Convert various label representations to Python bool.

    Returns None if the label is unrecognisable or uncertain.
    """
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, (int, float)):
        if raw_label == 1:
            return True
        if raw_label == 0:
            return False
    if isinstance(raw_label, str):
        v = raw_label.strip().lower()
        if v in ('true', '1', 'yes', 'entailed', 'proved', 'entailment'):
            return True
        if v in ('false', '0', 'no', 'not entailed', 'unproved', 'not entailment',
                 'non-entailment', 'contradiction'):
            return False
        if v in ('uncertain', 'unknown'):
            return None
    return None


def _label_distribution(examples: list[Example]) -> dict:
    """Return a dict with True/False/None counts."""
    counts = Counter(e.label for e in examples)
    return dict(counts)


def _report(name: str, split: str, examples: list[Example]) -> None:
    dist = _label_distribution(examples)
    logger.info(f"[{name}] {split}: {len(examples)} examples | labels: {dist}")
    print(f"  [{name}] {split}: {len(examples)} examples | label dist: {dist}")


# ---------------------------------------------------------------------------
# RuleTaker loader
# ---------------------------------------------------------------------------

RULETAKER_IDS = [
    "tasksource/ruletaker",
]


def load_ruletaker(max_examples: Optional[int] = None) -> dict[str, list[Example]]:
    """Load RuleTaker dataset from Hugging Face."""
    if not HF_AVAILABLE:
        logger.error("datasets library not available; cannot load RuleTaker.")
        return {}

    print("Loading RuleTaker ...")
    t0 = time.perf_counter()

    ds = None
    for ds_id in RULETAKER_IDS:
        try:
            ds = load_dataset(ds_id)
            print(f"  Loaded from: {ds_id}")
            break
        except Exception as e:
            logger.warning(f"  Could not load {ds_id}: {e}")

    if ds is None:
        print("  [X] RuleTaker could not be loaded from known identifiers.")
        return {}

    print(f"  Available splits: {list(ds.keys())}")
    result: dict[str, list[Example]] = {}

    for split_name, split_data in ds.items():
        examples = []
        seen_ids = set()
        for i, row in enumerate(split_data):
            if max_examples and i >= max_examples:
                break

            ex_id = str(row.get('id', row.get('question_id', row.get('idx', f"rt_{split_name}_{i}"))))
            if ex_id in seen_ids:
                ex_id = f"{ex_id}_{i}"
            seen_ids.add(ex_id)

            context = str(row.get('context', row.get('theory', row.get('passage', ''))))
            query = str(row.get('question', row.get('statement', row.get('query', ''))))
            raw_label = row.get('label', row.get('answer', row.get('target', None)))
            label = _to_bool(raw_label)

            if not context or not query:
                continue

            ex = Example(
                id=ex_id,
                context=context,
                query=query,
                label=label,
                dataset='ruletaker',
                split=split_name,
                raw=dict(row)
            )
            examples.append(ex)

        _report('ruletaker', split_name, examples)
        result[split_name] = examples

    print(f"  RuleTaker loaded in {time.perf_counter() - t0:.1f}s")
    return result


# ---------------------------------------------------------------------------
# CounterBench loader (Individual JSON File Loader)
# ---------------------------------------------------------------------------

COUNTERBENCH_REPO = "CounterBench/CounterBench"
COUNTERBENCH_FILES = {
    "alpha": "data_balanced_alpha_V1.json",
    "backdoor": "data_balanced_backdoor_V2.json"
}
COUNTERBENCH_META_FILES = {
    "meta_alpha": "meta_model_alpha_V1.json",
    "meta_backdoor": "meta_model_backdoor_V2.json"
}


def load_counterbench(max_examples: Optional[int] = None,
                      include_backdoor: bool = True) -> dict[str, list[Example]]:
    """Load CounterBench dataset by downloading and parsing specific JSON files.

    Handles individual JSON files from repository 'CounterBench/CounterBench':
      - data_balanced_alpha_V1.json (1,000 counterfactual questions)
      - data_balanced_backdoor_V2.json (200 backdoor questions)
    
    Validates:
      - JSON syntax and record count
      - Required fields: 'question_id', 'given_info', 'question', 'answer'
      - Unique example IDs with prefixing
      - Label conversion ('yes' -> True, 'no' -> False)
    """
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not available; cannot load CounterBench.")
        return {}

    print("Loading CounterBench from 'CounterBench/CounterBench' ...")
    t0 = time.perf_counter()
    result: dict[str, list[Example]] = {}

    target_files = {"alpha": COUNTERBENCH_FILES["alpha"]}
    if include_backdoor:
        target_files["backdoor"] = COUNTERBENCH_FILES["backdoor"]

    for split_key, filename in target_files.items():
        try:
            local_path = hf_hub_download(
                repo_id=COUNTERBENCH_REPO,
                filename=filename,
                repo_type="dataset"
            )
            print(f"  Downloaded {filename} -> {local_path}")
        except Exception as e:
            print(f"  [X] Failed to download {filename}: {e}")
            logger.error(f"Error downloading {filename} from {COUNTERBENCH_REPO}: {e}")
            continue

        try:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [X] JSON parse error in {filename}: {e}")
            continue

        if not isinstance(data, list):
            print(f"  [X] Unexpected format in {filename}: expected list, got {type(data)}")
            continue

        examples = []
        seen_ids = set()
        missing_fields_count = 0
        invalid_label_count = 0

        for idx, row in enumerate(data):
            if max_examples is not None and len(examples) >= max_examples:
                break

            if not isinstance(row, dict):
                continue

            raw_qid = str(row.get("question_id", idx))
            ex_id = f"cb_{split_key}_{raw_qid}"
            if ex_id in seen_ids:
                ex_id = f"{ex_id}_{idx}"
            seen_ids.add(ex_id)

            given_info = row.get("given_info", "")
            question = row.get("question", "")
            raw_answer = row.get("answer", "")

            if not given_info or not question:
                missing_fields_count += 1
                continue

            label = _to_bool(raw_answer)
            if label is None:
                invalid_label_count += 1

            ex = Example(
                id=ex_id,
                context=given_info,
                query=question,
                label=label,
                dataset="counterbench",
                split=split_key,
                raw=dict(row)
            )
            examples.append(ex)

        if missing_fields_count > 0:
            logger.warning(f"CounterBench {split_key}: skipped {missing_fields_count} rows with missing fields")
        if invalid_label_count > 0:
            logger.warning(f"CounterBench {split_key}: {invalid_label_count} rows with non-binary labels")

        _report("counterbench", split_key, examples)
        result[split_key] = examples

    print(f"  CounterBench loaded in {time.perf_counter() - t0:.1f}s")
    return result


def load_counterbench_metadata() -> dict[str, list[dict]]:
    """Download and return the meta-model definition files."""
    if not HF_AVAILABLE:
        return {}
    meta_data = {}
    for meta_key, filename in COUNTERBENCH_META_FILES.items():
        try:
            local_path = hf_hub_download(
                repo_id=COUNTERBENCH_REPO,
                filename=filename,
                repo_type="dataset"
            )
            with open(local_path, "r", encoding="utf-8") as f:
                meta_data[meta_key] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata {filename}: {e}")
    return meta_data


# ---------------------------------------------------------------------------
# FOLIO loader
# ---------------------------------------------------------------------------

FOLIO_IDS = [
    "tasksource/folio",
    "yale-nlp/folio",
]


def load_folio(max_examples: Optional[int] = None) -> dict[str, list[Example]]:
    """Load the FOLIO dataset from Hugging Face."""
    if not HF_AVAILABLE:
        return {}

    print("Loading FOLIO ...")
    t0 = time.perf_counter()

    ds = None
    for ds_id in FOLIO_IDS:
        try:
            ds = load_dataset(ds_id)
            print(f"  Loaded from: {ds_id}")
            break
        except Exception as e:
            logger.warning(f"  Could not load {ds_id}: {e}")

    if ds is None:
        print("  [X] FOLIO could not be loaded.")
        return {}

    print(f"  Available splits: {list(ds.keys())}")
    result: dict[str, list[Example]] = {}

    for split_name, split_data in ds.items():
        examples = []
        seen_ids = set()
        for i, row in enumerate(split_data):
            if max_examples and i >= max_examples:
                break

            raw_id = str(row.get('id', row.get('story_id', row.get('idx', f"fl_{split_name}_{i}"))))
            ex_id = f"fl_{split_name}_{raw_id}"
            if ex_id in seen_ids:
                ex_id = f"{ex_id}_{i}"
            seen_ids.add(ex_id)

            context_raw = row.get('premises', row.get('context', row.get('premise', '')))
            if isinstance(context_raw, list):
                context = ' '.join(context_raw)
            else:
                context = str(context_raw)

            query = str(row.get('conclusion', row.get('hypothesis', row.get('statement', ''))))
            raw_label = row.get('label', row.get('answer', None))
            label = _to_bool(raw_label)

            if not context or not query:
                continue

            ex = Example(
                id=ex_id,
                context=context,
                query=query,
                label=label,
                dataset='folio',
                split=split_name,
                raw=dict(row)
            )
            examples.append(ex)

        _report('folio', split_name, examples)
        result[split_name] = examples

    print(f"  FOLIO loaded in {time.perf_counter() - t0:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_datasets(max_per_dataset: Optional[int] = None) -> dict[str, dict[str, list[Example]]]:
    """Load all three datasets. Returns nested dict: dataset -> split -> examples."""
    all_data = {}
    all_data['ruletaker'] = load_ruletaker(max_examples=max_per_dataset)
    all_data['counterbench'] = load_counterbench(max_examples=max_per_dataset)
    all_data['folio'] = load_folio(max_examples=max_per_dataset)

    print("\n=== All Datasets Summary ===")
    for ds_name, splits in all_data.items():
        if not splits:
            print(f"  {ds_name}: NOT LOADED")
            continue
        total = sum(len(v) for v in splits.values())
        print(f"  {ds_name}: {total} total examples across {list(splits.keys())}")

    return all_data
