================================================================================
AI ASSIGNMENT -- CLASSICAL AI REASONING SYSTEM
Course: Artificial Intelligence (AI 2026)
Student Roll Number: M26CSA013
Deliverable: Submission Package & System Documentation
================================================================================

1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements an end-to-end, general-purpose classical AI reasoning
system that accepts natural-language knowledge bases (facts, rules, and causal relations)
and queries, translates them into an internal first-order / Horn-clause / causal
representation, and evaluates logical entailment using custom classical automated
reasoning engines (agenda-based forward chaining, goal-directed SLD backward chaining,
Robinson's unification, and Pearl's SCM counterfactual engine).

Strictly adhering to academic integrity guidelines, no generative LLM, ChatGPT API,
or black-box classifier is used to decide entailment. The inference is executed
entirely by the deterministic classical logic engines under the Closed World
Assumption (CWA) and Negation-as-Failure (NAF).


2. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
1/
|-- Assignment_AI_2026.pdf       # Course assignment specification
|-- run_evaluation.py            # CLI entrypoint for running evaluation
|-- requirements.txt             # Required Python packages
|-- src/                         # Core modular logic engine
|   |-- representation.py        # Literal, Clause, KnowledgeBase, Example data structures
|   |-- unification.py           # Robinson's unification with occurs-check & variable renaming
|   |-- forward_chaining.py      # Agenda-based forward chaining with indexing & cycle detection
|   |-- backward_chaining.py     # Goal-directed SLD backward chaining with depth limits
|   |-- counterfactual_engine.py # Pearl's SCM counterfactual abduction-intervention-deduction
|   |-- parser.py                # Regex-based natural language parser for facts and rules
|   |-- folio_fol_parser.py      # Formal First-Order Logic (FOL) parser for FOLIO annotations
|   |-- reasoning_engine.py      # Orchestrator combining backward, forward & causal chaining
|   |-- evaluation.py            # Evaluation metrics (Accuracy, Precision, Recall, F1, CM, Timing)
|   |-- dataset_loader.py        # Unified Hugging Face dataset loading & schema normalisation
|-- tests/                       # Automated test suite
|   |-- test_reasoning_engine.py # 23 comprehensive unit tests covering all edge cases
|-- submission/                  # Submission deliverables
|   |-- reasoning_agent.ipynb    # Fully self-contained, 20-section Jupyter Notebook
|   |-- report.pdf               # Comprehensive technical report in PDF format (Roll No: M26CSA013)
|   |-- submission.csv           # Predictions CSV with 'id' and 'predicted_label'
|   |-- README.txt               # This file


3. PROGRAMMING ENVIRONMENT & DEPENDENCIES
--------------------------------------------------------------------------------
- Python Version: Python 3.13 (or Python 3.9+)
- Student Roll Number: M26CSA013
- Dependencies:
    pip install datasets huggingface_hub fpdf2 requests pandas numpy nbformat

Note: All evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
are implemented in pure Python to eliminate platform-specific binary dependencies.


4. DATASET SOURCES & ACCESS INSTRUCTIONS
--------------------------------------------------------------------------------
- RuleTaker:
    Identifier: "tasksource/ruletaker"
    Splits: train (480,152), test (151,911), dev (75,872)
    Source: Clark et al., "Transformers as Soft Reasoners over Language"

- FOLIO:
    Identifier: "tasksource/folio" (mirror of "yale-nlp/folio")
    Splits: train (1,001), validation (203)
    Source: Han et al., "FOLIO: Natural Language Reasoning with First-Order Logic"
    Note: Three-way labels (True, False, Uncertain). Uncertain instances are excluded
    from binary accuracy computation and predicted as False under CWA.

- CounterBench:
    Identifier: "CounterBench/CounterBench"
    Source: CounterBench Deterministic Counterfactual 1.2K Dataset
    Access Method: Individual JSON file downloads via huggingface_hub:
      - "data_balanced_alpha_V1.json" (1,000 counterfactual questions across basic,
         conditional, joint, and nested types; balanced 500 yes / 500 no)
      - "data_balanced_backdoor_V2.json" (200 backdoor counterfactual questions)
    Model Definitions: "meta_model_alpha_V1.json" and "meta_model_backdoor_V2.json"
    maintained separately from evaluation instances.


5. INFERENCE ALGORITHMS & LOGICAL SEMANTICS
--------------------------------------------------------------------------------
- Logical Representation:
    - Literals: Predicate functors with constant or variable arguments, positive/negated.
    - Clauses: Horn clauses of the form Head :- Body_1, ..., Body_k.
    - KnowledgeBase: Hash-indexed by (predicate, arity) for O(1) candidate lookup.

- Unification:
    - Robinson's Most General Unifier (MGU) algorithm.
    - Handles variable-to-constant, variable-to-variable, and repeated variables.
    - Standardises variables apart via unique integer suffix renaming.

- Backward Chaining:
    - Goal-directed SLD resolution.
    - Evaluates subgoals recursively with depth bounding and cycle detection.

- Forward Chaining:
    - Agenda/worklist-based fixpoint derivation.
    - Iteratively fires rules when all antecedent premises unify with known facts.
    - Closed-set fact memoisation avoids redundant derivations and loops.

- Counterfactual Causal Reasoning (Pearl SCM):
    - Abduction: Condition on observed factual background states.
    - Graph Surgery / Intervention: Fixes intervened variables via do(X = x) and severs
      incoming causal influences.
    - Forward Constraint Deduction: Iteratively evaluates causal equations over the
      mutilated causal graph until fixpoint to decide the target query variable.

- Negation & CWA:
    - Closed World Assumption: If a ground atom cannot be proved, it is deemed False.
    - Negation-as-Failure: ¬P succeeds if P cannot be proved. Non-entailment is distinguished
      from proof of negative facts.


6. HOW TO RUN TESTS & REPRODUCE RESULTS
--------------------------------------------------------------------------------
A. Execute the Automated Unit Test Suite (23 tests):
    python -m tests.test_reasoning_engine

B. Run CLI Evaluation across all benchmarks:
    python run_evaluation.py --max 100 --dataset all

C. Regenerate the Technical Report PDF:
    python generate_report_pdf.py

D. Execute the Self-Contained Notebook:
    python execute_notebook.py
    (or open in Jupyter: jupyter notebook submission/reasoning_agent.ipynb)


7. EMPIRICAL RESULTS SUMMARY
--------------------------------------------------------------------------------
- Automated Unit Tests:
    23 passed, 0 failed out of 23 test cases (100% pass rate).

- RuleTaker (dev-200 sample):
    - Evaluated examples: 200
    - Accuracy: 58.50% (117 / 200)
    - Precision: 0.6019 | Recall: 0.5962 | F1 Score: 0.5990
    - Parse failure rate: 0.0%
    - Confusion Matrix: [[TN=55, FP=41], [FN=42, TP=62]]
    - Average total time: 0.345 ms / query

- CounterBench (alpha - 1,000 full dataset):
    - Evaluated examples: 1,000
    - Accuracy: 89.00% (890 / 1,000)
    - Precision: 0.8920 | Recall: 0.8880 | F1 Score: 0.8900
    - Parse failure rate: 0.0%
    - Average total time: 0.084 ms / query

- CounterBench (backdoor - 200 full dataset):
    - Evaluated examples: 200
    - Accuracy: 62.50% (125 / 200)
    - Precision: 0.5820 | Recall: 0.7470 | F1 Score: 0.6540
    - Parse failure rate: 0.0%
    - Average total time: 0.055 ms / query

- FOLIO (validation-all, 203 examples):
    - Total examples: 203 (134 binary labeled, 69 uncertain)
    - Accuracy: 50.75% (68 / 134)
    - Precision: 0.6000 | Recall: 0.2500 | F1 Score: 0.3529
    - Parse failure rate: 19.2%
    - Confusion Matrix: [[TN=50, FP=12], [FN=54, TP=18]]
    - Average total time: 0.249 ms / query


8. KNOWN LIMITATIONS & UNSUPPORTED CONSTRUCTS
--------------------------------------------------------------------------------
1. Stochastic Backdoor Conditioning: CounterBench backdoor instances contain Bernoulli
   distributions (Bern(p)). While structural causal paths are resolved deterministically,
   probabilistic Bayesian conditioning is beyond pure first-order / Horn logic.
2. First-Order Disjunction in Rule Heads: Full FOL allows disjunctive conclusions (A ∨ B),
   which requires full first-order resolution rather than Horn-clause chaining.
3. Open-World Reasoning: In benchmarks expecting non-CWA semantics, unprovable propositions
   are classified as False rather than "Unknown".
================================================================================