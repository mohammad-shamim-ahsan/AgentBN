# BN Agent

An agentic framework for diagnosing and repairing the conditional probability tables (CPTs) of a Bayesian Network (BN).

BN Agent starts from a flawed network, evaluates it against labeled reasoning scenarios, localizes failure-associated parameters, asks an LLM to propose constrained CPT patches, and uses deterministic inference to select and validate the best candidate. The LLM proposes; the harness measures, gates, retries, and records.

## Objective

The project investigates whether an agentic refinement loop can recover faulty BN parameters while preserving the network's structure and probabilistic validity.

Given:

- a Bayesian Network with one or more flawed CPTs;
- domain context describing the variables and their relationships;
- labeled train and test scenarios; and
- a target node whose prediction is evaluated,

the framework attempts to:

1. identify CPT parameters associated with repeated inference failures;
2. generate focused, valid CPT modifications;
3. improve scenario-level prediction accuracy through iterative refinement; and
4. compare the repaired BN with a ground-truth network using CPT-level distance metrics.

The current benchmarks are:

| Benchmark | Target node | Domain |
| --- | --- | --- |
| `alarm` | `HYPOVOLEMIA` | Clinical monitoring / ALARM network |
| `lung_cancer` | `either` | Asia lung-cancer network |
| `der` | `Root_Causes` | Distributed energy resource anomaly diagnosis |

## Pipeline

```mermaid
flowchart TD
    A["Load flawed BN, domain context, and train scenarios"] --> B["Run deterministic BN inference"]
    B --> C["Separate successes and failures"]
    C --> D["Trace activated CPT columns and parameters"]
    D --> E["Evaluator agent creates a CPT danger report"]
    E --> F["Refinement agent proposes candidate CPT patches"]
    F --> G["Harness validates and integrates each patch"]
    G --> H["Evaluate every candidate on the train set"]
    H --> I{"Accuracy improved?"}
    I -- "Yes" --> J["Store candidate as refinement memory"]
    I -- "No" --> K["Retry and retain the best candidate"]
    J --> L{"Target accuracy or iteration limit?"}
    K --> L
    L -- "Continue" --> B
    L -- "Stop" --> M["Select best BN for this restart"]
    M --> N{"More restarts?"}
    N -- "Yes" --> A
    N -- "No" --> O["Select best restart on held-out test set"]
    O --> P["Report accuracy and CPT distance metrics"]
```

### 1. Initialize

The orchestrator loads the selected benchmark, flawed BN, ground-truth BN, domain context, and train/test scenario files. It evaluates the flawed BN and computes an initial acceptance threshold:

```text
flawed_accuracy + INITIAL_IMPROVEMENT_RATIO × (1 - flawed_accuracy)
```

### 2. Diagnose

The evaluator builds the BN with `pgmpy`, runs inference, and records which CPT column and child state were activated for each relevant node. It aggregates those traces into failure and success weights and detects recurring cross-CPT activation patterns.

### 3. Localize

The evaluator agent receives the deterministic statistics, full BN, and domain context. It returns a structured danger report that ranks plausible CPT refinement targets and suggests adjustment directions. Deterministic evidence narrows the search space before the LLM is asked to reason.

### 4. Generate and select

The refinement agent receives the best accepted BN, its analysis record, domain context, target node, and inference thresholds. It proposes one or more CPT-only patches. The harness:

- validates the structured response;
- integrates each patch into a copy of the baseline BN;
- evaluates every candidate with deterministic inference; and
- selects the candidate with the fewest failures, breaking ties by accuracy.

### 5. Reflect and retry

Accepted BNs and their analyses become episodic memory for later iterations. If a candidate does not improve accuracy, it is removed from proposal memory and regenerated up to the configured retry limit. Each restart raises the LLM temperature slightly to diversify the search.

### 6. Validate

The best BN from each restart is retained. After all restarts, the harness evaluates each restart winner on the held-out test set and selects the result with the fewest failures. It then compares repaired CPTs with the ground truth using:

- Kullback–Leibler divergence;
- root mean squared error (RMSE);
- Hellinger distance; and
- expected-versus-observed CPT change analysis.

## Agent architecture

### Evaluator agent

Implemented in `agents/bn_evaluator.py`.

Its role is diagnostic, not generative. It combines deterministic activation traces with domain reasoning to produce a focused CPT danger report. This separation prevents the generator from changing arbitrary parts of the network without evidence.

### Refinement agent

Implemented in `agents/bn_generator_reflexion.py`.

It selects the best analyzed BN as its baseline, consumes the evaluator's diagnosis, and proposes localized CPT patches. It does not choose the winning candidate: candidate selection is delegated to the deterministic evaluation harness.

### Orchestrator

Implemented in `orchestration_pipeline.py`.

The orchestrator owns experiment state, initial acceptance, iterative refinement, retry policy, restarts, train/test separation, and final validation. It is the control plane that turns the two LLM roles into a reproducible workflow.

## Context engineering

Context is deliberately layered instead of sending the entire experiment history to every model call.

| Context layer | Source | Purpose |
| --- | --- | --- |
| Domain memory | `prompts/<benchmark>/context_agent.txt` | Defines variable semantics and domain relationships |
| Structural state | Current BN JSON | Grounds reasoning in graph structure and CPT values |
| Diagnostic state | `workspace/<benchmark>/bn_analysis.json` | Carries accuracy and the latest danger report |
| Parameter evidence | Activation trace and failure statistics | Links failures to specific CPT columns and states |
| Episodic memory | `last_proposed_bn.jsonl` | Preserves accepted proposals across iterations |
| Restart memory | `restart_final_bns.jsonl`, `restart_bn_analysis.jsonl` | Separates independent search trajectories |
| Runtime constraints | Target node, confidence, margin, thresholds | Keeps generation aligned with the experiment contract |

This design gives the LLM the smallest useful evidence bundle while keeping numerical inference, candidate scoring, and experiment state outside the model.

## Prompt engineering

The framework uses two principal prompt contracts:

- `prompts/ref_prompt.txt` instructs the refinement agent to return candidate CPT patches.
- The danger-report prompt in `agents/bn_evaluator.py` asks the evaluator to distinguish correlation from plausible independent contribution and return ranked refinement targets.

Prompt behavior is constrained through:

- explicit roles and task boundaries;
- domain context injection;
- definitions for every diagnostic statistic;
- stepwise reasoning criteria;
- strict JSON schemas;
- bounded output fields and enumerated labels;
- instructions to modify CPTs rather than the full network; and
- a separate temperature-zero formatting-repair prompt that may repair syntax but may not alter content.

Benchmark-specific context can be edited independently of the generic refinement prompt, allowing the same agent workflow to operate across domains.

## Harness engineering

The harness is responsible for correctness and control around probabilistic LLM output:

- **Deterministic scoring:** `pgmpy` inference, not model judgment, determines success.
- **Candidate tournaments:** every proposed patch is evaluated under the same scenarios.
- **Acceptance gates:** initial improvement and target-accuracy thresholds prevent unconditional adoption.
- **Bounded retries:** generation, no-improvement, JSON-format, and repair retries have explicit limits.
- **Schema validation:** malformed LLM responses are rejected before integration.
- **Content-preserving repair:** a separate call repairs JSON formatting at temperature `0.0`.
- **State isolation:** benchmark-specific workspaces prevent cross-experiment contamination.
- **Restart diversity:** independent runs use gradually higher temperatures.
- **Held-out selection:** restart winners are compared on the test set.
- **Artifact logging:** traces, diagnoses, proposals, restart winners, and CPT comparisons remain inspectable.

The central design principle is:

> Keep semantic hypothesis generation inside the agent; keep validation, state transitions, and acceptance decisions inside deterministic code.

## Repository structure

```text
BN_Agent/
├── agents/
│   ├── bn_evaluator.py              # Failure localization agent
│   └── bn_generator_reflexion.py    # CPT refinement agent
├── config/
│   └── settings.py                  # Paths, benchmarks, and hyperparameters
├── datasets/
│   ├── alarm/
│   ├── der/
│   └── lung_cancer/                 # BN files and train/test scenarios
├── evaluation/
│   ├── automatic_bn_reasoning_old.py
│   ├── bn_validator.py              # Accuracy and CPT-distance validation
│   └── recall_style_heuristic_validator.txt
├── prompts/
│   ├── alarm/
│   ├── der/
│   ├── lung_cancer/                 # Benchmark-specific context
│   ├── ref_prompt.txt               # Refinement prompt contract
│   └── scenario_gen_prompt.txt
├── utils/                            # BN I/O, inference, JSON, and LLM helpers
├── workspace/                        # Generated experiment state
├── logs/                             # Captured run logs
├── orchestration_pipeline.py         # Main entry point
└── run.sh                            # Convenience runner
```

`legacy/` contains earlier pipeline implementations and is not used by the current orchestrator.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- Bash, if using `run.sh`

Python packages used by the current codebase:

```text
openai
langchain-core
pgmpy
pandas
numpy
scikit-learn
```

## Installation

```bash
git clone https://github.com/mohammad-shamim-ahsan/BN_Agent.git
cd BN_Agent

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install openai langchain-core pgmpy pandas numpy scikit-learn
```

Set the API key in your environment:

```bash
export OPENAI_API_KEY="your-api-key"
```

The model is configured in `utils/llm.py`. Ensure that your OpenAI project has access to the configured model before running an experiment.

## Usage

Run the default `alarm` benchmark:

```bash
./run.sh
```

Run a specific benchmark:

```bash
./run.sh alarm
./run.sh lung_cancer
./run.sh der
```

Run the orchestrator directly:

```bash
python3 orchestration_pipeline.py --benchmark alarm
```

Optionally constrain the training sample to a success-to-failure ratio:

```bash
python3 orchestration_pipeline.py --benchmark der --SFR 5
```

`--SFR 5`, for example, constructs a temporary training sample with five successful scenarios for every failure scenario.

## Dataset contract

Each benchmark directory is expected to provide:

```text
datasets/<benchmark>/
├── BN_gt.json
├── flawed_BN_0.json
├── combined_train_scenarios.csv
└── combined_test_scenarios.csv
```

Scenario CSVs must contain:

- `Scenario #`: unique scenario identifier;
- `Ground Truth`: expected target-node state; and
- one column for each observed evidence variable.

The target node, evidence nodes, expected changed CPTs, and validation scope must be defined for the benchmark in `config/settings.py`. A matching domain context file must exist at `prompts/<benchmark>/context_agent.txt`.

## Configuration

Key controls live in `config/settings.py`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `MAX_RESTARTS` | `3` | Independent refinement trajectories |
| `BASE_TEMPERATURE` | `0.2` | Initial LLM sampling temperature |
| `MAX_ITER` | `3` | Refinement iterations per restart |
| `MAX_INITIAL_RETRIES` | `3` | Attempts to clear the initial acceptance threshold |
| `INITIAL_IMPROVEMENT_RATIO` | `0.30` | Required fraction of remaining accuracy gap |
| `MAX_NO_IMPROVEMENT_RETRIES` | `3` | Regeneration attempts after a non-improving proposal |
| `MIN_CONFIDENCE` | `0.50` | Confidence constraint passed to refinement |
| `MIN_MARGIN` | `0.20` | Prediction-margin constraint passed to refinement |
| `TARGET_ACCURACY` | `0.98` | Early-stop accuracy |
| `MAX_FORMAT_RETRIES` | `3` | Structured-output generation attempts |
| `MAX_REPAIR_RETRIES` | `2` | JSON repair attempts |

## Outputs

Each run writes benchmark-specific artifacts under `workspace/<benchmark>/`:

| Artifact | Description |
| --- | --- |
| `activation_trace.csv` | Activated CPT columns and selected states per scenario |
| `failure_parameter_statistics.json` | Failure/success weights and recurring activation patterns |
| `dangerous_cpt_report.json` | Evaluator agent's ranked refinement targets |
| `bn_analysis.json` | Per-BN accuracy and diagnostic memory |
| `last_proposed_bn.jsonl` | Candidate BN history |
| `restart_final_bns.jsonl` | Best BN retained from each restart |
| `restart_bn_analysis.jsonl` | Diagnostic history grouped by restart |
| `cpt_comparison_analysis.json` | Expected and observed CPT-change comparison |

Console output is captured under `logs/<benchmark>/` when using `run.sh`.

> **Important:** the pipeline clears several workspace artifacts at the beginning of an experiment and at each restart. Copy results you want to preserve before starting another run for the same benchmark.

## Reproducibility and limitations

- Dataset sampling for `--SFR` uses a fixed random seed (`42`).
- LLM generation remains stochastic; restart temperature increases by `0.1` per restart.
- Generated JSON is schema-checked, but domain correctness still depends on the supplied context and evaluation scenarios.
- Final restart selection uses the held-out test set. For strict research evaluation, consider selecting with a validation split and reserving the test set for one final, non-selective report.
- The repository currently declares dependencies in this README rather than a lock file; pin versions before running controlled experiments.
- API calls may incur cost and send the prompt context, BN data, and derived statistics to the configured model provider. Review data sensitivity before use.

## Extending the framework

To add a benchmark:

1. Create `datasets/<benchmark>/` using the dataset contract above.
2. Add `prompts/<benchmark>/context_agent.txt`.
3. Add the benchmark name to the CLI choices in `orchestration_pipeline.py`.
4. Define its target, evidence, expected-change, and validation nodes in `config/settings.py`.
5. Run a small configuration first and inspect the generated activation trace and danger report.

To use another model, update the model name and request parameters in `utils/llm.py`. Keep the structured-output validators and deterministic candidate evaluation in place when changing the agent layer.

## License

This project is licensed under the [MIT License](LICENSE).
