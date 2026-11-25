import pandas as pd
import numpy as np
import random

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pgmpy_tool import *
from utils.bn_io import *
from config.settings import *


# ============================================================
# Step 1: Read BN (JSON-like format: GeNIe XML equivalent)
# ============================================================
import json

def load_bn(filename):
    with open(filename, "r") as f:
        return json.load(f)


# ============================================================
# Step 4: Batch Evaluation
# ============================================================

def call_bn_inference(model, df, target_node=TARGET_NODE):

    results = []

    for _, row in df.iterrows():

        evidence = {}

        for col in df.columns:

            if col in ["Scenario #", "Ground Truth"]:
                continue

            val = row[col]

            if pd.isna(val):
                continue

            evidence[col] = str(val).strip()

        # print(f"\nRunning inference for Scenario: {row['Scenario #']}")
        # print("Evidence:", evidence)

        result = run_inference(
            model,
            query=target_node,
            evidence=evidence
        )

        pred_idx = np.argmax(result.values)
        pred_state = result.state_names[target_node][pred_idx]
        confidence = result.values[pred_idx]

        probs = sorted(result.values, reverse=True)
        max_prob = probs[0]
        second_prob = probs[1]
        margin = max_prob - second_prob

        posterior_probs = {
            state: float(prob)
            for state, prob in zip(
                result.state_names[target_node],
                result.values
            )
        }

        is_success = (
            pred_state == row["Ground Truth"]
            and confidence >= MIN_CONFIDENCE
            and margin >= MIN_MARGIN
        )

        results.append({
            "Scenario": row["Scenario #"],
            "Evidence": evidence,
            "Prediction": pred_state,
            "Confidence": confidence,
            "Ground Truth": row["Ground Truth"],
            "Posterior": posterior_probs,
            "Success": is_success
        })

    return results


# ============================================================
# Step 5: Run Evaluation
# ============================================================

def run_evaluation(bn_json, dataset_file, target_node=TARGET_NODE):

    model = build_model(bn_json)

    if isinstance(dataset_file, pd.DataFrame):
        df = dataset_file.copy()
    else:
        df = pd.read_csv(
            dataset_file,
            dtype=str,
            keep_default_na=False
        )

        df["Scenario #"] = df["Scenario #"].astype(int)
        # df["Ground Truth"] = df["Ground Truth"].str.upper()
        
    df.columns = df.columns.str.strip()

    results = call_bn_inference(model, df, target_node=target_node)

    results_df = pd.DataFrame(results)

    successes = results_df[
        results_df["Success"] == True
    ]

    failures = results_df[
        results_df["Success"] == False
    ]

    accuracy = len(successes) / len(results_df)

    # print("\n===================================================")
    # print("RESULTS")
    # print("===================================================")

    # print(results_df)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    # print("\nFailures:")
    # if failures.empty:
    #     print("No failures!")
    # else:
    #     print(failures)

    return failures, successes, accuracy, results

# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_cross_targets(bn_json, target_nodes, dataset_dir, suffix="", label=""):
    """Evaluate each secondary target and report macro/weighted accuracy."""
    total_successes = 0
    total_scenarios = 0
    target_accuracies = []

    for target_index, target in enumerate(target_nodes, start=2):
        cross_target_csv = (
            dataset_dir / f"combined_test_scenarios_t{target_index}{suffix}.csv"
        )
        failures, successes, accuracy, results = run_evaluation(
            bn_json,
            cross_target_csv,
            target_node=target,
        )
        print(
            f"{label}Cross-target T{target_index} ({target}): "
            f"{accuracy * 100:.2f}%"
        )

        if len(results) > 0:
            target_accuracies.append(accuracy)
            total_successes += len(successes)
            total_scenarios += len(results)

    macro_accuracy = (
        sum(target_accuracies) / len(target_accuracies)
        if target_accuracies else 0.0
    )
    weighted_accuracy = (
        total_successes / total_scenarios
        if total_scenarios > 0 else 0.0
    )

    print(f"{label}Macro cross-target accuracy: {macro_accuracy * 100:.2f}%")
    print(f"{label}Weighted cross-target accuracy: {weighted_accuracy * 100:.2f}%")


def evaluate_primary_b(bn_json, primary_csv_b, label):
    """Evaluate Experiment B primary target, including scenario counts."""
    failures, successes, accuracy, results = run_evaluation(
        bn_json,
        primary_csv_b,
        target_node=TARGET_NODE,
    )
    print(f"{label} primary-target accuracy (Experiment B): {accuracy * 100:.2f}%")
    print(f"Total scenarios: {len(results)}")
    print(f"Successful scenarios: {len(successes)}")
    print(f"Failed scenarios: {len(failures)}")


def evaluate_training_subsets(bn_json, train_subsets, learned_subset_id=None):
    """Evaluate all four subsets; distinguish own/cross training for learned BNs."""
    for eval_subset_id, subset_df in enumerate(train_subsets, start=1):
        failures, successes, accuracy, results = run_evaluation(
            bn_json,
            subset_df,
        )
        if learned_subset_id is None:
            label = "Training"
        elif eval_subset_id == learned_subset_id:
            label = "Own training"
        else:
            label = "Cross-training"
        print(f"{label} subset {eval_subset_id}: {accuracy * 100:.2f}%")


def evaluate_held_out(bn_json, test_csv):
    failures, successes, accuracy, results = run_evaluation(bn_json, test_csv)
    print(f"Held-out test: {accuracy * 100:.2f}%")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"BENCHMARK = {BENCHMARK!r}")

    DATASET_DIR = Path("datasets") / BENCHMARK
    TRAIN_CSV = DATASET_DIR / "combined_train_scenarios.csv"
    TEST_CSV = DATASET_DIR / "combined_test_scenarios.csv"
    PRIMARY_CSV_B = DATASET_DIR / "combined_test_scenarios_exp_b.csv"

    GT_BN_FILE = DATASET_DIR / "BN_gt.json"
    FLAWED_BN_FILE = DATASET_DIR / "flawed_BN_0.json"

    # ------------------------------------------------------------
    # Prepare training subsets (must exactly match batch_em.py)
    # ------------------------------------------------------------
    full_train_data = pd.read_csv(
        TRAIN_CSV,
        dtype=str,
        keep_default_na=False,
    )
    full_train_data["Scenario #"] = full_train_data["Scenario #"].astype(int)
    full_train_data = full_train_data.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)
    train_subsets = np.array_split(full_train_data, 4)

    # ------------------------------------------------------------
    # Experiment B: reconstruct evidence and secondary targets
    # ------------------------------------------------------------
    RANDOM_SEED_B = 42
    NUM_EVIDENCE_B = 7
    ground_truth_bn = load_bn(GROUND_TRUTH_BN_FILE)
    ALL_NODES = {node["name"] for node in ground_truth_bn["nodes"]}
    CANDIDATE_EVIDENCE_NODES = sorted(ALL_NODES - {TARGET_NODE})

    rng = random.Random(RANDOM_SEED_B)
    EVIDENCE_NODES_B = sorted(
        rng.sample(CANDIDATE_EVIDENCE_NODES, NUM_EVIDENCE_B)
    )
    CROSS_TARGET_NODES_B = sorted(
        ALL_NODES - set(EVIDENCE_NODES_B) - {TARGET_NODE}
    )
    print(f"Experiment B evidence nodes: {EVIDENCE_NODES_B}")
    print(f"Experiment B secondary targets: {CROSS_TARGET_NODES_B}")

    # ------------------------------------------------------------
    # Load AgentBN models once, indexed by learned subset
    # ------------------------------------------------------------
    agentbn_by_subset = {}
    with open(AGENTBN_SUBSET_BN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "subset_id" in record and "bn" in record:
                # Keep first matching record, as in the original AgentBN loop.
                agentbn_by_subset.setdefault(record["subset_id"], record["bn"])

    
    # ------------------------------------------------------------
    # Load sensitivity analysis models, indexed by learned subset
    # ------------------------------------------------------------
    SENSITIVITY_BN_FILE = (
        Path("baselines/results") / BENCHMARK / "sensitivity_analysis_bn.jsonl"
    )

    sensitivity_by_subset = {}
    with open(SENSITIVITY_BN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            subset_id = record["metadata"]["subset_id"]
            sensitivity_by_subset[subset_id] = record["bn"]

    print("Loaded sensitivity subsets:", sorted(sensitivity_by_subset))


    # # ============================================================
    # # 1. BATCH EM: each independently learned BN
    # # ============================================================
    # for learned_subset_id in range(2, 5):
    #     print("\n" + "=" * 60)
    #     print(f"EM BN LEARNED FROM SUBSET {learned_subset_id}")
    #     print("=" * 60)

    #     bn_json = get_bn(
    #         BATCH_EM_BN_FILE,
    #         bn_number=100,
    #         oracle_size=0,
    #         subset_id=learned_subset_id,
    #     )

    #     evaluate_training_subsets(bn_json, train_subsets, learned_subset_id)
    #     evaluate_held_out(bn_json, TEST_CSV)

    #     print("\nExperiment A: secondary targets")
    #     evaluate_cross_targets(bn_json, CROSS_TARGET_NODES, DATASET_DIR)

    #     print("\nExperiment B: secondary targets")
    #     evaluate_cross_targets(
    #         bn_json, CROSS_TARGET_NODES_B, DATASET_DIR, suffix="_exp_b"
    #     )

    #     print("\nExperiment B: primary target")
    #     evaluate_primary_b(bn_json, PRIMARY_CSV_B, "Batch EM")

    # # ============================================================
    # # 2. ORIGINAL DEPLOYED BN
    # # ============================================================
    # print("\n" + "=" * 60)
    # print("ORIGINAL DEPLOYED BN")
    # print("=" * 60)

    # flawed_bn_json = load_bn(FLAWED_BN_FILE)
    # evaluate_training_subsets(flawed_bn_json, train_subsets)
    # evaluate_held_out(flawed_bn_json, TEST_CSV)

    # print("\nExperiment A: secondary targets")
    # evaluate_cross_targets(
    #     flawed_bn_json, CROSS_TARGET_NODES, DATASET_DIR, label="Original BN "
    # )

    # print("\nExperiment B: secondary targets")
    # evaluate_cross_targets(
    #     flawed_bn_json,
    #     CROSS_TARGET_NODES_B,
    #     DATASET_DIR,
    #     suffix="_exp_b",
    #     label="Original BN ",
    # )

    # print("\nExperiment B: primary target")
    # evaluate_primary_b(flawed_bn_json, PRIMARY_CSV_B, "Original BN")

    # ============================================================
    # 3. AGENTBN: each independently learned BN
    # ============================================================
    for learned_subset_id in range(2, 5):
        print("\n" + "=" * 60)
        print(f"AGENTBN LEARNED FROM SUBSET {learned_subset_id}")
        print("=" * 60)

        bn_json = agentbn_by_subset.get(learned_subset_id)
        if bn_json is None:
            raise ValueError(
                f"No AgentBN found for subset_id={learned_subset_id}"
            )

        evaluate_training_subsets(bn_json, train_subsets, learned_subset_id)
        evaluate_held_out(bn_json, TEST_CSV)

        print("\nExperiment A: secondary targets")
        evaluate_cross_targets(bn_json, CROSS_TARGET_NODES, DATASET_DIR)

        print("\nExperiment B: secondary targets")
        evaluate_cross_targets(
            bn_json, CROSS_TARGET_NODES_B, DATASET_DIR, suffix="_exp_b"
        )

        print("\nExperiment B: primary target")
        evaluate_primary_b(bn_json, PRIMARY_CSV_B, "AgentBN")

    
    # # ============================================================
    # # 4. SENSITIVITY ANALYSIS: each independently refined BN
    # # ============================================================
    # for learned_subset_id in range(2, 5):
    #     print("\n" + "=" * 60)
    #     print(f"SENSITIVITY ANALYSIS BN LEARNED FROM SUBSET {learned_subset_id}")
    #     print("=" * 60)

    #     bn_json = sensitivity_by_subset.get(learned_subset_id)

    #     if bn_json is None:
    #         raise ValueError(
    #             f"No sensitivity analysis BN found for "
    #             f"subset_id={learned_subset_id}"
    #         )

    #     evaluate_training_subsets(bn_json, train_subsets, learned_subset_id)
    #     evaluate_held_out(bn_json, TEST_CSV)

    #     print("\nExperiment A: secondary targets")
    #     evaluate_cross_targets(
    #         bn_json, CROSS_TARGET_NODES, DATASET_DIR
    #     )

    #     print("\nExperiment B: secondary targets")
    #     evaluate_cross_targets(
    #         bn_json, CROSS_TARGET_NODES_B, DATASET_DIR, suffix="_exp_b"
    #     )

    #     print("\nExperiment B: primary target")
    #     evaluate_primary_b(bn_json, PRIMARY_CSV_B, "Sensitivity Analysis")
