import pandas as pd
import numpy as np

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
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"BENCHMARK = {BENCHMARK!r}")

    DATASET_DIR = Path("datasets") / BENCHMARK

    TRAIN_CSV = DATASET_DIR / "combined_train_scenarios.csv"
    TEST_CSV = DATASET_DIR / "combined_test_scenarios.csv"
    
    GT_BN_FILE = DATASET_DIR / "BN_gt.json"
    FLAWED_BN_FILE = DATASET_DIR / "flawed_BN_0.json"
    

    # # ============================================================
    # # Evaluation
    # # ============================================================
    full_train_data = pd.read_csv(
        TRAIN_CSV,
        dtype=str,
        keep_default_na=False
    )

    full_train_data["Scenario #"] = full_train_data["Scenario #"].astype(int)

    # Must exactly match batch_em.py
    full_train_data = full_train_data.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    train_subsets = np.array_split(
        full_train_data,
        4,
    )

    # ============================================================
    # Experiment B: Reconstruct evidence and secondary targets
    # ============================================================
    import random

    RANDOM_SEED_B = 42
    NUM_EVIDENCE_B = 7
    ground_truth_bn = load_bn(GROUND_TRUTH_BN_FILE)

    ALL_NODES = {
        node["name"]
        for node in ground_truth_bn["nodes"]
    }

    CANDIDATE_EVIDENCE_NODES = sorted(
        ALL_NODES - {TARGET_NODE}
    )

    rng = random.Random(RANDOM_SEED_B)

    EVIDENCE_NODES_B = sorted(
        rng.sample(
            CANDIDATE_EVIDENCE_NODES,
            NUM_EVIDENCE_B
        )
    )

    CROSS_TARGET_NODES_B = sorted(
        ALL_NODES - set(EVIDENCE_NODES_B) - {TARGET_NODE}
    )

    print(f"Experiment B evidence nodes: {EVIDENCE_NODES_B}")
    print(f"Experiment B secondary targets: {CROSS_TARGET_NODES_B}")

    # # ============================================================
    # # Step X: Evaluate each independently learned Batch EM BN
    # # ============================================================
    # for learned_subset_id in range(1, 5):

    #     print("\n" + "=" * 60)
    #     print(
    #         f"EM BN LEARNED FROM SUBSET "
    #         f"{learned_subset_id}"
    #     )
    #     print("=" * 60)

    #     # Load the BN learned from this subset
    #     bn_json = get_bn(
    #         BATCH_EM_BN_FILE,
    #         bn_number=100,
    #         oracle_size=0,
    #         subset_id=learned_subset_id,
    #     )

    #     # --------------------------------------------------------
    #     # Evaluate on all training subsets
    #     # --------------------------------------------------------
    #     for eval_subset_id, subset_df in enumerate(
    #         train_subsets,
    #         start=1,
    #     ):

    #         failures, successes, accuracy, results = run_evaluation(
    #             bn_json,
    #             subset_df,
    #         )

    #         if eval_subset_id == learned_subset_id:
    #             label = "Own training"
    #         else:
    #             label = "Cross-training"

    #         print(
    #             f"{label} subset "
    #             f"{eval_subset_id}: "
    #             f"{accuracy * 100:.2f}%"
    #         )

    #     # --------------------------------------------------------
    #     # Evaluate on common held-out test set
    #     # --------------------------------------------------------
    #     failures, successes, accuracy, results = run_evaluation(
    #         bn_json,
    #         TEST_CSV,
    #     )

    #     print(
    #         f"Held-out test: "
    #         f"{accuracy * 100:.2f}%"
    #     )

    #     # --------------------------------------------------------
    #     # Evaluate individual, macro, and weighted cross-target accuracy
    #     # --------------------------------------------------------
    #     total_successes = 0
    #     total_scenarios = 0
    #     target_accuracies = []

    #     for target_index, target in enumerate(CROSS_TARGET_NODES, start=2):

    #         cross_target_csv = (
    #             DATASET_DIR / f"combined_test_scenarios_t{target_index}.csv"
    #         )

    #         failures, successes, accuracy, results = run_evaluation(
    #             bn_json,
    #             cross_target_csv,
    #             target_node=target,
    #         )

    #         print(
    #             f"Cross-target T{target_index} ({target}): "
    #             f"{accuracy * 100:.2f}%"
    #         )

    #         if len(results) > 0:
    #             target_accuracies.append(accuracy)
    #             total_successes += len(successes)
    #             total_scenarios += len(results)

    #     macro_accuracy = (
    #         sum(target_accuracies) / len(target_accuracies)
    #         if target_accuracies else 0.0
    #     )

    #     weighted_accuracy = (
    #         total_successes / total_scenarios
    #         if total_scenarios > 0 else 0.0
    #     )

    #     print(
    #         f"Macro cross-target accuracy: "
    #         f"{macro_accuracy * 100:.2f}%"
    #     )

    #     print(
    #         f"Weighted cross-target accuracy: "
    #         f"{weighted_accuracy * 100:.2f}%"
    #     )

    #     # --------------------------------------------------------
    #     # Evaluate individual, macro, and weighted cross-target accuracy
    #     # Experiment B
    #     # --------------------------------------------------------
    #     total_successes = 0
    #     total_scenarios = 0
    #     target_accuracies = []

    #     for target_index, target in enumerate(CROSS_TARGET_NODES_B, start=2):

    #         cross_target_csv = (
    #             DATASET_DIR / f"combined_test_scenarios_t{target_index}_exp_b.csv"
    #         )

    #         failures, successes, accuracy, results = run_evaluation(
    #             bn_json,
    #             cross_target_csv,
    #             target_node=target,
    #         )

    #         print(
    #             f"Cross-target T{target_index} ({target}): "
    #             f"{accuracy * 100:.2f}%"
    #         )

    #         if len(results) > 0:
    #             target_accuracies.append(accuracy)
    #             total_successes += len(successes)
    #             total_scenarios += len(results)

    #     macro_accuracy = (
    #         sum(target_accuracies) / len(target_accuracies)
    #         if target_accuracies else 0.0
    #     )

    #     weighted_accuracy = (
    #         total_successes / total_scenarios
    #         if total_scenarios > 0 else 0.0
    #     )

    #     print(
    #         f"Macro cross-target accuracy: "
    #         f"{macro_accuracy * 100:.2f}%"
    #     )

    #     print(
    #         f"Weighted cross-target accuracy: "
    #         f"{weighted_accuracy * 100:.2f}%"
    #     )


    # # ============================================================
    # # Step Y: Evaluate original flawed BN
    # # ============================================================
    # print("\n" + "=" * 60)
    # print("ORIGINAL FLAWED BN")
    # print("=" * 60)

    # flawed_bn_json = load_bn(FLAWED_BN_FILE)

    # # Evaluate on all four training/refinement subsets
    # for subset_id, subset_df in enumerate(
    #     train_subsets,
    #     start=1,
    # ):
    #     failures, successes, accuracy, results = run_evaluation(
    #         flawed_bn_json,
    #         subset_df,
    #     )

    #     print(
    #         f"Training subset {subset_id}: "
    #         f"{accuracy * 100:.2f}%"
    #     )

    # # Evaluate on common held-out test set
    # failures, successes, accuracy, results = run_evaluation(
    #     flawed_bn_json,
    #     TEST_CSV,
    # )

    # print(
    #     f"Held-out test: "
    #     f"{accuracy * 100:.2f}%"
    # )

    # # ============================================================
    # # Evaluate original deployed BN on cross-target test datasets
    # # ============================================================
    # total_successes = 0
    # total_scenarios = 0
    # target_accuracies = []

    # for target_index, target in enumerate(CROSS_TARGET_NODES, start=2):

    #     cross_target_csv = (
    #         DATASET_DIR / f"combined_test_scenarios_t{target_index}.csv"
    #     )

    #     failures, successes, accuracy, results = run_evaluation(
    #         flawed_bn_json,
    #         cross_target_csv,
    #         target_node=target,
    #     )

    #     print(
    #         f"Cross-target T{target_index} ({target}): "
    #         f"{accuracy * 100:.2f}%"
    #     )

    #     if len(results) > 0:
    #         target_accuracies.append(accuracy)
    #         total_successes += len(successes)
    #         total_scenarios += len(results)

    # macro_accuracy = (
    #     sum(target_accuracies) / len(target_accuracies)
    #     if target_accuracies else 0.0
    # )

    # weighted_accuracy = (
    #     total_successes / total_scenarios
    #     if total_scenarios > 0 else 0.0
    # )

    # print(
    #     f"Original BN macro cross-target accuracy: "
    #     f"{macro_accuracy * 100:.2f}%"
    # )

    # print(
    #     f"Original BN weighted cross-target accuracy: "
    #     f"{weighted_accuracy * 100:.2f}%"
    # )

    # # ============================================================
    # # Evaluate original deployed BN on cross-target test datasets
    # # Experiment B
    # # ============================================================
    # total_successes = 0
    # total_scenarios = 0
    # target_accuracies = []

    # for target_index, target in enumerate(CROSS_TARGET_NODES_B, start=2):

    #     cross_target_csv = (
    #         DATASET_DIR / f"combined_test_scenarios_t{target_index}_exp_b.csv"
    #     )

    #     failures, successes, accuracy, results = run_evaluation(
    #         flawed_bn_json,
    #         cross_target_csv,
    #         target_node=target,
    #     )

    #     print(
    #         f"Cross-target T{target_index} ({target}): "
    #         f"{accuracy * 100:.2f}%"
    #     )

    #     if len(results) > 0:
    #         target_accuracies.append(accuracy)
    #         total_successes += len(successes)
    #         total_scenarios += len(results)

    # macro_accuracy = (
    #     sum(target_accuracies) / len(target_accuracies)
    #     if target_accuracies else 0.0
    # )

    # weighted_accuracy = (
    #     total_successes / total_scenarios
    #     if total_scenarios > 0 else 0.0
    # )

    # print(
    #     f"Original BN macro cross-target accuracy: "
    #     f"{macro_accuracy * 100:.2f}%"
    # )

    # print(
    #     f"Original BN weighted cross-target accuracy: "
    #     f"{weighted_accuracy * 100:.2f}%"
    # )


    # # ============================================================
    # # Step Z: Evaluate each independently learned AgentBN
    # # ============================================================
    # for learned_subset_id in range(1, 5):

    #     print("\n" + "=" * 60)
    #     print(
    #         f"AGENTBN LEARNED FROM SUBSET "
    #         f"{learned_subset_id}"
    #     )
    #     print("=" * 60)

    #     # --------------------------------------------------------
    #     # Load AgentBN learned from this subset
    #     # --------------------------------------------------------
    #     bn_json = None

    #     with open(
    #         AGENTBN_SUBSET_BN_FILE,
    #         "r",
    #         encoding="utf-8"
    #     ) as f:

    #         for line in f:

    #             if not line.strip():
    #                 continue

    #             record = json.loads(line)

    #             if record.get("subset_id") == learned_subset_id:
    #                 bn_json = record["bn"]
    #                 break

    #     if bn_json is None:
    #         raise ValueError(
    #             f"No AgentBN found for subset_id={learned_subset_id}"
    #         )

    #     # --------------------------------------------------------
    #     # Evaluate on all training subsets
    #     # --------------------------------------------------------
    #     for eval_subset_id, subset_df in enumerate(
    #         train_subsets,
    #         start=1,
    #     ):

    #         failures, successes, accuracy, results = run_evaluation(
    #             bn_json,
    #             subset_df,
    #         )

    #         if eval_subset_id == learned_subset_id:
    #             label = "Own training"
    #         else:
    #             label = "Cross-training"

    #         print(
    #             f"{label} subset "
    #             f"{eval_subset_id}: "
    #             f"{accuracy * 100:.2f}%"
    #         )

    #     # --------------------------------------------------------
    #     # Evaluate on common held-out test set
    #     # --------------------------------------------------------
    #     failures, successes, accuracy, results = run_evaluation(
    #         bn_json,
    #         TEST_CSV,
    #     )

    #     print(
    #         f"Held-out test: "
    #         f"{accuracy * 100:.2f}%"
    #     )
        
    #     # --------------------------------------------------------
    #     # Evaluate individual, macro, and weighted cross-target accuracy
    #     # --------------------------------------------------------
    #     total_successes = 0
    #     total_scenarios = 0
    #     target_accuracies = []

    #     for target_index, target in enumerate(CROSS_TARGET_NODES, start=2):

    #         cross_target_csv = (
    #             DATASET_DIR / f"combined_test_scenarios_t{target_index}.csv"
    #         )

    #         failures, successes, accuracy, results = run_evaluation(
    #             bn_json,
    #             cross_target_csv,
    #             target_node=target,
    #         )

    #         print(
    #             f"Cross-target T{target_index} ({target}): "
    #             f"{accuracy * 100:.2f}%"
    #         )

    #         if len(results) > 0:
    #             target_accuracies.append(accuracy)
    #             total_successes += len(successes)
    #             total_scenarios += len(results)

    #     macro_accuracy = (
    #         sum(target_accuracies) / len(target_accuracies)
    #         if target_accuracies else 0.0
    #     )

    #     weighted_accuracy = (
    #         total_successes / total_scenarios
    #         if total_scenarios > 0 else 0.0
    #     )

    #     print(f"Macro cross-target accuracy: {macro_accuracy * 100:.2f}%")
    #     print(f"Weighted cross-target accuracy: {weighted_accuracy * 100:.2f}%")

    #     # --------------------------------------------------------
    #     # Evaluate individual, macro, and weighted cross-target accuracy
    #     # Experiment B
    #     # --------------------------------------------------------
    #     total_successes = 0
    #     total_scenarios = 0
    #     target_accuracies = []

    #     for target_index, target in enumerate(CROSS_TARGET_NODES_B, start=2):

    #         cross_target_csv = (
    #             DATASET_DIR / f"combined_test_scenarios_t{target_index}_exp_b.csv"
    #         )

    #         failures, successes, accuracy, results = run_evaluation(
    #             bn_json,
    #             cross_target_csv,
    #             target_node=target,
    #         )

    #         print(
    #             f"Cross-target T{target_index} ({target}): "
    #             f"{accuracy * 100:.2f}%"
    #         )

    #         if len(results) > 0:
    #             target_accuracies.append(accuracy)
    #             total_successes += len(successes)
    #             total_scenarios += len(results)

    #     macro_accuracy = (
    #         sum(target_accuracies) / len(target_accuracies)
    #         if target_accuracies else 0.0
    #     )

    #     weighted_accuracy = (
    #         total_successes / total_scenarios
    #         if total_scenarios > 0 else 0.0
    #     )

    #     print(f"Macro cross-target accuracy: {macro_accuracy * 100:.2f}%")
    #     print(f"Weighted cross-target accuracy: {weighted_accuracy * 100:.2f}%")

    # ============================================================
    # Evaluate primary-target accuracy - Experiment B
    # Original BN, AgentBN, and Batch EM (All Subsets)
    # ============================================================

    import json

    PRIMARY_CSV_B = DATASET_DIR / "combined_test_scenarios_exp_b.csv"

    flawed_bn_json = load_bn(FLAWED_BN_FILE)

    # Load all AgentBN records
    agentbn_records = []

    with open(AGENTBN_SUBSET_BN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if "subset_id" in record and "bn" in record:
                    agentbn_records.append(record)

    if not agentbn_records:
        raise ValueError("No AgentBN records found.")

    # Evaluate original BN once (same model for all subsets)
    _, _, original_accuracy, _ = run_evaluation(
        flawed_bn_json,
        PRIMARY_CSV_B,
        target_node=TARGET_NODE,
    )

    print(
        f"\n==================================================\n"
        f"Original BN primary-target accuracy (Experiment B): "
        f"{original_accuracy * 100:.2f}%"
    )

    # Evaluate every stored AgentBN subset
    for record in agentbn_records:

        learned_subset_id = record["subset_id"]
        agentbn_bn_json = record["bn"]

        batch_em_bn_json = get_bn(
            BATCH_EM_BN_FILE,
            bn_number=100,
            oracle_size=0,
            subset_id=learned_subset_id,
        )

        if batch_em_bn_json is None:
            print(
                f"Skipping subset {learned_subset_id}: "
                f"No matching Batch EM BN."
            )
            continue

        print(f"\n{'=' * 50}")
        print(f"Subset ID: {learned_subset_id}")
        print(f"{'=' * 50}")

        for model_name, model_json in {
            "AgentBN": agentbn_bn_json,
            "Batch EM": batch_em_bn_json,
        }.items():

            failures, successes, accuracy, results = run_evaluation(
                model_json,
                PRIMARY_CSV_B,
                target_node=TARGET_NODE,
            )

            print(
                f"{model_name} primary-target accuracy (Experiment B): "
                f"{accuracy * 100:.2f}%"
            )

            print(f"Total scenarios: {len(results)}")
            print(f"Successful scenarios: {len(successes)}")
            print(f"Failed scenarios: {len(failures)}")
    print()
