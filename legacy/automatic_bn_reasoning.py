import os

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config.settings import *
from utils.pgmpy_tool import run_inference, build_model

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

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

def call_bn_inference(model, df):

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
            query="Root_Causes",
            evidence=evidence
        )

        pred_idx = np.argmax(result.values)
        pred_state = result.state_names["Root_Causes"][pred_idx]
        confidence = result.values[pred_idx]

        probs = sorted(result.values, reverse=True)
        max_prob = probs[0]
        second_prob = probs[1]
        margin = max_prob - second_prob

        posterior_probs = {
            state: float(prob)
            for state, prob in zip(
                result.state_names["Root_Causes"],
                result.values
            )
        }

        # Success only if:
        # 1. Predicted class matches Ground Truth
        # 2. Confidence >= 50%
        # 3. Margin >= 20%
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

###
import json

proposed_bn_filename="last_proposed_bn.jsonl"

def find_proposed_bn(bn_number_to_find, filename=proposed_bn_filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["bn_number"] == bn_number_to_find:
                return record["bn"]
    return None

# ============================================================
# Step 5: Run Evaluation
# ============================================================

def run_evaluation(bn_json, dataset_file, prefix, train_test_split_ratio=0.33):

    model = build_model(bn_json)

    df = pd.read_csv(dataset_file)
    df.columns = df.columns.str.strip()

    results = call_bn_inference(model, df)

    results_df = pd.DataFrame(results)

    successes = results_df[
        results_df["Success"] == True
    ]

    failures = results_df[
        results_df["Success"] == False
    ]

    accuracy = len(successes) / len(results_df)

    print("\n===================================================")
    print(f"Dataset: {dataset_file}")
    print("===================================================")

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    # ----------------------------------------------------
    # Use stratified split only when every class has
    # at least two samples.
    # ----------------------------------------------------
    class_counts = results_df["Success"].value_counts()

    use_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
    )

    if use_stratify:

        train_df, test_df = train_test_split(
            results_df,
            test_size=train_test_split_ratio,
            stratify=results_df["Success"],
            random_state=42
        )

        print("\nUsing stratified train/test split.")

    else:

        train_df, test_df = train_test_split(
            results_df,
            test_size=train_test_split_ratio,
            random_state=42
        )

        print(
            "\nWarning: Stratified split skipped because "
            "one class has fewer than two samples."
        )

        print("Class counts:")
        print(class_counts)

    # Preserve raw scenarios corresponding to the split
    train_scenarios = df.iloc[train_df.index].copy()
    test_scenarios = df.iloc[test_df.index].copy()

    train_successes = train_df[
        train_df["Success"] == True
    ]

    train_failures = train_df[
        train_df["Success"] == False
    ]

    test_successes = test_df[
        test_df["Success"] == True
    ]

    test_failures = test_df[
        test_df["Success"] == False
    ]

    print("\n===================================================")
    print("TRAIN / TEST SPLIT SUMMARY")
    print("===================================================")

    print(
        f"Train: {len(train_df)} "
        f"(Successes={len(train_successes)}, "
        f"Failures={len(train_failures)})"
    )

    print(
        f"Test:  {len(test_df)} "
        f"(Successes={len(test_successes)}, "
        f"Failures={len(test_failures)})"
    )

    with open(f"{prefix}_flawed_success_train.json", "w") as f:
        json.dump(
            train_successes.to_dict(orient="records"),
            f,
            indent=4
        )

    with open(f"{prefix}_flawed_success_test.json", "w") as f:
        json.dump(
            test_successes.to_dict(orient="records"),
            f,
            indent=4
        )

    with open(f"{prefix}_flawed_failure_train.json", "w") as f:
        json.dump(
            train_failures.to_dict(orient="records"),
            f,
            indent=4
        )

    with open(f"{prefix}_flawed_failure_test.json", "w") as f:
        json.dump(
            test_failures.to_dict(orient="records"),
            f,
            indent=4
        )

    # Save raw train/test scenarios for future BN evaluation
    train_scenarios.to_csv(
        f"{prefix}_scenarios_train.csv",
        index=False
    )

    test_scenarios.to_csv(
        f"{prefix}_scenarios_test.csv",
        index=False
    )

    return (
        train_failures,
        train_successes,
        test_failures,
        test_successes,
        accuracy,
        results
    )

def merge_json_files(input_files, output_file):

    merged = []

    for file in input_files:
        with open(file, "r") as f:
            merged.extend(json.load(f))

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"Merged -> {output_file}")


def remove_files(files):

    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")


if __name__ == "__main__":

    bn_json = load_bn("flawed_BN_0.json")

    train_test_split_ratio_original=0.33
    
    train_test_split_ratio_synthetic=0.20

    run_evaluation(
        bn_json,
        "Scenarios.csv",
        "original",
        train_test_split_ratio=train_test_split_ratio_original
    )

    run_evaluation(
        bn_json,
        "final_validated_dataset.csv",
        "synthetic",
        train_test_split_ratio=train_test_split_ratio_synthetic
    )

    merge_json_files(
        [
            "original_flawed_success_train.json",
            "synthetic_flawed_success_train.json"
        ],
        "merged_flawed_success_train.json"
    )

    merge_json_files(
        [
            "original_flawed_failure_train.json",
            "synthetic_flawed_failure_train.json"
        ],
        "merged_flawed_failure_train.json"
    )

    merge_json_files(
        [
            "original_flawed_success_test.json",
            "synthetic_flawed_success_test.json"
        ],
        "merged_flawed_success_test.json"
    )

    merge_json_files(
        [
            "original_flawed_failure_test.json",
            "synthetic_flawed_failure_test.json"
        ],
        "merged_flawed_failure_test.json"
    )

    print("\nReasoning Completed.")

    remove_files([
        "original_flawed_success_train.json",
        "original_flawed_success_test.json",
        "original_flawed_failure_train.json",
        "original_flawed_failure_test.json",
        "synthetic_flawed_success_train.json",
        "synthetic_flawed_success_test.json",
        "synthetic_flawed_failure_train.json",
        "synthetic_flawed_failure_test.json"
    ])

    original_df = pd.read_csv("original_scenarios_train.csv")
    synthetic_df = pd.read_csv("synthetic_scenarios_train.csv")
    train_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    train_df.to_csv("combined_train_scenarios.csv", index=False)

    original_df = pd.read_csv("original_scenarios_test.csv")
    synthetic_df = pd.read_csv("synthetic_scenarios_test.csv")
    test_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    test_df.to_csv("combined_test_scenarios.csv", index=False)
