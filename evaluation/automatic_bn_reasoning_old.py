from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

from utils.pgmpy_tool import *
from utils.bn_io import *
from utils.file_utils import *
from config.settings import *


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
            and confidence >= 0.50
            and margin >= 0.20
        )

        results.append({
            "Scenario": row["Scenario #"],
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

def run_evaluation(bn_json, dataset_file):

    model = build_model(bn_json)

    df = pd.read_csv(dataset_file)
    df.columns = df.columns.str.strip()

    results = call_bn_inference(model, df)
    results_df = pd.DataFrame(results)

    successes = results_df[results_df["Success"] == True]
    failures = results_df[results_df["Success"] == False]

    accuracy = len(successes) / len(results_df)

    print("\n===================================================")
    print(f"Dataset: {dataset_file}")
    print("===================================================")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total:", len(results_df))
    print("Successes:", len(successes))
    print("Failures:", len(failures))

    if failures.empty:
        print("\nNo failures!")
    # else:
    #     print("\nFailures:")
    #     print(failures)

    return failures, successes, accuracy, results


def run_evaluation_2(bn_json, dataset_file):

    model = build_model(bn_json)

    df = pd.read_csv(dataset_file)
    df.columns = df.columns.str.strip()

    results = call_bn_inference(model, df)
    results_df = pd.DataFrame(results)

    successes = results_df[results_df["Success"] == True]
    failures = results_df[results_df["Success"] == False]

    accuracy = len(successes) / len(results_df)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    return failures, successes, accuracy, results


def get_best_bn_number(filename=PROPOSED_BN_FILE, train_csv=TRAIN_CSV):
    best_bn_number = None
    best_bn_accuracy = None
    best_failure_count = float("inf")

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            bn_number = record.get("bn_number")
            bn_json = record.get("bn")

            if not bn_json:
                continue

            failures, successes, accuracy, results = run_evaluation_2(
                bn_json, train_csv
            )

            failure_count = len(failures)

            if failure_count <= best_failure_count:
                best_failure_count = failure_count
                best_bn_number = bn_number
                best_bn_accuracy = accuracy

    return best_bn_number, best_bn_accuracy


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # bn_json = load_bn(GROUND_TRUTH_BN_FILE)

    bn_json = load_bn(FLAWED_BN_FILE)

    dataset_files = [
        "original_scenarios_train.csv",
        "original_scenarios_test.csv",
        "synthetic_scenarios_train.csv",
        "synthetic_scenarios_test.csv",
        "combined_train_scenarios.csv",
        "combined_test_scenarios.csv",
        "Scenarios.csv"
    ]

    all_results = {}

    for dataset_file in dataset_files:
        
        failures, successes, accuracy, results = run_evaluation(
            bn_json,
            dataset_file
        )

        all_results[dataset_file] = {
            "failure_count": len(failures),
            "success_count": len(successes),
            "accuracy": accuracy,
            "results": results
        }

    print("\nReasoning Completed.")
