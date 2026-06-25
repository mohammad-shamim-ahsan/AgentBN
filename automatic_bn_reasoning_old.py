import os

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

# ============================================================
# Step 1: Read BN (JSON-like format: GeNIe XML equivalent)
# ============================================================
import json

def load_bn(filename):
    with open(filename, "r") as f:
        return json.load(f)

# ============================================================
# Step 2: Building BN Tool
# ============================================================

def build_model(bn):
    edges = bn["edges"]
    model = DiscreteBayesianNetwork(edges)

    cpds = []

    for node in bn["nodes"]:

        name = node["name"]
        states = node["states"]
        parents = node.get("parents", [])
        cpt = node["cpt"]

        state_names = {name: states}

        if parents:

            parent_order = cpt["parent_state_order"]

            for p in parents:
                state_names[p] = parent_order[p]

            evidence_card = [
                len(parent_order[p])
                for p in parents
            ]

        else:
            evidence_card = None

        cpd = TabularCPD(
            variable=name,
            variable_card=len(states),
            values=cpt["values"],
            evidence=parents if parents else None,
            evidence_card=evidence_card,
            state_names=state_names
        )

        cpds.append(cpd)

    model.add_cpds(*cpds)

    print("\nChecking model...")
    print(model.check_model())

    return model

# ============================================================
# Step 3: Inference
# ============================================================

def run_inference(model, query, evidence=None):

    infer = VariableElimination(model)

    result = infer.query(
        variables=[query],
        evidence=evidence or {}
    )

    return result

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

        posterior_probs = {
            state: float(prob)
            for state, prob in zip(
                result.state_names["Root_Causes"],
                result.values
            )
        }

        # Success only if:
        # 1. Predicted class matches Ground Truth
        # 2. Confidence >= 60%
        is_success = (
            pred_state == row["Ground Truth"]
            and confidence >= 0.60
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

def get_best_bn_number(filename="last_proposed_bn.jsonl", train_csv="combined_train_scenarios.csv"):
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

    bn_json = load_bn("BN_gt.json")
    # bn_json = load_bn("flawed_BN_0.json")

    # best_bn_number, best_bn_accuracy = get_best_bn_number(proposed_bn_filename)
    # print("\n\nBest BN Number:", best_bn_number)
    # print("Best BN Accuracy:", best_bn_accuracy)
    # bn_json = find_proposed_bn(bn_number_to_find=best_bn_number, filename=proposed_bn_filename)

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
