from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np

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

        print(f"\nRunning inference for Scenario: {row['Scenario #']}")
        print("Evidence:", evidence)

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

        results.append({
            "Scenario": row["Scenario #"],
            "Prediction": pred_state,
            "Confidence": float(confidence),
            "Ground Truth": row["Ground Truth"],
            "Posterior": posterior_probs
        })

    return results

# ============================================================
# Step 5: Run Evaluation
# ============================================================

def run_evaluation(bn_json):

    model = build_model(bn_json)

    df = pd.read_csv("Scenarios.csv")
    df.columns = df.columns.str.strip()

    results = call_bn_inference(model, df)

    results_df = pd.DataFrame(results)

    successes = results_df[
        results_df["Prediction"] == results_df["Ground Truth"]
    ]

    failures = results_df[
        results_df["Prediction"] != results_df["Ground Truth"]
    ]

    accuracy = len(successes) / len(results_df)

    print("\n===================================================")
    print("RESULTS")
    print("===================================================")

    print(results_df)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nFailures:")
    if failures.empty:
        print("No failures!")
    else:
        print(failures)

    return failures, successes, accuracy, results

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    bn_json = load_bn("BN_gt.json")
    failures, successes, accuracy, results = run_evaluation(bn_json)
    print("\nReasoning Completed.")
