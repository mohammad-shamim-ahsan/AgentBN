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
            query=TARGET_NODE,
            evidence=evidence
        )

        pred_idx = np.argmax(result.values)
        pred_state = result.state_names[TARGET_NODE][pred_idx]
        confidence = result.values[pred_idx]

        probs = sorted(result.values, reverse=True)
        max_prob = probs[0]
        second_prob = probs[1]
        margin = max_prob - second_prob

        posterior_probs = {
            state: float(prob)
            for state, prob in zip(
                result.state_names[TARGET_NODE],
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

def run_evaluation(bn_json, dataset_file):

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

    print(f"EXPERIMENT = {EXPERIMENT!r}")

    DATASET_DIR = Path("datasets") / EXPERIMENT

    INITIAL_SCENARIOS_FILE = DATASET_DIR / "Scenarios.csv"
    FINAL_OUTPUT_FILE = DATASET_DIR / "final_validated_dataset.csv"

    TRAIN_CSV = DATASET_DIR / "combined_train_scenarios.csv"
    TEST_CSV = DATASET_DIR / "combined_test_scenarios.csv"
    
    BN_FILE = DATASET_DIR / "BN_gt.json"
    FLAWED_BN_FILE = DATASET_DIR / "flawed_BN_0.json"

    bn_json = load_bn(FLAWED_BN_FILE)
    failures, successes, accuracy, results = run_evaluation(bn_json, TRAIN_CSV)
    
    print("\nReasoning Completed.")
