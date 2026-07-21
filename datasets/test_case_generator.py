import csv
import json
import re
import os
from io import StringIO
import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
from sklearn.model_selection import train_test_split

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import *
from utils.llm import *
from utils.pgmpy_tool import *


NUM_ITERATIONS = 5

DATASET_DIR = Path("datasets") / EXPERIMENT
PROMPT_DIR = Path("prompts") / EXPERIMENT

INITIAL_SCENARIOS_FILE = DATASET_DIR / "Scenarios.csv"
CONTEXT_FILE = PROMPT_DIR / "context_agent.txt"
PROMPT_FILE = Path("prompts") / "scenario_gen_prompt.txt"
BN_FILE = DATASET_DIR / "BN_gt.json"

CANDIDATE_OUTPUT_FILE = DATASET_DIR / "candidate_generated_scenarios.csv"
VALIDATED_OUTPUT_FILE = DATASET_DIR / "validated_generated_scenarios.csv"
INFERENCE_OUTPUT_FILE = DATASET_DIR / "scenario_inference_results.json"
FINAL_OUTPUT_FILE = DATASET_DIR / "final_validated_dataset.csv"
FINAL_INFERENCE_FILE = DATASET_DIR / "final_inference_results.json"

FINAL_NOT_VALIDATED_FILE = DATASET_DIR / "final_not_validated_scenarios.json"


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def clean_csv_output(text):
    text = text.strip()
    text = re.sub(r"^```csv", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def read_bn(filename=BN_FILE):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def read_bn_as_text(filename=BN_FILE):
    bn = read_bn(filename)
    return json.dumps(bn, indent=2)


def store_text(text, filename):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write(text.strip() + "\n")


def call_bn_inference(model, df):
    results = []

    for _, row in df.iterrows():
        evidence = {}

        for col in df.columns:
            col_clean = col.strip()

            if col_clean in ["Scenario #", "Ground Truth"]:
                continue

            val = row[col]

            if pd.isna(val):
                continue

            evidence[col_clean] = str(val).strip()

        print("Running inference for scenario:", row["Scenario #"])

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

        ground_truth = str(row["Ground Truth"]).strip()

        is_success = bool(
            pred_state == ground_truth
            and confidence >= MIN_CONFIDENCE
            and margin >= MIN_MARGIN
        )

        results.append({
            "Scenario": row["Scenario #"],
            "Prediction": pred_state,
            "Confidence": float(confidence),
            "Ground Truth": ground_truth,
            "Matched": is_success,
            "Posterior": posterior_probs
        })

    return results


def generate_candidate_scenarios(examples_csv_text):
    full_context = read_file(CONTEXT_FILE)

    latest_bn_text = read_bn_as_text(BN_FILE)

    prompt_template_text = read_file(PROMPT_FILE)

    expected_columns = next(
        csv.reader(StringIO(examples_csv_text))
    )

    prompt_template = PromptTemplate(
        input_variables=[
            "full_context",
            "examples_csv",
            "columns",
            "latest_bn",
            "target_node",
            "min_confidence",
            "min_margin",
        ],
        template=prompt_template_text
    )

    prompt = prompt_template.format(
        full_context=full_context,
        examples_csv=examples_csv_text,
        latest_bn=latest_bn_text,
        target_node=TARGET_NODE,
        min_confidence=MIN_CONFIDENCE,
        min_margin=MIN_MARGIN,
        columns=",".join(expected_columns),
    )

    return llm(prompt)


def dataframe_to_csv_text(df):
    return df.to_csv(index=False)


def load_candidate_df(csv_text):
    csv_text = clean_csv_output(csv_text)

    df = pd.read_csv(StringIO(csv_text))

    # Fix accidental spaces in headers from LLM
    df.columns = [c.strip() for c in df.columns]

    # Fix accidental spaces in cell values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df


def filter_matched_scenarios(df, inference_results):
    matched_ids = {
        result["Scenario"]
        for result in inference_results
        if result["Matched"]
    }

    matched_df = df[df["Scenario #"].isin(matched_ids)].copy()

    matched_df = matched_df.reset_index(drop=True)

    for i in range(len(matched_df)):
        matched_df.loc[i, "Scenario #"] = i + 1

    return matched_df


def store_inference_results(results, filename=INFERENCE_OUTPUT_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    bn = read_bn(BN_FILE)

    model = build_model(bn)

    # -----------------------------
    # Iteration 1 examples
    # -----------------------------
    current_examples_csv = read_file(
        INITIAL_SCENARIOS_FILE
    )

    all_validated_dfs = []

    for iteration in range(1, NUM_ITERATIONS + 1):

        print("\n" + "=" * 80)
        print(f"ITERATION {iteration}")
        print("=" * 80)

        # -----------------------------
        # Generate candidate scenarios
        # -----------------------------
        candidate_csv = generate_candidate_scenarios(
            current_examples_csv
        )

        candidate_csv = clean_csv_output(candidate_csv)

        candidate_filename = (
            f"candidate_scenarios_iter_{iteration}.csv"
        )

        store_text(candidate_csv, candidate_filename)

        candidate_df = load_candidate_df(candidate_csv)

        print(
            f"\nGenerated {len(candidate_df)} candidate scenarios"
        )

        # -----------------------------
        # BN inference validation
        # -----------------------------
        inference_results = call_bn_inference(
            model,
            candidate_df
        )

        inference_filename = (
            f"inference_results_iter_{iteration}.json"
        )

        store_inference_results(
            inference_results,
            filename=inference_filename
        )

        # -----------------------------
        # Keep matched scenarios only
        # -----------------------------
        validated_df = filter_matched_scenarios(
            candidate_df,
            inference_results
        )

        validated_filename = (
            f"validated_scenarios_iter_{iteration}.csv"
        )

        validated_df.to_csv(
            validated_filename,
            index=False
        )

        print(
            f"Validated {len(validated_df)} / "
            f"{len(candidate_df)} scenarios"
        )

        # -----------------------------
        # Store validated set
        # -----------------------------
        all_validated_dfs.append(validated_df)

        # -----------------------------
        # Next iteration uses validated
        # scenarios as examples
        # -----------------------------
        current_examples_csv = dataframe_to_csv_text(
            validated_df
        )

    # -----------------------------
    # Final merged dataset
    # -----------------------------
    final_df = pd.concat(
        all_validated_dfs,
        ignore_index=True
    )

    # Remove duplicates
    dedup_columns = [
        col for col in final_df.columns
        if col != "Scenario #"
    ]

    final_df = final_df.drop_duplicates(
        subset=dedup_columns
    ).reset_index(drop=True)

    final_df["Scenario #"] = range(
        1,
        len(final_df) + 1
    )

    # Final BN validation check
    final_inference_results = call_bn_inference(
        model,
        final_df
    )

    store_inference_results(
        final_inference_results,
        filename=FINAL_INFERENCE_FILE
    )

    not_validated = [
        result for result in final_inference_results
        if not result["Matched"]
    ]

    if not_validated:
        print("\nWARNING: Some merged scenarios are NOT BN-validated.")
        print(f"Number of non-validated scenarios: {len(not_validated)}")

        with open(FINAL_NOT_VALIDATED_FILE, "w", encoding="utf-8") as f:
            json.dump(not_validated, f, indent=2)

    else:
        print("\nAll merged unique scenarios are BN-validated.")

    # Keep only validated rows
    final_df = filter_matched_scenarios(
        final_df,
        final_inference_results
    )

    final_df["Scenario #"] = range(
        1,
        len(final_df) + 1
    )

    final_df.to_csv(
        FINAL_OUTPUT_FILE,
        index=False
    )

    print("\nFINAL DATASET GENERATED")
    print(f"Total final validated unique scenarios: {len(final_df)}")

    # --------------------------------
    # Cleanup iteration files
    # --------------------------------
    for iteration in range(1, NUM_ITERATIONS + 1):

        files_to_remove = [
            f"candidate_scenarios_iter_{iteration}.csv",
            f"validated_scenarios_iter_{iteration}.csv",
            f"inference_results_iter_{iteration}.json"
        ]

        for filepath in files_to_remove:
            if os.path.exists(filepath):
                os.remove(filepath)

    print("\nIntermediate iteration files removed.")


def remove_seed_scenarios():
    # Read files
    scenarios_df = pd.read_csv(INITIAL_SCENARIOS_FILE)
    final_df = pd.read_csv(FINAL_OUTPUT_FILE)

    before_count = len(final_df)

    # Ignore Scenario # when comparing
    compare_cols = [c for c in final_df.columns if c != "Scenario #"]

    # Remove rows that exist in Scenarios.csv
    remaining_df = (
        final_df.merge(
            scenarios_df[compare_cols].drop_duplicates(),
            on=compare_cols,
            how="left",
            indicator=True
        )
    )

    remaining_df = (
        remaining_df[remaining_df["_merge"] == "left_only"]
        .drop(columns=["_merge"])
    )

    # Renumber Scenario #
    remaining_df = remaining_df.reset_index(drop=True)
    if "Scenario #" in remaining_df.columns:
        remaining_df["Scenario #"] = range(1, len(remaining_df) + 1)

    after_count = len(remaining_df)

    # Save
    remaining_df.to_csv(FINAL_OUTPUT_FILE, index=False)

    print(f"Rows before removal: {before_count}")
    print(f"Rows after removal:  {after_count}")
    print(f"Rows removed:        {before_count - after_count}")


def combine_and_split():
    # Read files
    seed_df = pd.read_csv(INITIAL_SCENARIOS_FILE)
    generated_df = pd.read_csv(FINAL_OUTPUT_FILE)

    # Combine
    combined_df = pd.concat([seed_df, generated_df], ignore_index=True)

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if "Scenario #" in combined_df.columns:
        combined_df["Scenario #"] = range(1, len(combined_df) + 1)

    # Check class distribution
    class_counts = combined_df["Ground Truth"].value_counts()

    print("\nGround Truth distribution:")
    print(class_counts)

    # Decide whether to stratify
    if class_counts.min() >= 2:
        print("\nUsing stratified train/test split.")
        stratify = combined_df["Ground Truth"]
    else:
        print("\nNot enough samples for stratification. Using random split.")
        stratify = None

    train_df, test_df = train_test_split(
        combined_df,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )

    # Renumber
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["Scenario #"] = range(1, len(train_df) + 1)
    test_df["Scenario #"] = range(1, len(test_df) + 1)

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\nTotal : {len(combined_df)}")
    print(f"Train : {len(train_df)}")
    print(f"Test  : {len(test_df)}")


### --------------------------------------------------------
if __name__ == "__main__":
    main()
    remove_seed_scenarios()
    combine_and_split()
