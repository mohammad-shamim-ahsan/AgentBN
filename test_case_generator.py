import csv
import json
import re
import os
from io import StringIO

import pandas as pd
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


client = OpenAI(api_key="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")

MODEL = "gpt-5.4"

INITIAL_SCENARIOS_FILE = "Scenarios.csv"
CONTEXT_FILE = "context_eval_agent.txt"
PROMPT_FILE = "scenario_gen_prompt.txt"
# proposed_bn_filename = "last_proposed_bn.jsonl"
BN_FILE = "BN_gt.json"

CANDIDATE_OUTPUT_FILE = "candidate_generated_scenarios.csv"
VALIDATED_OUTPUT_FILE = "validated_generated_scenarios.csv"
INFERENCE_OUTPUT_FILE = "scenario_inference_results.json"


def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    return response.output[0].content[0].text.strip()


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def clean_csv_output(text):
    text = text.strip()
    text = re.sub(r"^```csv", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


# def get_last_bn(filename):
#     last_record = None

#     with open(filename, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 last_record = json.loads(line.strip())

#     if last_record is None:
#         raise ValueError(f"No BN found in {filename}")

#     return last_record["bn"]


# def get_last_bn_as_text(filename):
#     bn = get_last_bn(filename)
#     return json.dumps(bn, indent=2)

def read_bn(filename=BN_FILE):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def read_bn_as_text(filename=BN_FILE):
    bn = read_bn(filename)
    return json.dumps(bn, indent=2)


def store_text(text, filename):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write(text.strip() + "\n")


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

        if not parents:
            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=cpt["values"],
                state_names=state_names
            )
        else:
            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=cpt["values"],
                evidence=parents,
                evidence_card=evidence_card,
                state_names=state_names
            )

        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()

    return model


def run_inference(model, query, evidence=None):
    infer = VariableElimination(model)

    result = infer.query(
        variables=[query],
        evidence=evidence or {}
    )

    return result


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
            query="Root_Causes",
            evidence=evidence
        )

        pred_idx = result.values.argmax()
        pred_state = result.state_names["Root_Causes"][pred_idx]
        confidence = result.values.max()

        posterior_probs = {
            state: float(prob)
            for state, prob in zip(
                result.state_names["Root_Causes"],
                result.values
            )
        }

        ground_truth = str(row["Ground Truth"]).strip()

        results.append({
            "Scenario": row["Scenario #"],
            "Prediction": pred_state,
            "Confidence": float(confidence),
            "Ground Truth": ground_truth,
            "Matched": pred_state == ground_truth,
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
            "latest_bn"
        ],
        template=prompt_template_text
    )

    prompt = prompt_template.format(
        full_context=full_context,
        examples_csv=examples_csv_text,
        columns=",".join(expected_columns),
        latest_bn=latest_bn_text
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


NUM_ITERATIONS = 3

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
        filename="final_inference_results.json"
    )

    not_validated = [
        result for result in final_inference_results
        if not result["Matched"]
    ]

    if not_validated:
        print("\nWARNING: Some merged scenarios are NOT BN-validated.")
        print(f"Number of non-validated scenarios: {len(not_validated)}")

        with open("final_not_validated_scenarios.json", "w", encoding="utf-8") as f:
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
        "final_validated_dataset.csv",
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


if __name__ == "__main__":
    main()

# import pandas as pd

# # Read CSV files
# df1 = pd.read_csv("Scenarios.csv")
# df2 = pd.read_csv("final_validated_dataset.csv")

# # Merge rows
# merged_df = pd.concat([df1, df2], ignore_index=True)

# # Remove duplicate rows
# merged_df = merged_df.drop_duplicates()

# # Renumber Scenario #
# merged_df = merged_df.reset_index(drop=True)

# if "Scenario #" in merged_df.columns:
#     merged_df["Scenario #"] = range(1, len(merged_df) + 1)

# # Save merged dataset
# merged_df.to_csv("final_validated_dataset.csv", index=False)

# print(f"Total merged rows: {len(merged_df)}")
