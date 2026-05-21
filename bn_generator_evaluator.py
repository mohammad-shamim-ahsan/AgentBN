from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json
import pandas as pd
from datetime import datetime

client = OpenAI(api_key="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")

def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return response.output[0].content[0].text.strip()

###--------Step 1: Building BN Tool---------------------
def build_model(bn):
    edges = bn["edges"]
    model = DiscreteBayesianNetwork(edges)

    cpds = []

    for node in bn["nodes"]:
        name = node["name"]
        states = node["states"]
        parents = node.get("parents", [])

        cpt = node["cpt"]

        # -----------------------------
        # Build state_names safely
        # -----------------------------
        state_names = {name: states}

        if parents:
            parent_order = cpt["parent_state_order"]
            for p in parents:
                state_names[p] = parent_order[p]

            evidence_card = [len(parent_order[p]) for p in parents]

        else:
            evidence_card = None

        # -----------------------------
        # Root node
        # -----------------------------
        if not parents:
            values = cpt["values"]

            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=values,
                state_names=state_names
            )

        # -----------------------------
        # Child node
        # -----------------------------
        else:
            values = cpt["values"]

            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=values,
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

# -----------------------------
# Step 2: Calling BN Tool for the Scenarios
# -----------------------------
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

            evidence[col] = val.strip()

        # print("Running inference for scenario:", row["Scenario #"])
        # print(evidence)

        result = run_inference(
            model,
            query="Root_Causes",
            evidence=evidence
        )

        # print(result)

        pred_idx = result.values.argmax()
        pred_state = result.state_names["Root_Causes"][pred_idx]
        confidence = result.values.max()

        # Store posterior probabilities
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

###--------Step 3: Getting Analysis Results for LLM---------------------
def format_results_for_llm(results_df, original_df):

    lines = []

    for i, row in results_df.iterrows():

        original_row = original_df.iloc[i].to_dict()

        # remove metadata fields
        original_row.pop("Scenario #", None)
        original_row.pop("Ground Truth", None)

        status = "CORRECT" if row["Prediction"] == row["Ground Truth"] else "WRONG"

        lines.append(
            f"Scenario {row['Scenario']}:\n"
            f"- Evidence: {original_row}\n"
            f"- Prediction: {row['Prediction']} (confidence={row['Confidence']:.3f})\n"
            f"- Ground Truth: {row['Ground Truth']}\n"
            f"- Status: {status}\n"
        )

    return "\n".join(lines)

###------------------RUN---------------------------------
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

import json

proposed_bn_filename="last_proposed_bn.jsonl"

def find_proposed_bn(bn_number_to_find, filename=proposed_bn_filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["bn_number"] == bn_number_to_find:
                return record["bn"]
    return None

df = pd.read_csv("final_validated_dataset.csv")
df.columns = df.columns.str.strip()

context = read_file("context_eval_agent.txt")
base_prompt = read_file("eval_prompt.txt")

def clean_text(text):
    return (
        text.replace("\\n", " ")
            .replace("\n", " ")
            .replace("- ", "")
            .strip()
    )

def run_evaluation(bn_json):
    model = build_model(bn_json) # --- Step 1: Build the BN model from the generated JSON
    results = call_bn_inference(model, df) # --- Step 2: Call the BN inference for each scenario and get predictions
    results_df = pd.DataFrame(results)

    failures = results_df[results_df["Prediction"] != results_df["Ground Truth"]]
    successes = results_df[results_df["Prediction"] == results_df["Ground Truth"]]

    failure_text = format_results_for_llm(failures, df) # --- Step 3: Format the results for LLM analysis
    success_text = format_results_for_llm(successes, df)

    final_prompt = base_prompt.format(
        context=context,
        bn_json=json.dumps(bn_json, indent=2),
        failure_text=failure_text,
        success_text=success_text
    )

    analysis = llm(final_prompt)

    return failure_text, success_text, analysis, results

def store_analysis(bn_number, failure_text, success_text, analysis, results):
    record = {
        "timestamp": str(datetime.now()),
        "bn_number": bn_number,
        "failure_text": failure_text,
        "success_text": success_text,
        "analysis": analysis,
        "scenario_results": results
    }

    with open("bn_analysis.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        print(f"\nAnalysis for BN #{bn_number} stored in bn_analysis.json")

if __name__ == "__main__":
    bn_number = 1
    bn_json = find_proposed_bn(bn_number, proposed_bn_filename)
    failure_text, success_text, analysis, results = run_evaluation(bn_json)
    print(analysis)
    store_analysis(bn_number, failure_text, success_text, analysis, results)
