from langchain_core.prompts import PromptTemplate
from openai import OpenAI

client = OpenAI(api_key="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")

# -------------------------------
# 🔌 Unified LLM Call
# -------------------------------
def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return response.output[0].content[0].text.strip()

###
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# -----------------------------
# Build structure
# -----------------------------
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import json

with open("proposed_bn.json", "r") as f:
    bn_json = json.load(f)

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

# -----------------------------
def run_inference(model, query, evidence=None):
    infer = VariableElimination(model)

    result = infer.query(
        variables=[query],
        evidence=evidence or {}
    )

    return result

# -----------------------------
# RUN (Dynamic from CSV)
# -----------------------------
import pandas as pd

model = build_model(bn_json)

# Load scenarios
df = pd.read_csv("Scenarios.csv")
df.columns = df.columns.str.strip()

results = []

# -----------------------------
# INFERENCE LOOP
# -----------------------------
for _, row in df.iterrows():

    evidence = {}

    for col in df.columns:
        if col in ["Scenario #", "Ground Truth"]:
            continue

        val = row[col]

        if pd.isna(val):
            continue

        evidence[col] = val.strip()

    print("Running inference for scenario:", row["Scenario #"])
    print(evidence)

    # -----------------------------
    # Run inference
    # -----------------------------
    result = run_inference(
        model,
        query="Root_Causes",
        evidence=evidence
    )

    print(result)

    # Extract prediction
    pred_idx = result.values.argmax()
    pred_state = result.state_names["Root_Causes"][pred_idx]
    confidence = result.values.max()

    # Store results
    results.append({
        "Scenario": row["Scenario #"],
        "Prediction": pred_state,
        "Confidence": float(confidence),
        "Ground Truth": row["Ground Truth"]
    })

# -----------------------------
# RESULTS DF
# -----------------------------
results_df = pd.DataFrame(results)
print(results_df)

# -----------------------------
# ACCURACY
# -----------------------------
accuracy = (results_df["Prediction"] == results_df["Ground Truth"]).mean()
print(f"Accuracy: {accuracy:.3f}")

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

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
    
failures = results_df[results_df["Prediction"] != results_df["Ground Truth"]]
successes = results_df[results_df["Prediction"] == results_df["Ground Truth"]]
failure_text = format_results_for_llm(failures, df)
success_text = format_results_for_llm(successes, df)

print(failure_text)
print(success_text)

context = read_file("context_eval_agent.txt")
base_prompt = read_file("eval_prompt.txt")

final_prompt = base_prompt.format(
    context=context,
    bn_json=bn_json,
    failure_text=failure_text,
    success_text=success_text
)

analysis = llm(final_prompt)
print(analysis)

from datetime import datetime

def clean_text(text):
    return (
        text.replace("\\n", " ")
            .replace("\n", " ")
            .replace("- ", "")
            .strip()
    )

bn_number = 1
record = {
    "timestamp": str(datetime.now()),
    "bn_number": bn_number,
    "failure_text": clean_text(failure_text),
    "success_text": clean_text(success_text),
    "analysis": clean_text(analysis)
}

with open("bn_analysis.json", "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
