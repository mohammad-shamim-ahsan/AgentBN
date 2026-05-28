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

###-----------------------------
PARAMETER_RISK_FILE = "failure_cpt_parameter_risks.jsonl"
CPT_DANGER_REPORT_FILE = "dangerous_cpt_report.json"

def safe_json_loads(text):
    if not text:
        return None

    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    
def analyze_failure_parameters_with_llm(bn_json, failure_scenario_text):
    prompt = f"""
You are analyzing a Bayesian Network failure case.

Domain context:
The Bayesian Network is used to distinguish undetected faults and cyberattacks
in DER systems.

GPTN nodes represent global cyber-physical consistency/deviation patterns.
LPTN nodes represent local program-variable deviation patterns.

GPTN/LPTN nodes are evidence/prior nodes in the BN.
CPTs are expert-driven and model uncertainty from imperfect logs, packets,
measurements, and insufficient data sources.

Your task:
Identify the conditional CPT paths activated by this failure scenario.
A conditional CPT path means the sequence of CPT rows/parameters used or influenced
by the scenario evidence, from observed evidence nodes through intermediate nodes
to the predicted/root-cause node.

Important:
- Do not assume there is only one path.
- Include paths supporting the wrong prediction.
- Include paths that should have supported the ground truth but appear weak.

A CPT parameter means:
- CPT name / node name
- parent condition values
- target state probability that may be wrong

Return ONLY valid JSON.

Required JSON format:
{{
  "failure_scenario": "",
  "identified_cpt_parameters": [
    {{
      "cpt": "",
      "target_state": "",
      "parent_conditions": {{}},
      "suspected_issue": "",
      "reason": ""
    }}
  ]
}}

Bayesian Network JSON:
{json.dumps(bn_json, indent=2)}

Failure scenario:
{failure_scenario_text}
"""

    response = llm(prompt)
    parsed = safe_json_loads(response)

    if parsed is None:
        return {
            "failure_scenario": "unknown",
            "identified_cpt_parameters": [],
            "raw_response": str(response)
        }

    return parsed

def check_parameter_against_successes_with_llm(
    bn_json,
    parameter_record,
    success_text
):
    prompt = f"""
You are checking whether a suspicious CPT parameter from a failed scenario is also strongly involved in successful scenarios.

Decision rule:
- If this CPT parameter appears necessary for many/all success scenarios, mark it as PROTECTED.
- If it is not clearly involved in successes, mark it as NOT PROTECTED.
- If uncertain, mark it as NOT PROTECTED.

Return ONLY valid JSON.

Required JSON format:
{{
  "cpt": "",
  "target_state": "",
  "parent_conditions": {{}},
  "protected_by_successes": true,
  "success_involvement_summary": "",
  "should_store_as_risky": false
}}

Suspicious CPT parameter:
{json.dumps(parameter_record, indent=2)}

Bayesian Network JSON:
{json.dumps(bn_json, indent=2)}

Successful scenarios:
{success_text}
"""

    response = llm(prompt)
    parsed = safe_json_loads(response)

    if parsed is None:
        return {
            **parameter_record,
            "protected_by_successes": False,
            "success_involvement_summary": "Could not parse LLM response; storing as risky by default.",
            "should_store_as_risky": True,
            "raw_response": str(response)
        }

    return parsed

def append_risky_failure_record(record, filename=PARAMETER_RISK_FILE):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def generate_cpt_danger_report(bn_json, risk_file=PARAMETER_RISK_FILE):
    risky_records = []

    try:
        with open(risk_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    risky_records.append(json.loads(line))
    except FileNotFoundError:
        risky_records = []

    prompt = f"""
You are analyzing risky CPT parameters found across failed Bayesian Network scenarios.

Domain context:
The BN is designed to distinguish undetected faults and cyberattacks in DER systems.
GPTN/LPTN pattern nodes bridge cyber-side observations and physical-side observations.
Repeated risky CPTs may indicate unstable expert-driven probability assumptions.

Your task:
Identify which CPTs are most dangerous overall.

A dangerous CPT is one that:
- appears repeatedly in failure scenarios
- contains multiple suspicious parameters
- may strongly affect wrong predictions
- is not strongly protected by successful scenarios

Return ONLY valid JSON.

Required JSON format:
{{
  "dangerous_cpts": [
    {{
      "cpt": "",
      "risk_level": "high/medium/low",
      "number_of_failure_scenarios": 0,
      "main_problem": "",
      "summary": "",
      "recommended_action": ""
    }}
  ],
  "overall_summary": ""
}}

Bayesian Network JSON:
{json.dumps(bn_json, indent=2)}

Risky CPT parameter records:
{json.dumps(risky_records, indent=2)}
"""

    response = llm(prompt)
    parsed = safe_json_loads(response)

    if parsed is None:
        parsed = {
            "dangerous_cpts": [],
            "overall_summary": "Could not parse final LLM report.",
            "raw_response": str(response)
        }

    with open(CPT_DANGER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    return parsed

def run_evaluation(bn_json):
    # Step 1: Build BN model
    model = build_model(bn_json)

    # Step 2: Run BN inference
    results = call_bn_inference(model, df)
    results_df = pd.DataFrame(results)

    # Step 3: Split failures and successes
    failures = results_df[results_df["Prediction"] != results_df["Ground Truth"]]
    successes = results_df[results_df["Prediction"] == results_df["Ground Truth"]]

    # Step 4: Format successes once
    success_text = format_results_for_llm(successes, df)

    all_failure_records = []

    total_failures = len(failures)
    processed_failures = 0

    # Step 5: Process one failure scenario at a time
    for failure_index, failure_row in failures.iterrows():
        single_failure_df = pd.DataFrame([failure_row])
        failure_scenario_text = format_results_for_llm(single_failure_df, df)

        # Step 5.1: Identify CPT parameters involved in this failure
        failure_parameter_analysis = analyze_failure_parameters_with_llm(
            bn_json=bn_json,
            failure_scenario_text=failure_scenario_text
        )

        risky_parameters = []

        # Step 5.2: Check each parameter against success scenarios
        for parameter_record in failure_parameter_analysis.get(
            "identified_cpt_parameters", []
        ):
            checked_parameter = check_parameter_against_successes_with_llm(
                bn_json=bn_json,
                parameter_record=parameter_record,
                success_text=success_text
            )

            if checked_parameter.get("should_store_as_risky", False):
                risky_parameters.append(checked_parameter)

        # Step 5.3: Store only if risky parameters exist
        if risky_parameters:
            failure_record = {
                "timestamp": str(datetime.now()),
                "failure_scenario_index": int(failure_index),
                "failure_scenario_text": failure_scenario_text,
                "identified_cpt_parameters": risky_parameters
            }

            append_risky_failure_record(failure_record)
            all_failure_records.append(failure_record)

            processed_failures += 1
            
            print(
                f"Processed failure scenario "
                f"{processed_failures}/{total_failures}"
            )

    # Step 6: Final CPT-level danger report
    cpt_danger_report = generate_cpt_danger_report(bn_json)

    print("\nFinal CPT-level danger report generated.")

    return {
        "results": results,
        "failure_count": len(failures),
        "success_count": len(successes),
        "risky_failure_records": all_failure_records,
        "cpt_danger_report": cpt_danger_report
    }

def store_analysis(bn_number, evaluation_output):
    record = {
        "timestamp": str(datetime.now()),
        "bn_number": bn_number,
        "failure_count": evaluation_output["failure_count"],
        "success_count": evaluation_output["success_count"],
        "risky_failure_records": evaluation_output["risky_failure_records"],
        "cpt_danger_report": evaluation_output["cpt_danger_report"],
        "scenario_results": evaluation_output["results"]
    }

    with open("bn_analysis.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\nNew workflow analysis for BN #{bn_number} stored in bn_analysis.json")

if __name__ == "__main__":
    bn_number = 7

    bn_json = find_proposed_bn(
        bn_number,
        proposed_bn_filename
    )

    evaluation_output = run_evaluation(bn_json)

    print(json.dumps(evaluation_output["cpt_danger_report"], indent=2))

    store_analysis(
        bn_number,
        evaluation_output
    )
