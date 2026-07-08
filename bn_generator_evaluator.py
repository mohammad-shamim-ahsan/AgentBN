from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json
import pandas as pd
from datetime import datetime
from collections import deque

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

        probs = sorted(result.values, reverse=True)
        max_prob = probs[0]
        second_prob = probs[1]
        margin = max_prob - second_prob

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
            "Margin": float(margin),
            "Ground Truth": row["Ground Truth"],
            "Posterior": posterior_probs
        })

    return results


###--------Step 3: Getting Analysis Results for LLM---------------------
def format_results_for_llm(results_df, original_df):

    lines = []

    for _, row in results_df.iterrows():

        scenario_id = row["Scenario"]

        original_row = original_df[
            original_df["Scenario #"] == scenario_id
        ].iloc[0].to_dict()

        # remove metadata fields
        original_row.pop("Scenario #", None)
        original_row.pop("Ground Truth", None)

        is_success = (
            row["Prediction"] == row["Ground Truth"]
            and row["Confidence"] >= 0.50
            and row["Margin"] >= 0.20
        )
        status = "SUCCESS" if is_success else "FAILURE"

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

df = pd.read_csv("combined_train_scenarios.csv")
df.columns = df.columns.str.strip()

context = read_file("context_agent.txt")


def clean_text(text):
    return (
        text.replace("\\n", " ")
            .replace("\n", " ")
            .replace("- ", "")
            .strip()
    )


###-----------------------------
def generate_activation_trace_csv(
    bn_json,
    scenarios_df,
    inference_results,
    output_csv="activation_trace.csv"
):
    """
    Deterministically propagates node states (argmax) for every scenario
    and stores the activated state of every node.

    Parameters
    ----------
    bn_json : dict
        Bayesian Network JSON.

    scenarios_df : pd.DataFrame
        Original scenario CSV.

    inference_results : list
        Output of call_bn_inference().

    output_csv : str
        Output activation trace CSV.
    """

    # ----------------------------------------------------
    # Build node lookup
    # ----------------------------------------------------
    node_lookup = {
        node["name"]: node
        for node in bn_json["nodes"]
    }

    # ----------------------------------------------------
    # Topological order
    # ----------------------------------------------------
    indegree = {n["name"]: 0 for n in bn_json["nodes"]}
    children = {n["name"]: [] for n in bn_json["nodes"]}

    for parent, child in bn_json["edges"]:
        children[parent].append(child)
        indegree[child] += 1

    q = deque([n for n in indegree if indegree[n] == 0])

    topo_order = []

    while q:
        node = q.popleft()
        topo_order.append(node)

        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                q.append(child)

    # ----------------------------------------------------
    # Trace every scenario
    # ----------------------------------------------------
    rows = []

    for (_, scenario), inference in zip(
        scenarios_df.iterrows(),
        inference_results
    ):

        activated = {}

        activated["Scenario"] = scenario["Scenario #"]
        activated["Prediction"] = inference["Prediction"]
        activated["Confidence"] = inference["Confidence"]
        activated["Ground Truth"] = inference["Ground Truth"]
        activated["Confidence"] = float(inference["Confidence"])
        activated["Margin"] = float(inference["Margin"])

        activated["Status"] = (
            "SUCCESS"
            if (
                inference["Prediction"] == inference["Ground Truth"]
                and inference["Confidence"] >= 0.50
                and inference["Margin"] >= 0.20
            )
            else "FAILURE"
        )

        # --------------------------------------------
        # Initialize evidence
        # --------------------------------------------
        for col in scenarios_df.columns:

            if col in ["Scenario #", "Ground Truth"]:
                continue

            value = scenario[col]

            if pd.isna(value):
                continue

            activated[col] = value.strip()

        # --------------------------------------------
        # Deterministic forward propagation
        # --------------------------------------------
        for node_name in topo_order:

            if node_name in activated:
                continue

            node = node_lookup[node_name]

            parents = node.get("parents", [])

            parent_state_order = node["cpt"].get(
                "parent_state_order",
                {}
            )

            # Determine CPT column
            column = 0
            multiplier = 1

            for parent in reversed(parents):

                state = activated[parent]

                idx = parent_state_order[parent].index(state)

                column += idx * multiplier

                multiplier *= len(parent_state_order[parent])

            probs = [
                row[column]
                for row in node["cpt"]["values"]
            ]

            best_idx = probs.index(max(probs))

            selected_state = node["states"][best_idx]
            selected_prob = probs[best_idx]

            # Store everything
            activated[node_name] = selected_state
            activated[f"{node_name}_col"] = column
            activated[f"{node_name}_argmax_prob"] = selected_prob

        rows.append(activated)

        activation_df = pd.DataFrame(rows)

        activation_df.to_csv(
            output_csv,
            index=False
        )

    print(
        f"\nActivation trace saved to {output_csv}"
    )

    return activation_df



def generate_failure_parameter_statistics(
    activation_df, bn_number=None
):

    df = activation_df

    failure_df = df[df["Status"] == "FAILURE"]
    success_df = df[df["Status"] == "SUCCESS"]

    results = {}

    # --------------------------------------------------
    # Cross-CPT activation signatures for failure scenarios
    # --------------------------------------------------
    common_failure_columns = {}

    # Every inferred node has a *_col column
    col_fields = [
        c for c in df.columns
        if c.endswith("_col")
    ]

    for col_field in col_fields:

        cpt = col_field[:-4]

        results[cpt] = []

        # --------------------------------------------------
        # Get unique (column, state) pairs for this CPT
        # --------------------------------------------------
        unique_parameters = (
            failure_df[[col_field, cpt]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        for _, parameter in unique_parameters.iterrows():

            column = int(parameter[col_field])
            state = parameter[cpt]

            # ----------------------------------------------
            # Failure occurrences
            # ----------------------------------------------
            failure_mask = (
                (failure_df[col_field] == column)
                &
                (failure_df[cpt] == state)
            )

            failure_rows = failure_df[failure_mask]

            failure_weight = len(failure_rows)

            # ----------------------------------------------
            # Success occurrences
            # ----------------------------------------------
            success_mask = (
                (success_df[col_field] == column)
                &
                (success_df[cpt] == state)
            )

            success_rows = success_df[success_mask]

            success_weight = len(success_rows)

            # Keep only failure-dominant parameters
            if failure_weight <= success_weight:
                continue

            # ----------------------------------------------
            # Record cross-CPT activation signatures
            # ----------------------------------------------
            for scenario in (
                failure_rows["Scenario"]
                .astype(int)
                .tolist()
            ):

                if scenario not in common_failure_columns:
                    common_failure_columns[scenario] = {}

                common_failure_columns[scenario][cpt] = {
                    "column": column,
                    "state": state
                }

            failure_scenarios = (
                failure_rows["Scenario"]
                .astype(int)
                .tolist()
            )

            success_scenarios = (
                success_rows["Scenario"]
                .astype(int)
                .tolist()
            )

            # ----------------------------------------------
            # Argmax probability
            # (same for every occurrence of this parameter)
            # ----------------------------------------------
            argmax_prob = float(
                failure_rows.iloc[0][
                    f"{cpt}_argmax_prob"
                ]
            )

            results[cpt].append({

                "column": column,

                "state": state,

                "argmax_probability": argmax_prob,

                "failure_weight": failure_weight,

                "success_weight": success_weight,

                # "failure_scenarios": failure_scenarios,

                # "success_scenarios": success_scenarios

            })

    results = {
        cpt: params
        for cpt, params in results.items()
        if params
    }

    # ----------------------------------------------
    # Store JSON
    # ----------------------------------------------
    output = {
        "parameter_statistics": results,
        "common_failure_columns": common_failure_columns
    }

    # --------------------------------------------------
    # Discover common activation patterns
    # --------------------------------------------------
    from collections import defaultdict

    pattern_map = defaultdict(list)

    scenarios = sorted(common_failure_columns.keys())

    for i in range(len(scenarios)):
        s1 = scenarios[i]

        for j in range(i + 1, len(scenarios)):
            s2 = scenarios[j]

            common = {}

            for cpt in common_failure_columns[s1]:

                if cpt not in common_failure_columns[s2]:
                    continue

                if (
                    common_failure_columns[s1][cpt]["column"]
                    ==
                    common_failure_columns[s2][cpt]["column"]
                ):

                    common[cpt] = (
                        common_failure_columns[s1][cpt]["column"]
                    )

            if len(common) >= 2:
                key = tuple(sorted(common.items()))
                pattern_map[key].extend([s1, s2])

    # --------------------------------------------------
    # Convert to JSON-friendly format
    # --------------------------------------------------
    common_activation_patterns = []

    for pattern, scenarios in pattern_map.items():

        common_activation_patterns.append({

            "pattern": {
                cpt: column
                for cpt, column in pattern
            },

            "failure_scenarios": sorted(
                list(set(scenarios))
            )

        })

    output = {

        "bn_number": bn_number,

        "parameter_statistics": results,

        # "common_failure_columns": common_failure_columns,

        "common_activation_patterns":
            common_activation_patterns
    }

    with open(
        "failure_parameter_statistics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=4)

    print(
        "Failure parameter statistics saved to "
        "'failure_parameter_statistics.json'"
    )

    return output


###-----------------------------
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


def generate_cpt_danger_report(
    bn_json,
    statistics_json,
    bn_number=None,
    temperature=0.3
):
    
    if (
        bn_number is not None
        and statistics_json.get("bn_number") != bn_number
    ):
        statistics_json = {
            "bn_number": bn_number,
            "parameter_statistics": {},
            "common_failure_columns": {},
            "common_activation_patterns": []
        }

    prompt = f"""
You are analyzing deterministic Bayesian Network (BN) parameter statistics.

The deterministic analysis has already identified candidate failure-associated CPT parameters.

Your task is to determine which, if any, candidate CPTs are genuinely plausible contributors to the observed failure patterns. Base your judgment on the domain context, the Bayesian Network structure, and the provided deterministic statistics.

The objective is to identify every CPT for which there is strong, independent evidence that its probability assignments may contribute to repeated incorrect predictions.

Report only CPTs that are plausible candidates for refinement. Do not attempt to determine the exact subset of CPTs that should ultimately be modified. That decision will be made during the refinement stage.

You are given:

1. Domain context.

{context}

2. The complete Bayesian Network.

{json.dumps(bn_json, indent=2)}

3. Failure parameter statistics.

{json.dumps(statistics_json, indent=2)}

Definitions:

- column:
  Activated CPT column (i.e., the parent-state configuration).

- state:
  Selected child state obtained by deterministic forward propagation using the argmax rule.

- argmax_probability:
  The probability assigned by the activated CPT column to the selected child state.

- failure_weight:
  Number of unique FAILURE scenarios in which the same CPT parameter
  (same CPT, same column, same child state) was activated.

- success_weight:
  Number of unique SUCCESS scenarios in which the same CPT parameter
  (same CPT, same column, same child state) was activated.

Reasoning principles:

1. The reported statistics identify deterministic candidate parameters only. Their presence alone does not necessarily imply that the corresponding CPT requires modification.

2. Evaluate each candidate CPT holistically. Consider the deterministic statistics, common activation patterns, Bayesian Network semantics, domain knowledge, and the CPT's role within the causal reasoning path. No single factor should be treated as sufficient evidence for recommending modification.

3. Evaluate both failure_weight and success_weight together. A parameter's importance is not determined solely by how frequently it is activated, but by whether its activation is disproportionately associated with failure scenarios after considering its occurrence in successful scenarios.

4. Consider the common activation patterns across failure scenarios. Repeated co-occurrence of CPT columns may indicate a common reasoning path within the Bayesian Network; however, recurring activation alone does not imply that every CPT in the pattern requires modification.

5. Use argmax_probability as supporting evidence rather than a standalone criterion. Whether a strongly activated CPT parameter is problematic should be judged together with the domain semantics and the overall causal reasoning path.

6. Recommend modifying a CPT only when the collective evidence consistently indicates that its probability assignments are plausible contributors to repeated incorrect predictions and that modifying the CPT is likely to improve the overall diagnostic behavior of the Bayesian Network while preserving successful reasoning.

7. Report every CPT for which there is strong independent evidence of potential modeling error. If multiple CPTs belong to the same causal reasoning path, distinguish between CPTs whose apparent behavior is fully explained by upstream probability assignments and CPTs whose probability assignments themselves appear independently flawed.

Your task:

Step 1.
Evaluate every candidate CPT by jointly considering:
- the deterministic parameter statistics,
- and the provided domain context.

Step 2.
Determine which candidate CPTs are sufficiently supported as plausible refinement candidates.

Step 3.
If modification is justified, classify each CPT as one of:
- HIGH
- HIGH_MEDIUM
- MEDIUM
- LOW

The assigned risk level should reflect the overall strength of evidence that the CPT independently contributes to repeated incorrect predictions.

Step 4.
Construct the refinement search space using the following priority:
1. Prioritize HIGH-risk CPTs.
2. If one or more HIGH-risk CPTs exist, report all HIGH-risk CPTs together with any HIGH_MEDIUM-risk CPTs.
3. Otherwise, if one or more HIGH_MEDIUM-risk CPTs exist, report all HIGH_MEDIUM-risk CPTs.
4. Otherwise, return "none".

If no CPT is sufficiently supported for refinement, return "none".

Do NOT blindly report all candidate CPTs.

Return ONLY valid JSON.

Output format:

{{
  "reported_risk_level":  "high | high_medium | medium | none",

  "dangerous_cpts": [
    {{
      "cpt": "",

      "risk_level": "high | high_medium | medium | low",

      "number_of_failure_scenarios": 0,

      "main_problem": "Briefly explain why this CPT is considered a plausible contributor to repeated incorrect predictions based on the deterministic statistics, the BN structure, and the domain semantics.",

      "suspicious_parameters": [
        {{
          "rank": 1,

          "parameter": "Column X → ChildState",

          "failure_weight": 0,

          "success_weight": 0,

          "failure_to_success_ratio": 0.0,

          "argmax_probability": 0.0,

          "recommended_adjustment":
              "increase | decrease | slightly_increase | slightly_decrease",

          "justification": ""
        }}
      ]
    }}
  ],

  "overall_summary": ""
}}

If no CPT is selected for the refinement search space, return:

{{
    "reported_risk_level": "none",
    "dangerous_cpts": [],
    "overall_summary": "No CPTs were sufficiently supported for refinement."
}}
"""

    response = llm(prompt, temperature=temperature)

    report = safe_json_loads(response)

    if report is None:
        report = {
            "reported_risk_level": "none",
            "dangerous_cpts": [],
            "overall_summary": "Unable to parse LLM response.",
            "raw_response": str(response)
        }

    with open(
        "dangerous_cpt_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    return report


### -----------------------------
def run_evaluation(bn_json, bn_number=None, temperature=0.3):

    # --------------------------------------------------
    # Step 1: Build BN model
    # --------------------------------------------------
    model = build_model(bn_json)

    # --------------------------------------------------
    # Step 2: Run BN inference
    # --------------------------------------------------
    results = call_bn_inference(model, df)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Step 3: Deterministic activation tracing
    # --------------------------------------------------
    activation_df = generate_activation_trace_csv(
        bn_json=bn_json,
        scenarios_df=df,
        inference_results=results,
        output_csv="activation_trace.csv"
    )

    # --------------------------------------------------
    # Step 4: Deterministic parameter statistics
    # --------------------------------------------------
    parameter_statistics = generate_failure_parameter_statistics(
        activation_df=activation_df, bn_number=bn_number
    )

    # --------------------------------------------------
    # Step 5: LLM danger report
    # --------------------------------------------------
    cpt_danger_report = generate_cpt_danger_report(
        bn_json=bn_json,
        statistics_json=parameter_statistics,
        bn_number=bn_number,
        temperature=temperature
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    failures = results_df[
        (results_df["Prediction"] != results_df["Ground Truth"])
        |
        (
            (results_df["Confidence"] < 0.50)
            |
            (results_df["Margin"] < 0.20)
        )
    ]

    successes = results_df.drop(failures.index)

    accuracy = (
        len(successes) / len(results_df)
        if len(results_df) > 0 else 0
    )

    print("\n===================================================")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    return {
        "failure_count": len(failures),
        "success_count": len(successes),
        "accuracy": accuracy,
        "cpt_danger_report": cpt_danger_report,
    }


def store_analysis(bn_number, evaluation_output):

    record = {
        "timestamp": str(datetime.now()),
        "bn_number": bn_number,
        "failure_count": evaluation_output["failure_count"],
        "success_count": evaluation_output["success_count"],
        "accuracy": evaluation_output["accuracy"],
        "cpt_danger_report":
            evaluation_output["cpt_danger_report"]

    }

    with open(
        "bn_analysis.json",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(json.dumps(record) + "\n")

    print(
        f"\nWorkflow analysis for BN #{bn_number} "
        "stored in bn_analysis.json"
    )


def load_bn(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
### ------------------------------MAIN--------------------------------------
if __name__ == "__main__":
    
    bn_number = 1
    
    bn_json = find_proposed_bn(
        bn_number,
        proposed_bn_filename
    )

    # bn_json = load_bn("flawed_BN_0.json")

    evaluation_output = run_evaluation(bn_json, bn_number=bn_number, temperature=0.3)

    store_analysis(bn_number, evaluation_output)
