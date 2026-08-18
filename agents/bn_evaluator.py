from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import json
import pandas as pd
from datetime import datetime
from collections import deque

from utils.llm import *
from utils.pgmpy_tool import *
from utils.bn_io import *
from utils.file_utils import *
from utils.json_utils import *
from config.settings import *


###--------Step: Getting Analysis Results for LLM---------------------
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

        status = "SUCCESS" if row["Success"] else "FAILURE"

        lines.append(
            f"Scenario {row['Scenario']}:\n"
            f"- Evidence: {original_row}\n"
            f"- Prediction: {row['Prediction']} (confidence={row['Confidence']:.3f})\n"
            f"- Ground Truth: {row['Ground Truth']}\n"
            f"- Status: {status}\n"
        )

    return "\n".join(lines)


def generate_activation_trace_csv(
    bn_json,
    scenarios_df,
    inference_results,
    relevant_nodes,
    path_nodes,
    added_parent_nodes,
    path_list,
    output_csv=ACTIVATION_TRACE_FILE
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

        activated["Status"] = (
            "SUCCESS" if inference["Success"] else "FAILURE"
        )

        # --------------------------------------------
        # Initialize evidence
        # --------------------------------------------
        for col in scenarios_df.columns:

            if col in ["Scenario #", "Ground Truth"]:
                continue

            # Only keep evidence that belongs to the relevant subgraph
            if col not in relevant_nodes:
                continue

            value = scenario[col]

            if pd.isna(value):
                continue

            activated[col] = value.strip()

        
        # Ground-truth value of the target node
        activated[TARGET_NODE] = inference["Ground Truth"]

        # --------------------------------------------
        # Deterministic forward propagation
        # --------------------------------------------
        for node_name in topo_order:

            # Skip nodes that cannot influence the target
            if node_name not in relevant_nodes:
                continue

            node = node_lookup[node_name]
            parents = node.get("parents", [])

            # --------------------------------------------------
            # Determine active CPT column
            # --------------------------------------------------
            if len(parents) == 0:

                # Root prior
                column = 0

            else:

                parent_state_order = node["cpt"]["parent_state_order"]

                column = 0
                multiplier = 1

                for parent in reversed(parents):

                    if parent not in activated:
                        raise ValueError(
                            f"Parent '{parent}' of '{node_name}' has no assigned state."
                        )

                    state = activated[parent]

                    idx = parent_state_order[parent].index(state)

                    column += idx * multiplier

                    multiplier *= len(parent_state_order[parent])

            probs = [
                row[column]
                for row in node["cpt"]["values"]
            ]

            # --------------------------------------------------
            # Node state already known
            # (observed evidence or ground-truth target)
            # --------------------------------------------------
            if node_name in activated:

                state = activated[node_name]

                state_idx = node["states"].index(state)

                activated[f"{node_name}_col"] = column
                activated[f"{node_name}_selected_prob"] = probs[state_idx]

                continue

            # --------------------------------------------------
            # Hidden node:
            # deterministic forward propagation
            # --------------------------------------------------
            best_idx = probs.index(max(probs))

            activated[node_name] = node["states"][best_idx]
            activated[f"{node_name}_col"] = column
            activated[f"{node_name}_selected_prob"] = probs[best_idx]

        
        # rows.append(activated)

        output = {}

        # Always keep metadata
        for key in [
            "Scenario",
            "Prediction",
            "Confidence",
            "Ground Truth",
            "Status",
        ]:
            output[key] = activated[key]

        # Keep only path nodes
        for node in path_nodes:

            if node in activated:
                output[node] = activated[node]

            col_key = f"{node}_col"
            prob_key = f"{node}_selected_prob"

            if col_key in activated:
                output[col_key] = activated[col_key]

            if prob_key in activated:
                output[prob_key] = activated[prob_key]

        rows.append(output)

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
    activation_df, bn_number=None, exclude_evidence_nodes=EXCLUDE_EVIDENCE_NODES
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
        and (
            not exclude_evidence_nodes
            or c[:-4] not in EVIDENCE_NODES
        )
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

            # # Keep only failure-dominant parameters
            # if failure_weight <= success_weight:
            #     continue

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

            # failure_scenarios = (
            #     failure_rows["Scenario"]
            #     .astype(int)
            #     .tolist()
            # )

            # success_scenarios = (
            #     success_rows["Scenario"]
            #     .astype(int)
            #     .tolist()
            # )

            # ----------------------------------------------
            # Argmax probability
            # (same for every occurrence of this parameter)
            # ----------------------------------------------
            argmax_prob = float(
                failure_rows.iloc[0][
                    f"{cpt}_selected_prob"
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

        # "common_activation_patterns":
        #     common_activation_patterns
    }

    with open(
        FAILURE_PARAMETER_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=4)

    print(
        "Failure parameter statistics saved to "
        f"'{FAILURE_PARAMETER_FILE}'"
    )

    return output


DANGER_REPORT_SCHEMA = """
{
  "reported_risk_level": "high | high_medium | medium | none",

  "dangerous_cpts": [
    {
      "cpt": "",

      "risk_level": "high | high_medium | medium | low",

      "number_of_failure_scenarios": 0,

      "main_problem": "Brief explanation in 1-2 sentences."

      "suspicious_parameters": [
        {
          "rank": 1,

          "parameter": "Column X → ChildState",

          "failure_weight": 0,

          "success_weight": 0,

          "failure_to_success_ratio": 0.0,

          "argmax_probability": 0.0,

          "recommended_adjustment":
              "increase | decrease | slightly_increase | slightly_decrease",

          "justification": "Brief justification in 1-2 sentences."
        }
      ]
    }
  ],

  "overall_summary": "Brief overall summary in 1-2 sentences."
}
"""


def validate_danger_report(response):
    report = json.loads(response)

    required_fields = [
        "reported_risk_level",
        "dangerous_cpts",
        "overall_summary"
    ]

    for field in required_fields:
        if field not in report:
            raise ValueError(
                f"Missing required field '{field}'."
            )

    return report


def validate_diagnostic_report(response):
    report = json.loads(response)

    required_fields = [
        "path_diagnostics",
        "overall_summary"
    ]

    for field in required_fields:
        if field not in report:
            raise ValueError(
                f"Missing required field '{field}'."
            )

    return report


# def generate_cpt_danger_report(
#     bn_json,
#     statistics_json,
#     paths,
#     bn_number=None,
#     temperature=0.3,
#     oracle = None
# ):

#     # ==================================================
#     # PREPARATION
#     # ==================================================
    
#     if (
#         bn_number is not None
#         and statistics_json.get("bn_number") != bn_number
#     ):
#         statistics_json = {
#             "bn_number": bn_number,
#             "parameter_statistics": {},
#             # "common_failure_columns": {},
#             # "common_activation_patterns": []
#         }

#     context = read_file(CONTEXT_AGENT_FILE)

#     prompt = f"""
# You are analyzing deterministic Bayesian Network (BN) parameter statistics to identify CPTs that are plausible contributors to repeated incorrect predictions.

# You are given:

# 1. Domain context.

# {context}

# 2. The complete Bayesian Network.

# {json.dumps(bn_json, indent=2)}

# 3. Failure parameter statistics.

# {json.dumps(statistics_json, indent=2)}

# 4. Oracle-provided relevant nodes.

# {oracle}

# Oracle constraint:

# - ONLY CPTs corresponding to oracle-identified nodes may be considered as candidates.
# - The oracle defines candidate eligibility; it does not imply that an eligible CPT is problematic.
# - Use the complete BN to understand dependencies, semantics, and reasoning paths among the relevant nodes, but do not introduce candidates outside the oracle.

# Definitions:

# - column: Activated CPT column (parent-state configuration).
# - state: Selected child state obtained by deterministic forward propagation using the argmax rule.
# - argmax_probability: Probability assigned to the selected child state.
# - failure_weight: Number of unique FAILURE scenarios activating the same CPT parameter (CPT, column, state).
# - success_weight: Number of unique SUCCESS scenarios activating the same CPT parameter.

# Reasoning principles:

# 1. Restrict candidate analysis to {oracle}-identified CPTs.

# 2. Treat the deterministic {statistics_json} as evidence of association, not proof that a CPT requires modification.

# 3. Evaluate each candidate CPT holistically using its parameter statistics, BN structure, domain semantics, and role in the reasoning path.

# 4. Consider "failure_weight" and "success_weight" fields jointly, located in the {statistics_json} file. Give greater concern to parameters disproportionately associated with failures rather than those frequently activated in both failures and successes.

# 5. Use "argmax_probability" as supporting evidence, not as a standalone criterion.

# 6. Distinguish CPTs whose behavior may primarily reflect upstream activations from CPTs whose own probability assignments plausibly contribute to the failures.

# 7. Report CPTs as refinement candidates only when the available evidence reasonably supports their contribution to repeated incorrect predictions.

# Your task:

# 1. Evaluate every oracle-eligible candidate CPT.

# 2. Assign each supported candidate one risk level:
#    - HIGH
#    - HIGH_MEDIUM
#    - MEDIUM
#    - LOW

#    The risk level should reflect the strength of evidence that the CPT's own probability assignments contribute to repeated incorrect predictions.

# 3. Construct the refinement search space using this priority:
#    - If HIGH-risk CPTs exist, report all HIGH-risk CPTs.
#    - Otherwise, if HIGH_MEDIUM-risk CPTs exist, report all HIGH_MEDIUM-risk CPTs.
#    - Otherwise, report the strongest MEDIUM-risk CPTs.
#    - Return "none" if no CPT is sufficiently supported.

# If no CPT is sufficiently supported for refinement, return "none".

# Return ONLY valid JSON.

# Output format:

# {{
#   "reported_risk_level":  "high | high_medium | medium | none",

#   "dangerous_cpts": [
#     {{
#       "cpt": "",

#       "risk_level": "high | high_medium | medium | low",

#       "number_of_failure_scenarios": 0,

#       "main_problem": "Briefly explain why this CPT is considered a plausible contributor to repeated incorrect predictions based on the deterministic statistics, the BN structure, and the domain semantics.",

#       "suspicious_parameters": [
#         {{
#           "rank": 1,

#           "parameter": "Column X → ChildState",

#           "failure_weight": 0,

#           "success_weight": 0,

#           "failure_to_success_ratio": 0.0,

#           "argmax_probability": 0.0,

#           "recommended_adjustment":
#               "increase | decrease | slightly_increase | slightly_decrease",

#           "justification": ""
#         }}
#       ]
#     }}
#   ],

#   "overall_summary": ""
# }}

# If no CPT is selected for the refinement search space, return:

# {{
#     "reported_risk_level": "none",
#     "dangerous_cpts": [],
#     "overall_summary": "No individual CPT could be confidently localized as a refinement candidate based on the available evidence."
# }}
# """

#     response = llm(prompt, temperature=temperature)

#     last_response = None
#     last_error = None

#     # ==================================================
#     # STAGE 1
#     # Retry the original evaluation prompt
#     # ==================================================
#     for attempt in range(1, MAX_FORMAT_RETRIES + 1):

#         if attempt > 1:
#             response = llm(
#                 prompt,
#                 temperature=temperature
#             )

#         last_response = response

#         try:

#             report = validate_danger_report(
#                 response
#             )

#             print(
#                 f"Valid danger report received on attempt {attempt}."
#             )

#             with open(
#                 DANGER_REPORT_FILE,
#                 "w",
#                 encoding="utf-8"
#             ) as f:
#                 json.dump(report, f, indent=4)

#             return report

#         except (json.JSONDecodeError, ValueError) as error:

#             last_error = error

#             print(
#                 f"[Format Retry "
#                 f"{attempt}/{MAX_FORMAT_RETRIES}] "
#                 f"{error}"
#             )

#     # ==================================================
#     # STAGE 2
#     # Repair the formatting of the final invalid response
#     # ==================================================
#     for repair_attempt in range(1, MAX_REPAIR_RETRIES + 1):

#         repair_prompt = f"""
# Your previous response was generated for a Bayesian Network evaluation task.

# The response violates the required JSON format.

# Validation error:
# {last_error}

# Previous response:
# {last_response}

# Your task is ONLY to repair the JSON formatting.

# Do NOT:
# - modify any reported risk level;
# - modify any CPT;
# - modify any suspicious parameter;
# - modify any probability;
# - modify any recommendation;
# - modify any explanation;
# - add or remove any field;
# - change the Bayesian Network evaluation content in any way.

# Repair only the JSON syntax and structure so that the response conforms to the required output format.

# The required output format is:

# {DANGER_REPORT_SCHEMA}

# Return ONLY the repaired JSON.
#     """

#         repaired_response = llm(
#             repair_prompt,
#             temperature=0.0
#         )

#         last_response = repaired_response

#         try:

#             report = validate_danger_report(
#                 repaired_response
#             )

#             print(
#                 f"Response successfully repaired "
#                 f"on attempt {repair_attempt}."
#             )

#             with open(
#                 DANGER_REPORT_FILE,
#                 "w",
#                 encoding="utf-8"
#             ) as f:
#                 json.dump(report, f, indent=4)

#             return report

#         except (json.JSONDecodeError, ValueError) as error:

#             last_error = error

#             print(
#                 f"[Repair Retry "
#                 f"{repair_attempt}/{MAX_REPAIR_RETRIES}] "
#                 f"{error}"
#             )

#     # ==================================================
#     # Final fallback
#     # ==================================================
#     report = {
#         "reported_risk_level": "none",
#         "dangerous_cpts": [],
#         "overall_summary": (
#             "Unable to obtain a valid JSON response after "
#             "format verification and repair."
#         ),
#         "raw_response": str(last_response)
#     }

#     with open(
#         DANGER_REPORT_FILE,
#         "w",
#         encoding="utf-8"
#     ) as f:
#         json.dump(report, f, indent=4)

#     return report


def run_llm_with_retry_and_repair(
    prompt,
    validator,
    schema,
    temperature=0.3
):
    """
    Executes an LLM prompt with:
    1. validation,
    2. regeneration retries,
    3. formatting-only repair retries.
    """

    last_response = None
    last_error = None

    # ==================================================
    # STAGE 1
    # Generate / retry original response
    # ==================================================
    for attempt in range(1, MAX_FORMAT_RETRIES + 1):

        response = llm(
            prompt,
            temperature=temperature
        )

        last_response = response

        try:
            report = validator(response)

            print(
                f"Valid response received on attempt {attempt}."
            )

            return report

        except (json.JSONDecodeError, ValueError) as error:
            last_error = error

            print(
                f"[Format Retry "
                f"{attempt}/{MAX_FORMAT_RETRIES}] "
                f"{error}"
            )


    # ==================================================
    # STAGE 2
    # Repair the formatting of the final invalid response
    # ==================================================
    response_to_repair = last_response

    for repair_attempt in range(1, MAX_REPAIR_RETRIES + 1):

        repair_prompt = f"""
Your previous response violates the required JSON format.

Validation error:
{last_error}

Previous response:
{response_to_repair}

Your task is ONLY to repair the JSON syntax and structure.

Preserve all substantive content exactly.

Do NOT:
- change any assessment or risk level;
- change any CPT, parameter, column, or state;
- change any probability or statistic;
- change any recommendation;
- change any justification, explanation, or summary;
- add or remove any substantive field or entry;
- perform the evaluation task again.

Repair only the JSON syntax and structure so that the response conforms to the required output format.

Required output format:

{schema}

Return ONLY the repaired JSON.
"""

        repaired_response = llm(
            repair_prompt,
            temperature=0.0
        )

        try:
            report = validator(repaired_response)

            print(
                f"Response successfully repaired "
                f"on attempt {repair_attempt}."
            )

            return report

        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            last_response = repaired_response

            print(
                f"[Repair Retry "
                f"{repair_attempt}/{MAX_REPAIR_RETRIES}] "
                f"{error}"
            )

    # ==================================================
    # Failure
    # ==================================================
    failure_record = {
        "error": str(last_error),
        "original_response": str(response_to_repair),
        "last_repair_response": str(last_response)
    }

    with open(
        FAILED_LLM_RESPONSE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(failure_record, f, indent=4)

    raise ValueError(
        "Unable to obtain a valid LLM response after "
        f"generation and repair attempts. Last error: {last_error}"
    )


DIAGNOSTIC_REPORT_SCHEMA = """
{
  "path_diagnostics": [
    {
      "path_id": "P1",

      "path": [
        "NodeA",
        "NodeB",
        "NodeC"
      ],

      "path_assessment": "suspicious | uncertain | consistent",

      "parameter_assessments": [
        {
          "cpt": "",

          "column": 0,

          "state": "",

          "argmax_probability": 0.0,

          "failure_weight": 0,

          "success_weight": 0,

          "assessment": "suspicious | uncertain | consistent",

          "justification": "Brief justification in 1-2 sentences."
        }
      ],

      "path_summary": "Brief path-level summary in 1-2 sentences."
    }
  ],

  "overall_summary": "Brief overall summary in 1-2 sentences."
}
"""

def generate_cpt_diagnostic_report(
    bn_json,
    statistics_json,
    paths,
    context,
    temperature=0.3
):

    prompt = f"""
You are performing path-wise diagnosis of Bayesian Network (BN) parameters associated with repeated incorrect predictions.

Your role is DIAGNOSIS AND LOCALIZATION ONLY.

You must identify activated CPT parameters that are plausible contributors to the observed inference failures.

You must NOT:
- decide the final refinement search space;
- recommend parameter adjustments;
- prioritize one suspicious CPT over another for refinement;
- propose probability changes.

You are given:

1. Domain context.

{context}

2. The complete Bayesian Network.

{json.dumps(bn_json, indent=2)}

3. Deterministic parameter statistics.

{json.dumps(statistics_json, indent=2)}

4. Directed target-evidence paths.

{json.dumps(paths, indent=2)}


Definitions:

- path: A directed path in the BN connecting an evidence node and the target node.

- column: An activated CPT column corresponding to a particular parent-state configuration.

- state: The child state associated with the parameter.

- argmax_probability: The probability assigned to the selected child state.

- failure_weight: Number of unique FAILURE scenarios activating the same CPT parameter (CPT, column, state).

- success_weight: Number of unique SUCCESS scenarios activating the same CPT parameter.


Path-wise analysis procedure:

For EACH provided path:

1. Examine the CPTs corresponding to nodes on that path.

2. Restrict diagnostic candidates to nodes that are present on the current path. Nodes outside the current path may be used only as contextual information for interpreting CPT columns and BN dependencies.

3. For each eligible CPT, examine every parameter entry provided in the deterministic parameter statistics.

4. Use the complete BN structure to interpret the meaning (not values) of the CPT column , including the corresponding parent-state configuration.

5. Evaluate whether each parameter is a plausible contributor to the repeated incorrect predictions using THREE sources of evidence:

   A. Local semantic evidence
      - Determine whether the probability assigned to the child state is reasonable given the activated parent-state configuration and domain semantics.
      - Use argmax_probability as supporting evidence, not as a standalone criterion.

   B. Path-level evidence
      - Consider whether the parameter's behavior is consistent with the expected relationships among variables along the complete path.
      - Distinguish a parameter whose own probability assignment appears problematic from one whose activation may merely reflect upstream behavior.

   C. Statistical evidence
      - Consider failure_weight and success_weight jointly.
      - Greater concern is warranted when a parameter is disproportionately associated with failures.
      - Frequent activation in both failures and successes weakens evidence that the parameter itself is faulty.
      - Treat these statistics as association, not proof.

6. Classify the overall path as:
   - suspicious: one or more parameters have reasonable evidence of contributing to the incorrect inference;
   - uncertain: the available evidence does not support a clear conclusion;
   - consistent: the parameters on the path appear reasonable and consistent with the domain semantics and available evidence.

7. Report parameters according to the overall path assessment:
   - For a suspicious path, report parameters assessed as suspicious or uncertain.
   - For an uncertain or consistent path, return an empty "parameter_assessments" list.

8. Keep "path_summary" brief for uncertain and consistent paths.

Do not report parameters assessed as consistent.

   
Return ONLY valid JSON.

Required output format:

{DIAGNOSTIC_REPORT_SCHEMA}
"""

    return run_llm_with_retry_and_repair(
        prompt=prompt,
        validator=validate_diagnostic_report,
        schema=DIAGNOSTIC_REPORT_SCHEMA,
        temperature=temperature
    )


def generate_cpt_refinement_recommendation(
    bn_json,
    statistics_json,
    diagnostic_report,
    context,
    temperature=0.3
):

    prompt = f"""
You are performing refinement recommendation for a Bayesian Network (BN) after a separate path-wise diagnostic analysis has already been completed.

Your role is REFINEMENT RECOMMENDATION.

The diagnostic agent has already evaluated CPT parameters within individual diagnostic paths. It reports only parameters assessed as suspicious, while each path is classified as suspicious, uncertain, or consistent.

Your task is to determine which of the diagnosed CPT parameters should actually be recommended for inclusion in the refinement search space.

You are given:

1. Domain context.

{context}

2. The complete Bayesian Network.

{json.dumps(bn_json, indent=2)}

3. Deterministic parameter statistics.

{json.dumps(statistics_json, indent=2)}

4. Path-wise diagnostic report produced by the diagnostic agent.

{json.dumps(diagnostic_report, indent=2)}


Definitions:

- column: An activated CPT column corresponding to a particular parent-state configuration.

- state: The child state associated with the parameter.

- argmax_probability: The probability assigned to the selected child state.

- failure_weight: Number of unique FAILURE scenarios activating the same CPT parameter (CPT, column, state).

- success_weight: Number of unique SUCCESS scenarios activating the same CPT parameter.

- path assessment: The overall diagnostic assessment of a path as suspicious, uncertain, or consistent.

- suspicious parameter: A CPT parameter reported by the diagnostic agent because there is positive evidence that its own probability assignment plausibly contributes to the observed inference failures.

Refinement recommendation procedure:

1. Consider parameters reported on suspicious paths as refinement candidates. Treat parameters assessed as suspicious as primary candidates and parameters assessed as uncertain as secondary candidates.

2. Evaluate all candidates jointly across paths using the diagnostic evidence, failure_weight, success_weight, BN structure, and domain semantics. Do not automatically recommend a primary or secondary candidate. Repeated suspicion and disproportionate association with failures strengthen the case for refinement, while uncertainty and substantial association with successful scenarios weaken it.

3. Use the complete BN, parameter statistics, and domain semantics to distinguish parameters whose own probability assignments plausibly contribute to the failures from those whose behavior may primarily reflect upstream effects.

4. Recommend the smallest well-supported set of parameters that can plausibly address the observed failures while preserving successful inference behavior.

5. For each selected parameter, determine the direction of refinement based on the BN semantics, CPT configuration, path-wise diagnostic evidence, and observed failure behavior:
   - increase
   - decrease
   - slightly_increase
   - slightly_decrease

6. Group selected parameters by CPT and assign each selected CPT one risk level:
   - HIGH
   - HIGH_MEDIUM
   - MEDIUM
   - LOW

The risk level represents the strength of evidence that modifying parameters in that CPT is appropriate for correcting the observed inference failures while minimizing disruption to successful inference behavior.

Construct the refinement search space using this priority:

- If HIGH-risk CPTs exist, report all HIGH-risk CPTs.
- Otherwise, if HIGH_MEDIUM-risk CPTs exist, report all HIGH_MEDIUM-risk CPTs.
- Otherwise, report the strongest MEDIUM-risk CPTs.
- Return "none" if no CPT is sufficiently supported for refinement.

Return ONLY valid JSON.

Required output format:

{DANGER_REPORT_SCHEMA}

If no CPT is sufficiently supported for refinement, return:

{{
  "reported_risk_level": "none",
  "dangerous_cpts": [],
  "overall_summary": "No individual CPT could be sufficiently supported as a refinement candidate based on the combined diagnostic, statistical, structural, and semantic evidence."
}}
"""

    response = run_llm_with_retry_and_repair(
        prompt=prompt,
        validator=validate_danger_report,
        schema=DANGER_REPORT_SCHEMA,
        temperature=temperature
    )

    return response


def get_diagnostic_paths(paths, statistics_json):
    """
    Removes nodes that have no parameter statistics and
    collapses duplicate diagnostic paths.
    """

    parameter_nodes = set(
        statistics_json.get("parameter_statistics", {}).keys()
    )

    unique_paths = []
    seen = set()

    for path in paths:

        diagnostic_path = [
            node for node in path
            if node in parameter_nodes
        ]

        if not diagnostic_path:
            continue

        path_key = tuple(diagnostic_path)

        if path_key not in seen:
            seen.add(path_key)
            unique_paths.append(diagnostic_path)

    return unique_paths


def generate_cpt_danger_report(
    bn_json,
    statistics_json,
    paths,
    bn_number=None,
    temperature=0.3
):
    """
    Orchestrates the two-stage CPT diagnosis and refinement
    recommendation process.
    """

    # --------------------------------------------------
    # Basic preparation
    # --------------------------------------------------
    if (
        bn_number is not None
        and statistics_json.get("bn_number") != bn_number
    ):
        statistics_json = {
            "bn_number": bn_number,
            "parameter_statistics": {}
        }

    context = read_file(CONTEXT_AGENT_FILE)

    diagnostic_paths = get_diagnostic_paths(
        paths,
        statistics_json
    )

    # ==================================================
    # AGENT 1
    # Path-wise CPT diagnosis
    # ==================================================
    diagnostic_report = generate_cpt_diagnostic_report(
        bn_json=bn_json,
        statistics_json=statistics_json,
        paths=diagnostic_paths,
        context=context,
        temperature=temperature
    )

    # --------------------------------------------------
    # Save diagnostic report for monitoring
    # --------------------------------------------------
    with open(
        DIAGNOSTIC_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(diagnostic_report, f, indent=4)

    # ==================================================
    # AGENT 2
    # Refinement recommendation
    # ==================================================
    danger_report = generate_cpt_refinement_recommendation(
        bn_json=bn_json,
        statistics_json=statistics_json,
        diagnostic_report=diagnostic_report,
        context=context,
        temperature=temperature
    )

    # --------------------------------------------------
    # Save final danger report
    # --------------------------------------------------
    with open(
        DANGER_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(danger_report, f, indent=4)

    return danger_report


### -----------------------------

def run_evaluation(bn_json, bn_number=None, temperature=0.3, dataset_file=None):

    if dataset_file is None:
        dataset_file = TRAIN_CSV
    
    df = pd.read_csv(dataset_file)
    
    # --------------------------------------------------
    # Step 1: Build BN model
    # --------------------------------------------------
    model = build_model(bn_json)

    # --------------------------------------------------
    # Compute relevant nodes once
    # --------------------------------------------------
    evidence_nodes = [
        col for col in df.columns
        if col not in ["Scenario #", "Ground Truth"]
    ]

    relevant_nodes, path_nodes, added_parent_nodes, path_list = get_target_evidence_paths(
        model,
        evidence_nodes
    )

    with open(ORACLE_FILE, "w") as f:
        json.dump(list(relevant_nodes), f, indent=2)

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
        relevant_nodes=relevant_nodes,
        path_nodes = path_nodes, 
        added_parent_nodes = added_parent_nodes,
        path_list = path_list,
        output_csv=ACTIVATION_TRACE_FILE
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
        paths=path_list,
        bn_number=bn_number,
        temperature=temperature
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    successes = results_df[results_df["Success"]]
    failures = results_df[~results_df["Success"]]

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
        BN_ANALYSIS_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(json.dumps(record) + "\n")

    print(
        f"\nWorkflow analysis for BN #{bn_number} "
        f"stored in {BN_ANALYSIS_FILE}"
    )


### ------------------------------MAIN--------------------------------------

if __name__ == "__main__":

    bn_number = 1
    bn_json = find_proposed_bn(
        bn_number,
        PROPOSED_BN_FILE
    )

    # bn_json = load_bn(FLAWED_BN_FILE)

    evaluation_output = run_evaluation(bn_json, bn_number=bn_number, temperature=0.3)

    store_analysis(bn_number, evaluation_output)
