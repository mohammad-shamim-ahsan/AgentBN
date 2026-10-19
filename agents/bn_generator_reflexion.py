from langchain_core.prompts import PromptTemplate

import json
import copy
import math

from utils.bn_io import *
from utils.json_utils import *
from utils.file_utils import *
from config.settings import *
from utils.llm import *
from evaluation.automatic_bn_reasoning_old import run_evaluation as initial_run_evaluation

# -----------------------------------------------

OUTPUT_SCHEMA = """
{
  "candidates": [
    {
      "candidate_id": 1,
      "modified_cpts": [
        {
          "name": "...",
          "cpt": {
            "parent_state_order": {},
            "values": [
              [...],
              [...]
            ]
          }
        }
      ]
    },
    {
      "candidate_id": 2,
      "modified_cpts": [
        ...
      ]
    },
    {
      "candidate_id": 3,
      "modified_cpts": [
        ...
      ]
    }
  ]
}
"""


def validate_refinement_response(response, num_cpts_to_refine=NUM_CPTS_TO_REFINE):
    """Validate the generated CPT patches."""
    response_json = json.loads(response)

    candidates = response_json["candidates"]

    if len(candidates) != 3:
        raise ValueError("Expected exactly 3 candidates.")

    for candidate in candidates:
        modified = candidate["modified_cpts"]

        if len(modified) != num_cpts_to_refine:
            raise ValueError(f"Expected exactly {num_cpts_to_refine} CPT patches.")

        names = [item["name"] for item in modified]

        if len(names) != len(set(names)):
            raise ValueError("Duplicate CPT names.")

        for item in modified:
            values = item["cpt"]["values"]

            for row in values:
                for p in row:
                    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                        raise ValueError(f"Invalid probability in {item['name']}")

            for column in zip(*values):
                if not math.isclose(sum(column), 1.0, abs_tol=1e-6, rel_tol=0):
                    raise ValueError(f"Invalid CPT column in {item['name']}")

    return response_json


def build_return_object(best_bn_number, baseline_bn, response_json):
    return {
        "best_bn_number": best_bn_number,
        "baseline_bn": baseline_bn,
        "candidates": response_json["candidates"]
    }


def format_analysis_record(record):
    cpt_report = record.get("cpt_danger_report", {})

    if isinstance(cpt_report, str):
        try:
            cpt_report = json.loads(cpt_report)
        except Exception:
            cpt_report = {}

    return f"""
BN NUMBER:
{record.get("bn_number", "unknown")}

FAILURE COUNT:
{record.get("failure_count", "unknown")}

SUCCESS COUNT:
{record.get("success_count", "unknown")}

ACCURACY:
{record.get("accuracy", "unknown")}

FAILURE SCENARIOS:
{record.get("failure_scenarios_text", "No failure scenarios available.")}

SUCCESS SCENARIOS:
{record.get("success_scenarios_text", "No success scenarios available.")}

CPT DANGER REPORT:
{json.dumps(cpt_report, indent=2)}
"""


def get_baseline_modified_cpts(original_bn, baseline_bn):
    original_lookup = {
        node["name"]: node["cpt"]["values"]
        for node in original_bn["nodes"]
    }

    return [
        node["name"]
        for node in baseline_bn["nodes"]
        if node["cpt"]["values"] != original_lookup[node["name"]]
    ]


def integrate_modified_cpts(original_bn, patch):
    refined_bn = copy.deepcopy(original_bn)

    node_lookup = {
        node["name"]: node
        for node in refined_bn["nodes"]
    }

    for item in patch.get("modified_cpts", []):
        name = item["name"]

        if name not in node_lookup:
            raise ValueError(f"Modified CPT node not found in BN: {name}")

        node_lookup[name]["cpt"] = copy.deepcopy(item["cpt"])

    return refined_bn


def generate_refined_cpt_patch(
    full_context,
    proposed_bn_filename=PROPOSED_BN_FILE,
    bn_analysis_filename=BN_ANALYSIS_FILE,
    temperature=0.3,
    oracle=None
):
    
    # --------------------------------------------------
    # Deterministically Select the baseline BN
    # --------------------------------------------------
    best_bn_number = get_best_bn_number(
        bn_analysis_filename
    )

    baseline_bn = find_proposed_bn(
        best_bn_number,
        proposed_bn_filename
    )

    with open(FLAWED_BN_FILE, "r") as f:
        original_bn = json.load(f)

    baseline_modified_cpts = get_baseline_modified_cpts(
        original_bn,
        baseline_bn
    )

    print("Baseline modified CPTs:", baseline_modified_cpts)
    print("Baseline modified CPT count:", len(baseline_modified_cpts))

    analysis_record = get_analysis_record(
        best_bn_number,
        bn_analysis_filename
    )

    formatted_analysis_record = format_analysis_record(
        analysis_record
    )

    # --------------------------------------------------
    # Build refinement prompt
    # --------------------------------------------------
    prompt_gen_template = PromptTemplate(
        input_variables=[
            "full_context",
            "baseline_bn",
            "analysis_record",
            "oracle",
            "target_node",
            "min_confidence",
            "min_margin",
            "num_cpts_to_refine",
            "baseline_modified_cpts",
        ],
        template=read_file(REF_PROMPT_FILE)
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        baseline_bn=json.dumps(
            baseline_bn,
            indent=2
        ),
        analysis_record=formatted_analysis_record,
        oracle=json.dumps(oracle, indent=2),
        target_node=TARGET_NODE,
        min_confidence=MIN_CONFIDENCE,
        min_margin=MIN_MARGIN,
        num_cpts_to_refine=NUM_CPTS_TO_REFINE,
        baseline_modified_cpts=json.dumps(baseline_modified_cpts),
    )

    last_response = None
    last_error = None

    # ==================================================
    # STAGE 1
    # Retry the original refinement prompt
    # ==================================================
    for attempt in range(1, MAX_FORMAT_RETRIES + 1):

        response = llm(
            prompt,
            temperature=temperature
        )

        last_response = response

        try:
            response_json = validate_refinement_response(
                response,
                num_cpts_to_refine=NUM_CPTS_TO_REFINE
            )

            print(
                f"Valid response received on attempt {attempt}."
            )

            return build_return_object(
                best_bn_number,
                baseline_bn,
                response_json
            )

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
    for repair_attempt in range(1, MAX_REPAIR_RETRIES + 1):

        repair_prompt = f"""
Your previous response was generated for a Bayesian Network refinement task.

The response violates the required JSON format, CPT-count constraint, or CPT validity requirements.

Validation error:
{last_error}

Previous response:
{last_response}

Your task is to repair the response while ensuring:
- Exactly 3 independent refinement candidates.
- Each candidate's modified_cpts array must contain exactly
  {NUM_CPTS_TO_REFINE} distinct CPTs representing its complete
  global refinement set.

Preserve valid CPT values and candidate IDs wherever possible.
Do not change the BN structure, nodes, edges, states, or parent lists.

The required output format is:

{OUTPUT_SCHEMA}

Return ONLY the repaired valid JSON.
"""

        repaired_response = llm(
            repair_prompt,
            temperature=0.0
        )

        last_response = repaired_response

        try:

            response_json = validate_refinement_response(
                repaired_response,
                num_cpts_to_refine=NUM_CPTS_TO_REFINE
            )

            print(
                f"Response successfully repaired "
                f"on attempt {repair_attempt}."
            )

            return build_return_object(
                best_bn_number,
                baseline_bn,
                response_json
            )

        except (json.JSONDecodeError, ValueError) as error:

            last_error = error

            print(
                f"[Repair Retry "
                f"{repair_attempt}/{MAX_REPAIR_RETRIES}] "
                f"{error}"
            )

    # ==================================================
    # Failure
    # ==================================================
    raise RuntimeError(
        "The LLM failed to produce a valid refinement response after "
        f"{MAX_FORMAT_RETRIES} generation attempt(s) and "
        f"{MAX_REPAIR_RETRIES} repair attempt(s). "
        f"Last validation error: {last_error}"
    )


def generate_and_select_best_candidate(
    full_context,
    proposed_bn_filename=PROPOSED_BN_FILE,
    bn_analysis_filename=BN_ANALYSIS_FILE,
    train_csv=TRAIN_CSV,
    temperature=0.3
):

    with open(ORACLE_FILE, "r") as f:
        relevant_nodes = list(json.load(f))
    
    refinement_output = generate_refined_cpt_patch(
        full_context=full_context,
        proposed_bn_filename=proposed_bn_filename,
        bn_analysis_filename=bn_analysis_filename,
        temperature=temperature,
        oracle= relevant_nodes
    )

    best_bn_number = refinement_output["best_bn_number"]
    baseline_bn = refinement_output["baseline_bn"]
    candidates = refinement_output["candidates"]

    print(f"Selected baseline BN: {best_bn_number}")

    candidate_results = []

    with open(FLAWED_BN_FILE, "r") as f:
        original_bn = json.load(f)

    for candidate in candidates:
        refined_bn = integrate_modified_cpts(
            original_bn,
            candidate
        )

        failures, successes, accuracy, results = (
            initial_run_evaluation(
                refined_bn,
                train_csv
            )
        )

        candidate_results.append({
            "candidate_id": candidate["candidate_id"],
            "patch": candidate,
            "bn": refined_bn,
            "accuracy": accuracy,
            "failure_count": len(failures),
            "success_count": len(successes),
            "raw_results": results
        })

        print(
            f"\nCandidate {candidate['candidate_id']}: "
            f"Accuracy = {accuracy:.4f}, "
            f"Failures = {len(failures)}, "
            f"Successes = {len(successes)}"
        )

    best_candidate = min(
        candidate_results,
        key=lambda x: (
            x["failure_count"],
            -x["accuracy"]
        )
    )

    print("\n========================================")
    print("BEST REFINEMENT CANDIDATE")
    print("========================================")
    print(f"Candidate: {best_candidate['candidate_id']}")
    print(f"Modified CPTs: {[x['name'] for x in best_candidate['patch']['modified_cpts']]}")
    print(f"Accuracy: {best_candidate['accuracy']:.4f}")
    print(f"Failures: {best_candidate['failure_count']}")
    print(f"Successes: {best_candidate['success_count']}")

    # return {
    #     "baseline_bn_number": best_bn_number,
    #     "selected_candidate_id": best_candidate["candidate_id"],
    #     "selected_bn": best_candidate["bn"],
    #     "selected_accuracy": best_candidate["accuracy"],
    #     "selected_failure_count": best_candidate["failure_count"],
    #     "selected_success_count": best_candidate["success_count"],
    #     "candidate_results": candidate_results
    # }

    return best_candidate["bn"]


### -----------------------------

if __name__ == "__main__":
    new_bn = generate_and_select_best_candidate(
        full_context=CONTEXT_AGENT_FILE,
        proposed_bn_filename=PROPOSED_BN_FILE,
        bn_analysis_filename=BN_ANALYSIS_FILE,
        train_csv=TRAIN_CSV,
        temperature=0.3
    )

    store_new_bn(2, new_bn)
