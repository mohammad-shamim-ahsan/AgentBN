from langchain_core.prompts import PromptTemplate

import json
import copy

from utils.bn_io import *
from utils.json_utils import *
from utils.file_utils import *
from config.settings import *
from utils.llm import *
from evaluation.automatic_bn_reasoning_old import run_evaluation as initial_run_evaluation

# -----------------------------------------------

MAX_FORMAT_RETRIES = 3
MAX_REPAIR_RETRIES = 2

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


def validate_refinement_response(response):
    """
    Perform top-level format verification.

    Returns:
        Parsed JSON object if valid.

    Raises:
        json.JSONDecodeError
        ValueError
    """
    response_json = json.loads(response)

    if not isinstance(response_json, dict):
        raise ValueError("The top-level JSON value must be an object.")

    if "candidates" not in response_json:
        raise ValueError("Missing required top-level field: 'candidates'.")

    if not isinstance(response_json["candidates"], list):
        raise ValueError("'candidates' must be a list.")

    if len(response_json["candidates"]) != 3:
        raise ValueError(
            "'candidates' must contain exactly 3 refinement candidates."
        )

    return response_json


def build_return_object(best_bn_number, baseline_bn, response_json):
    return {
        "best_bn_number": best_bn_number,
        "baseline_bn": baseline_bn,
        "candidates": response_json["candidates"]
    }


def read_all_analysis_records(filename=PROPOSED_BN_FILE):
    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except Exception:
                continue

    return records


def get_best_bn_number(filename=PROPOSED_BN_FILE):
    records = read_all_analysis_records(filename)

    if not records:
        raise ValueError("No analysis records found.")

    best_record = min(
        records,
        key=lambda r: r.get("failure_count", float("inf"))
    )

    return best_record["bn_number"]


def get_analysis_record(bn_number, filename=BN_ANALYSIS_FILE):
    records = read_all_analysis_records(filename)

    for record in records:
        if record.get("bn_number") == bn_number:
            return record

    raise ValueError(f"No analysis record found for BN #{bn_number}")


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


def generate_refined_cpt_patch(
    full_context,
    proposed_bn_filename=PROPOSED_BN_FILE,
    bn_analysis_filename=BN_ANALYSIS_FILE,
    temperature=0.3
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
            "analysis_record"
        ],
        template=read_file(REF_PROMPT_FILE)
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        baseline_bn=json.dumps(
            baseline_bn,
            indent=2
        ),
        analysis_record=formatted_analysis_record
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
                response
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

The response violates the required JSON format.

Validation error:
{last_error}

Previous response:
{last_response}

Your task is ONLY to repair the formatting.

Do NOT:
- modify any candidate;
- modify any CPT;
- modify any probability;
- modify candidate IDs;
- add or remove candidates;
- change any Bayesian Network content.

Repair only the JSON syntax and structure so that the response conforms to the required output format.

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
                repaired_response
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


def integrate_modified_cpts(baseline_bn, patch):
    refined_bn = copy.deepcopy(baseline_bn)

    node_lookup = {
        node["name"]: node
        for node in refined_bn["nodes"]
    }

    modified_cpts = patch.get("modified_cpts", [])

    for item in modified_cpts:
        name = item["name"]

        if name not in node_lookup:
            raise ValueError(f"Modified CPT node not found in BN: {name}")

        node_lookup[name]["cpt"] = item["cpt"]

    return refined_bn


def generate_and_select_best_candidate(
    full_context,
    proposed_bn_filename=PROPOSED_BN_FILE,
    bn_analysis_filename=BN_ANALYSIS_FILE,
    train_csv=TRAIN_CSV,
    temperature=0.3
):
    refinement_output = generate_refined_cpt_patch(
        full_context=full_context,
        proposed_bn_filename=proposed_bn_filename,
        bn_analysis_filename=bn_analysis_filename,
        temperature=temperature
    )

    best_bn_number = refinement_output["best_bn_number"]
    baseline_bn = refinement_output["baseline_bn"]
    candidates = refinement_output["candidates"]

    print(f"Selected baseline BN: {best_bn_number}")

    candidate_results = []

    for candidate in candidates:

        refined_bn = integrate_modified_cpts(
            baseline_bn,
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
    print(f"Candidate {best_candidate['candidate_id']}")
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
