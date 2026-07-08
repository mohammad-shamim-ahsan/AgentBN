from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import json
import copy
import re

from automatic_bn_reasoning_old import run_evaluation as initial_run_evaluation

client = OpenAI(api_key="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")

def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    return response.output[0].content[0].text.strip()


# -----------------------------
# SAFE JSON LOADER
# -----------------------------
def safe_json_loads(text):
    if not text or not text.strip():
        return None

    text = text.strip()

    # remove markdown code fences if present
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    

###-----------------------------
bn_analysis_filename="bn_analysis.json"
max_records=3
proposed_bn_filename="last_proposed_bn.jsonl"
train_csv="combined_train_scenarios.csv"

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
    
full_context = read_file("context_agent.txt")
prompt_template_text = read_file("ref_prompt.txt")

### -----------------------------
def read_all_analysis_records(filename=bn_analysis_filename):
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


def get_best_bn_number(filename=bn_analysis_filename):
    records = read_all_analysis_records(filename)

    if not records:
        raise ValueError("No analysis records found.")

    best_record = min(
        records,
        key=lambda r: r.get("failure_count", float("inf"))
    )

    return best_record["bn_number"]


def get_analysis_record(bn_number, filename=bn_analysis_filename):
    records = read_all_analysis_records(filename)

    for record in records:
        if record.get("bn_number") == bn_number:
            return record

    raise ValueError(f"No analysis record found for BN #{bn_number}")


def find_proposed_bn(bn_number, filename=proposed_bn_filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("bn_number") == bn_number:
                return record["bn"]

    raise ValueError(f"No proposed BN found for BN #{bn_number}")


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
    proposed_bn_filename=proposed_bn_filename,
    bn_analysis_filename=bn_analysis_filename,
    temperature=0.3
):
    # Deterministically select the baseline BN
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

    prompt_gen_template = PromptTemplate(
        input_variables=[
            "full_context",
            "baseline_bn",
            "analysis_record"
        ],
        template=prompt_template_text
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        baseline_bn=json.dumps(
            baseline_bn,
            indent=2
        ),
        analysis_record=formatted_analysis_record
    )

    response = llm(prompt, temperature=temperature)

    response_json = json.loads(response)

    print(response_json)

    return {
        "best_bn_number": best_bn_number,
        "baseline_bn": baseline_bn,
        "candidates": response_json["candidates"]
    }


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


def store_new_bn(bn_number, bn_new):
    record = {
        "bn_number": bn_number,
        "bn": bn_new
    }

    with open(proposed_bn_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def generate_and_select_best_candidate(
    full_context,
    proposed_bn_filename=proposed_bn_filename,
    bn_analysis_filename=bn_analysis_filename,
    train_csv=train_csv,
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
        full_context=full_context,
        proposed_bn_filename=proposed_bn_filename,
        bn_analysis_filename=bn_analysis_filename,
        train_csv=train_csv,
        temperature=0.3
    )

    store_new_bn(
        2,
        new_bn
    )