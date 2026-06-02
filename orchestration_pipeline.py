import os
import json
import re

from bn_generator import generate_bn, store_bn_proposal
from bn_generator_evaluator import find_proposed_bn, run_evaluation, store_analysis
from bn_generator_reflexion import generate_refined_bn, store_new_bn
from bn_validator import get_best_bn_number, compare_all_cpts
from bn_generator_evaluator_old import run_evaluation as run_evaluation_final, store_analysis as store_analysis_final

bn_analysis_filename = "bn_analysis.json"
proposed_bn_filename = "last_proposed_bn.jsonl"

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

# -----------------------------
# CLEAR OLD FILES
# -----------------------------
for filename in [bn_analysis_filename, proposed_bn_filename]:
    if os.path.exists(filename):
        open(filename, "w").close()
        print(f"Cleared: {filename}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
def compute_failure_ratio_from_results(evaluation_output):
    failed = evaluation_output.get("failure_count", 0)
    succeeded = evaluation_output.get("success_count", 0)

    total = failed + succeeded

    if total == 0:
        return 0.0, 0, 0, 0

    return failed / total, failed, succeeded, total


def keep_only_last_jsonl_record(filename):
    last_line = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()

    if last_line is not None:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(last_line + "\n")


def keep_only_last_analysis_record(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return

    # Case 1: JSONL-style analysis file
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) > 1:
        try:
            json.loads(lines[-1])
            with open(filename, "w", encoding="utf-8") as f:
                f.write(lines[-1] + "\n")
            return
        except json.JSONDecodeError:
            pass

    # Case 2: normal JSON list
    try:
        data = json.loads(content)

        if isinstance(data, list) and data:
            data = [data[-1]]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    except json.JSONDecodeError:
        print(f"Warning: Could not trim {filename}; invalid JSON format.")

# -----------------------------
MAX_ITER = 5
max_records = 3
MAX_RESTARTS = 0

restart_count = 0
# previous_failure_ratio = 100

FAILURE_RATIO_THRESHOLD = 0.03  # stop if failed cases <= X%

while restart_count <= MAX_RESTARTS:

    print(f"\n==============================")
    print(f"PIPELINE RESTART #{restart_count}")
    print(f"==============================")

    # -----------------------------
    # CLEAR OLD FILES
    # -----------------------------
    for filename in [bn_analysis_filename, proposed_bn_filename]:
        if os.path.exists(filename):
            open(filename, "w").close()
            print(f"Cleared: {filename}")

    # -----------------------------
    # STEP 1: INITIAL BN
    # -----------------------------
    
    bn_number = 0
    full_context = read_file("context_agent.txt")
    success_report = read_file("flawed_success_results.json")
    failure_report = read_file("flawed_failure_results.json")
    gen_prompt_template_text = read_file("gen_prompt.txt")
    flawed_bn=read_file("flawed_BN_0.json")

    bn_text = generate_bn(full_context, flawed_bn, success_report, failure_report, gen_prompt_template_text)

    bn_json = safe_json_loads(bn_text)

    if bn_json is None:
        print("\nInitial BN generation failed.")
        restart_count += 1
        continue

    store_bn_proposal(bn_json, bn_number, proposed_bn_filename)

    print("\nInitial BN generated")

    solved = False

    # -----------------------------
    # STEP 2 & 3: ITERATIONS
    # -----------------------------
    for i in range(MAX_ITER):

        print(f"\n===== ITERATION {i+1} =====")

        ### Evaluate
        prev_bn_json = find_proposed_bn(
            bn_number,
            proposed_bn_filename
        )

        for filename in ["failure_cpt_parameter_risks.jsonl", "dangerous_cpt_report.json"]:
            if os.path.exists(filename):
                open(filename, "w").close()
                print(f"Cleared: {filename}")

        evaluation_output = run_evaluation(
            prev_bn_json, bn_number=bn_number
        )

        store_analysis(
            bn_number,
            evaluation_output
        )

        print("\nEvaluation completed. Analysis stored.")

        # -----------------------------
        # SUCCESS CONDITION
        # -----------------------------
        if not evaluation_output.get("failure_scenarios_text") or not evaluation_output.get("failure_scenarios_text").strip():

            print(
                f"\nNo failure cases found for BN {bn_number}. "
                f"Stopping iterations."
            )

            solved = True
            break

        failure_ratio, failed, succeeded, total = compute_failure_ratio_from_results(
            evaluation_output
        )

        print(
            f"\nFailure ratio: {failure_ratio:.4f} "
            f"({failed}/{total} failed)"
        )
        
        if failure_ratio <= FAILURE_RATIO_THRESHOLD:

            print(
                f"\nFailure ratio is below threshold "
                f"({failure_ratio:.4f} <= {FAILURE_RATIO_THRESHOLD}). "
                f"Stopping iterations."
            )

            solved = True
            break

        # -----------------------------
        # REFLEXION
        # -----------------------------
        new_bn = generate_refined_bn(
            full_context,
            bn_number,
            proposed_bn_filename,
            bn_analysis_filename,
            max_records
        )

        new_bn_json = safe_json_loads(new_bn)

        if new_bn_json is None:

            print("\nInvalid JSON from LLM during Reflexion.")
            print("Stopping safely.")

            break

        bn_number += 1

        store_new_bn(
            bn_number,
            new_bn_json
        )

        print(f"\nReflexion completed. BN#{bn_number} stored.")


    # -----------------------------
    # FINAL CHECK AFTER MAX ITER
    # -----------------------------
    prev_bn_json = find_proposed_bn(
        bn_number,
        proposed_bn_filename
    )

    failure_text, success_text, analysis, results = run_evaluation_final(
        prev_bn_json
    )

    store_analysis_final(
        bn_number, failure_text, success_text, analysis, results
    )

    # -----------------------------
    # KEEP ONLY FINAL RECORDS
    # -----------------------------
    keep_only_last_jsonl_record(
        proposed_bn_filename
    )

    keep_only_last_analysis_record(
        bn_analysis_filename
    )

    failure_ratio, failed, succeeded, total = compute_failure_ratio_from_results(
        evaluation_output
    )

    print(
        f"\nFinal failure ratio: {failure_ratio:.4f} "
        f"({failed}/{total} failed)"
    )

    if failure_ratio > FAILURE_RATIO_THRESHOLD:
        print(
            "\nFailure ratio still above threshold after "
            f"{MAX_ITER} iterations."
        )

        print("\nRestarting pipeline from STEP 1...\n")

        restart_count += 1
        continue

    # -----------------------------
    # SUCCESS
    # -----------------------------
    print(
        "\nPipeline solved successfully "
        f"with failure ratio {failure_ratio:.4f}."
    )
    
    break

# -----------------------------
# VALIDATION
# -----------------------------
best_bn_number, best_bn_accuracy = get_best_bn_number(proposed_bn_filename)
print("Best BN Number:", best_bn_number)
print("Best BN Accuracy:", best_bn_accuracy)
compare_all_cpts(bn_number=best_bn_number)
print("\nPipeline finished.")
