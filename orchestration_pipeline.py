import os
import json
import re
import copy
import pandas as pd

from bn_generator import generate_bn, store_bn_proposal
from automatic_bn_reasoning_old import run_evaluation as initial_run_evaluation
from bn_generator_evaluator import find_proposed_bn, run_evaluation, store_analysis
from bn_generator_reflexion import generate_refined_bn, store_new_bn
from bn_validator import GT_FILE, compute_average_cpt_hellinger, compute_average_cpt_kl, compute_average_cpt_rmse, get_best_bn_number, compare_all_cpts, get_bn, normalize_bn, read_json

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


def remove_bn(bn_number, filename="last_proposed_bn.jsonl"):
    if not os.path.exists(filename):
        print(f"{filename} does not exist.")
        return

    records = []
    removed_count = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if record.get("bn_number") == bn_number:
                    removed_count += 1
                else:
                    records.append(record)

            except json.JSONDecodeError:
                continue

    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(
        f"Removed {removed_count} record(s) "
        f"with bn_number={bn_number}"
    )


# -----------------------------
def compute_failure_ratio_from_results(evaluation_output):
    failed = evaluation_output.get("failure_count", 0)
    succeeded = evaluation_output.get("success_count", 0)

    total = failed + succeeded

    if total == 0:
        return 0.0, 0, 0, 0

    return failed / total, failed, succeeded, total


# -----------------------------
MAX_RESTARTS = 0
restart_count = 0

while restart_count <= MAX_RESTARTS:

    print(f"\n==============================")
    print(f"PIPELINE RESTART #{restart_count}")
    print(f"==============================")

    MAX_ITER = 3
    max_records = 3
    previous_accuracy = 0
    FAILURE_RATIO_THRESHOLD = 0.03  # stop if failed cases <= X%

    # -----------------------------
    # CLEAR OLD FILES
    # -----------------------------
    for filename in [bn_analysis_filename, proposed_bn_filename, "failure_cpt_parameter_risks.jsonl"]:
        if os.path.exists(filename):
            open(filename, "w").close()
            print(f"Cleared: {filename}")

    # -----------------------------
    # STEP 1: INITIAL BN
    # -----------------------------
    bn_number = 0
    full_context = read_file("context_agent.txt")
    success_report = read_file("merged_flawed_success_train.json")
    failure_report = read_file("merged_flawed_failure_train.json")
    original_success_report = read_file("merged_flawed_success_train.json")
    original_failure_report = read_file("merged_flawed_failure_train.json")
    gen_prompt_template_text = read_file("gen_prompt.txt")
    flawed_bn=read_file("flawed_BN_0.json")

    train_csv="combined_train_scenarios.csv"

    # bn_text = generate_bn(full_context, flawed_bn, success_report, failure_report, gen_prompt_template_text)

    # bn_json = safe_json_loads(bn_text)

    # if bn_json is None:
    #     print("\nInitial BN generation failed. Trying again...")
    #     continue

    # store_bn_proposal(bn_json, bn_number, proposed_bn_filename)

    ###------------------------------------------------------

    failures, successes, flawed_accuracy, results = initial_run_evaluation(safe_json_loads(flawed_bn), train_csv)

    best_bn = None
    best_accuracy = float("-inf")

    initial_retry = 0
    MAX_INITIAL_RETRIES = 3
    INITIAL_IMPROVEMENT_RATIO = 0.30  # recover at least 30% of the gap

    INITIAL_ACCURACY_THRESHOLD = (
        flawed_accuracy +
        INITIAL_IMPROVEMENT_RATIO * (1.0 - flawed_accuracy)
    )

    print(f"Flawed BN Accuracy: {100*flawed_accuracy:.2f}%")
    print(f"Initial Acceptance Threshold: {100*INITIAL_ACCURACY_THRESHOLD:.2f}%")

    for initial_retry in range(MAX_INITIAL_RETRIES):

        print(f"\nInitial Generation Attempt {initial_retry+1}")

        bn_text = generate_bn(
            full_context,
            flawed_bn,
            success_report,
            failure_report,
            gen_prompt_template_text
        )

        bn_json = safe_json_loads(bn_text)

        if bn_json is None:
            print("Invalid JSON.")
            continue

        failures, successes, accuracy, results = initial_run_evaluation(
            bn_json,
            train_csv
        )

        print(f"Accuracy = {100*accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_bn = copy.deepcopy(bn_json)

        if accuracy >= INITIAL_ACCURACY_THRESHOLD:
            print("Initial BN accepted.")
            break

    if best_bn is None:
        print("Unable to generate a valid BN.")
        continue

    bn_json = best_bn

    store_bn_proposal(
        bn_json,
        bn_number,
        proposed_bn_filename
    )

    if best_accuracy >= INITIAL_ACCURACY_THRESHOLD:
        print(f"Accepted initial BN ({100*best_accuracy:.2f}%).")
    else:
        print(
            f"Threshold ({100*INITIAL_ACCURACY_THRESHOLD:.2f}%) "
            f"not reached after {MAX_INITIAL_RETRIES} attempts. "
            f"Using best generated BN ({100*best_accuracy:.2f}%)."
        )

    print("\nInitial BN generated")

    ###------------------------------------------------------

    solved = False
    i = 0
    MAX_NO_IMPROVEMENT_RETRIES = 3
    no_improvement_retry = 0

    best_retry_bn = None
    best_retry_accuracy = float("-inf")

    # -----------------------------
    # STEP 2 & 3: ITERATIONS
    # -----------------------------
    while i <= MAX_ITER:

        print(f"\n===== ITERATION {i+1} =====")

        if no_improvement_retry == 0:
            best_retry_bn = None
            best_retry_accuracy = float("-inf")

        if i == MAX_ITER: # no need to reflex if it's the last iteration
            
            prev_bn_json = find_proposed_bn(
                bn_number,
                proposed_bn_filename
            )

            ### ------------------------------
            # Initial evaluation to check for refinement
            ### ------------------------------
            failures, successes, accuracy, results = initial_run_evaluation(
                prev_bn_json, train_csv
            )

            # Track best BN seen during retry cycle
            if accuracy > best_retry_accuracy:
                best_retry_accuracy = accuracy
                best_retry_bn = copy.deepcopy(prev_bn_json)

            if accuracy <= previous_accuracy:

                if no_improvement_retry >= MAX_NO_IMPROVEMENT_RETRIES:
                    print(
                        f"\nNo improvement after {MAX_NO_IMPROVEMENT_RETRIES} retries. "
                        f"Using best retry candidate "
                        f"(accuracy={100*best_retry_accuracy:.2f}%)."
                    )

                    remove_bn(
                        bn_number,
                        proposed_bn_filename
                    )

                    store_new_bn(
                        bn_number,
                        best_retry_bn
                    )

                    previous_accuracy = best_retry_accuracy
                    no_improvement_retry = 0

                    break
                
                # Remove the unimproved BN before refinement so it is not included in previous_bns memory. 
                # The new BN will be generated using only earlier accepted BNs and analysis memory.
                remove_bn(bn_number, proposed_bn_filename)

                new_bn = generate_refined_bn(
                    full_context,
                    flawed_bn,
                    original_success_report,
                    original_failure_report,
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

                store_new_bn(
                    bn_number,
                    new_bn_json
                )

                no_improvement_retry += 1

                print(
                    f"\nNo improvement at final iteration. "
                    f"Retry {no_improvement_retry}/{MAX_NO_IMPROVEMENT_RETRIES}. "
                    f"Regenerated BN#{bn_number}."
                )

                continue

            previous_accuracy = accuracy
            no_improvement_retry = 0
            break


        ### ------------------------------
        # Evaluate
        ### ------------------------------
        prev_bn_json = find_proposed_bn(
            bn_number,
            proposed_bn_filename
        )

        ### ------------------------------
        # Initial evaluation to get failures for reflexion and analysis
        ### ------------------------------
        failures, successes, accuracy, results = initial_run_evaluation(
            prev_bn_json, train_csv
        )

        # Track best BN seen during retry cycle
        if accuracy > best_retry_accuracy:
            best_retry_accuracy = accuracy
            best_retry_bn = copy.deepcopy(prev_bn_json)

        ### ------------------------------
        # Check if accuracy has improved
        ### ------------------------------
        if accuracy <= previous_accuracy and no_improvement_retry < MAX_NO_IMPROVEMENT_RETRIES:
            
            # Remove the unimproved BN before refinement so it is not included in previous_bns memory. 
            # The new BN will be generated using only earlier accepted BNs and analysis memory.
            remove_bn(bn_number, proposed_bn_filename)

            new_bn = generate_refined_bn(
                full_context,
                flawed_bn,
                original_success_report,
                original_failure_report,
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

            store_new_bn(
                bn_number,
                new_bn_json
            )

            no_improvement_retry += 1

            print(f"\nNo improvement, skip analysis and reflexion for this iteration BN#{bn_number} stored. New BN generated without analysis.")

            continue   
        

        else:

            if accuracy <= previous_accuracy:
                print(
                    f"\nNo improvement after {MAX_NO_IMPROVEMENT_RETRIES} retries. "
                    f"Selecting best retry candidate "
                    f"(accuracy={100*best_retry_accuracy:.2f}%)."
                )

                # Restore best BN found during retry cycle
                remove_bn(bn_number, proposed_bn_filename)

                store_new_bn(
                    bn_number,
                    best_retry_bn
                )

                prev_bn_json = best_retry_bn
                accuracy = best_retry_accuracy

            previous_accuracy = accuracy
            no_improvement_retry = 0

            # Reset tracker for next iteration
            best_retry_bn = None
            best_retry_accuracy = float("-inf")

            for filename in ["dangerous_cpt_report.json"]:
                if os.path.exists(filename):
                    open(filename, "w").close()
                    print(f"Cleared: {filename}")

            prev_bn_json = find_proposed_bn(
                bn_number,
                proposed_bn_filename
            )
            
            evaluation_output = run_evaluation(
                prev_bn_json, bn_number=bn_number
            )

            failure_ratio, failed, succeeded, total = compute_failure_ratio_from_results(
                evaluation_output
            )

            print(
                f"\nFailure ratio: {failure_ratio:.4f} "
                f"({failed}/{total} failed)"
            )

            store_analysis(
                bn_number,
                evaluation_output
            )

            print("\nEvaluation completed. Analysis stored.")

            # -----------------------------
            # SUCCESS CONDITION
            # -----------------------------
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
                flawed_bn,
                original_success_report,
                original_failure_report,
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

            i+=1

    if solved:
        print("\nProblem solved! Maximum Restarts: ", restart_count)
        break
    else:
        restart_count += 1


# -----------------------------
# VALIDATION
# -----------------------------
best_bn_number, best_bn_accuracy = get_best_bn_number(proposed_bn_filename)
print("\n\nBest BN Number:", best_bn_number)
print("Best BN Accuracy:", best_bn_accuracy)

final_output = compare_all_cpts(bn_number=best_bn_number)
print(final_output)

gt_bn = normalize_bn(read_json(GT_FILE))
prop_bn = normalize_bn(get_bn(proposed_bn_filename, bn_number=best_bn_number))

TARGET_NODES = {
    "Network_Manipulation",
    "Physical_Anomaly",
    "Program_Anomaly",
    "Execution_Integrity",
    "Deviation_in_Response",
    "Deviation_in_Dispatch",
    "Root_Causes",
}

compute_average_cpt_kl(gt_bn, prop_bn, target_nodes=TARGET_NODES)
compute_average_cpt_rmse(gt_bn, prop_bn, target_nodes=TARGET_NODES)
compute_average_cpt_hellinger(gt_bn, prop_bn, target_nodes=TARGET_NODES)
print("\nPipeline finished.")
