import os
import json
import re
import copy

# from bn_generator import generate_bn, store_bn_proposal
from automatic_bn_reasoning_old import run_evaluation as initial_run_evaluation
from bn_generator_evaluator import find_proposed_bn, run_evaluation, store_analysis
from bn_generator_reflexion import generate_and_select_best_candidate, store_new_bn
from bn_validator import GT_FILE, compute_average_cpt_hellinger, compute_average_cpt_kl, compute_average_cpt_rmse, get_best_bn_number, compare_all_cpts, get_bn, normalize_bn, read_json

bn_analysis_filename = "bn_analysis.json"
proposed_bn_filename = "last_proposed_bn.jsonl"
train_csv="combined_train_scenarios.csv"

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
for filename in [bn_analysis_filename, proposed_bn_filename, "dangerous_cpt_report.json", "failure_parameter_statistics.json", "restart_final_bns.jsonl"]:
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


def store_restart_final_bn(
    restart_count,
    best_bn_number,
    best_bn_accuracy,
    best_bn,
    filename
):
    record = {
        "restart_number": restart_count,
        "best_bn_number": best_bn_number,
        "best_bn_accuracy": best_bn_accuracy,
        "bn": best_bn
    }

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_best_restart_bn(filename="restart_final_bns.jsonl"):

    best_restart = None
    best_accuracy = -1
    best_failure_count = float("inf")
    best_bn = None

    with open(filename, "r", encoding="utf-8") as f:

        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            bn = record["bn"]

            print("Restart:", record["restart_number"])
            print("BN keys:", record["bn"].keys())

            print(f"Evaluating restart {record['restart_number']}")

            failures, successes, accuracy, _ = initial_run_evaluation(
                bn,
                train_csv
            )

            failure_count = len(failures)

            if (
                failure_count < best_failure_count or
                (
                    failure_count == best_failure_count and
                    accuracy > best_accuracy
                )
            ):
                best_failure_count = failure_count
                best_accuracy = accuracy
                best_restart = record["restart_number"]
                best_bn = bn

    return (
        best_restart,
        best_bn,
        best_accuracy,
        best_failure_count
    )


def remove_analysis(
    bn_number,
    filename="bn_analysis.json"
):

    if not os.path.exists(filename):
        return

    retained_records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("bn_number") != bn_number:
                retained_records.append(record)

    with open(filename, "w", encoding="utf-8") as f:
        for record in retained_records:
            f.write(json.dumps(record) + "\n")

    print(
        f"Removed workflow analysis for BN #{bn_number} "
        f"from {filename}"
    )


# -----------------------------
MAX_RESTARTS = 3
restart_count = 0
restart_final_bn_filename = "restart_final_bns.jsonl"

BASE_TEMPERATURE = 0.2

while restart_count < MAX_RESTARTS:

    print(f"\n==============================")
    print(f"PIPELINE RESTART #{restart_count}")
    print(f"==============================")

    CURRENT_TEMPERATURE = BASE_TEMPERATURE + 0.1 * restart_count
    MAX_ITER = 3
    max_records = 3

    # -----------------------------
    # CLEAR OLD FILES
    # -----------------------------
    for filename in [bn_analysis_filename, proposed_bn_filename, "dangerous_cpt_report.json", "failure_parameter_statistics.json"]:
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

    ###------------------------------------------------------
    # failures, successes, flawed_accuracy, results = initial_run_evaluation(safe_json_loads(flawed_bn), train_csv)

    # best_bn = None
    # best_accuracy = float("-inf")

    # initial_retry = 0
    # MAX_INITIAL_RETRIES = 3
    # INITIAL_IMPROVEMENT_RATIO = 0.30  # recover at least 30% of the gap

    # INITIAL_ACCURACY_THRESHOLD = (
    #     flawed_accuracy +
    #     INITIAL_IMPROVEMENT_RATIO * (1.0 - flawed_accuracy)
    # )

    # print(f"Flawed BN Accuracy: {100*flawed_accuracy:.2f}%")
    # print(f"Initial Acceptance Threshold: {100*INITIAL_ACCURACY_THRESHOLD:.2f}%")

    # for initial_retry in range(MAX_INITIAL_RETRIES):

    #     print(f"\nInitial Generation Attempt {initial_retry+1}")

    #     bn_text = generate_bn(
    #         full_context,
    #         flawed_bn,
    #         success_report,
    #         failure_report,
    #         gen_prompt_template_text,
    #         temperature=CURRENT_TEMPERATURE
    #     )

    #     bn_json = safe_json_loads(bn_text)

    #     if bn_json is None:
    #         print("Invalid JSON.")
    #         continue

    #     failures, successes, accuracy, results = initial_run_evaluation(
    #         bn_json,
    #         train_csv
    #     )

    #     print(f"Accuracy = {100*accuracy:.2f}%")

    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_bn = copy.deepcopy(bn_json)

    #     if accuracy >= INITIAL_ACCURACY_THRESHOLD:
    #         print("Initial BN accepted.")
    #         break

    # if best_bn is None:
    #     print("Unable to generate a valid BN.")
    #     continue

    # bn_json = best_bn

    # store_bn_proposal(
    #     bn_json,
    #     bn_number,
    #     proposed_bn_filename
    # )

    # if best_accuracy >= INITIAL_ACCURACY_THRESHOLD:
    #     print(f"Accepted initial BN ({100*best_accuracy:.2f}%).")
    # else:
    #     print(
    #         f"Threshold ({100*INITIAL_ACCURACY_THRESHOLD:.2f}%) "
    #         f"not reached after {MAX_INITIAL_RETRIES} attempts. "
    #         f"Using best generated BN ({100*best_accuracy:.2f}%)."
    #     )

    # print("\nInitial BN generated")

    ###------------------------------------------------------
    # Start a fresh proposed BN file
    ###------------------------------------------------------

    open(proposed_bn_filename, "w").close()

    flawed_bn = safe_json_loads(flawed_bn)

    # ----------------------------------------
    # Store flawed BN
    # ----------------------------------------
    store_new_bn(
        0,
        flawed_bn
    )

    # ----------------------------------------
    # Evaluate flawed BN
    # ----------------------------------------
    failures, successes, flawed_accuracy, _ = initial_run_evaluation(flawed_bn, train_csv)

    best_bn = None
    best_accuracy = float("-inf")
    MAX_INITIAL_RETRIES = 3
    INITIAL_IMPROVEMENT_RATIO = 0.30  # recover at least 30% of the gap

    INITIAL_ACCURACY_THRESHOLD = (
        flawed_accuracy +
        INITIAL_IMPROVEMENT_RATIO * (1.0 - flawed_accuracy)
    )

    print(f"Flawed BN Accuracy: {100*flawed_accuracy:.2f}%")
    print(f"Initial Acceptance Threshold: {100*INITIAL_ACCURACY_THRESHOLD:.2f}%")

    for retry in range(MAX_INITIAL_RETRIES):
        
        print(f"\nInitial Generation Attempt {retry}")

        if retry > 0:
            remove_analysis(bn_number, bn_analysis_filename)

        evaluation_output = run_evaluation(flawed_bn, bn_number=bn_number, temperature=CURRENT_TEMPERATURE)
        store_analysis(bn_number, evaluation_output)

        new_bn = generate_and_select_best_candidate(full_context, proposed_bn_filename, bn_analysis_filename, train_csv, temperature=CURRENT_TEMPERATURE) 
        
        failures, successes, accuracy, _ = initial_run_evaluation(new_bn, train_csv)

        print(f"Accuracy = {100*accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_bn = copy.deepcopy(new_bn)

        if accuracy >= INITIAL_ACCURACY_THRESHOLD:
            print("Initial BN accepted.")
            break


    if best_accuracy >= INITIAL_ACCURACY_THRESHOLD:
        print(f"Accepted initial BN ({100*best_accuracy:.2f}%).")
    else:
        print(
            f"Threshold ({100*INITIAL_ACCURACY_THRESHOLD:.2f}%) "
            f"not reached after {MAX_INITIAL_RETRIES} attempts. "
            f"Using best generated BN ({100*best_accuracy:.2f}%)."
        )
    
    
    remove_analysis(bn_number, bn_analysis_filename)
    bn_number += 1 
    store_new_bn(bn_number, best_bn)

    print("\nInitial BN generated")

    ###------------------------------------------------------
    solved = False
    i = 1
    MAX_NO_IMPROVEMENT_RETRIES = 3

    # -----------------------------
    # STEP 2 & 3: ITERATIONS
    # -----------------------------
    while i <= MAX_ITER:

        print(f"\n===== ITERATION {i+1} =====")

        best_retry_bn = None
        best_retry_accuracy = float("-inf")

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

        evaluation_output = run_evaluation(prev_bn_json, bn_number=bn_number, temperature=CURRENT_TEMPERATURE)
        
        store_analysis(bn_number, evaluation_output)

        new_bn = generate_and_select_best_candidate(
                full_context,
                proposed_bn_filename,
                bn_analysis_filename,
                train_csv,
                temperature=CURRENT_TEMPERATURE
        )

        bn_number += 1

        store_new_bn(
                bn_number,
                new_bn
        )

        failures, successes, new_accuracy, results = initial_run_evaluation(
            new_bn, train_csv
        )

        if new_accuracy > accuracy:
            print(
                f"\nAccuracy improved from {100*accuracy:.2f}% "
                f"to {100*new_accuracy:.2f}%."
            )
        
        else:
            print(
                f"\nAccuracy did not improve. Need improvement from {100*accuracy:.2f}% ")
        
        # Track best BN seen during retry cycle
        best_retry_accuracy = new_accuracy
        best_retry_bn = copy.deepcopy(new_bn)

        ### ------------------------------
        # Check if accuracy has improved
        ### ------------------------------
        improved = new_accuracy > accuracy
        no_improvement_retry = 0

        while not improved and no_improvement_retry < MAX_NO_IMPROVEMENT_RETRIES:
            
            # Remove the unimproved BN before refinement so it is not included in previous_bns memory. 
            # The new BN will be generated using only earlier accepted BNs and analysis memory.
            remove_bn(bn_number, proposed_bn_filename)

            new_bn = generate_and_select_best_candidate(
                full_context,
                proposed_bn_filename,
                bn_analysis_filename,
                train_csv,
                temperature=CURRENT_TEMPERATURE
            )

            store_new_bn(
                bn_number,
                new_bn
            )

            failures, successes, new_accuracy, results = initial_run_evaluation(
                new_bn, train_csv
            )

            if new_accuracy > best_retry_accuracy:
                best_retry_accuracy = new_accuracy
                best_retry_bn = copy.deepcopy(new_bn)

            if new_accuracy > accuracy:
                print(
                    f"\nAccuracy improved from {100*accuracy:.2f}% "
                    f"to {100*new_accuracy:.2f}%."
                )
                improved = True   
                break

            no_improvement_retry += 1

            if no_improvement_retry >= MAX_NO_IMPROVEMENT_RETRIES:
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

                new_bn= best_retry_bn
                new_accuracy = best_retry_accuracy

                print(
                    f"Continuing with retry candidate "
                    f"(accuracy={100*new_accuracy:.2f}%)."
                )

        
        i+=1

    # -----------------------------
    # VALIDATION
    # -----------------------------
    best_bn_number, best_bn_accuracy = get_best_bn_number(proposed_bn_filename)
    print("\n\nBest BN Number:", best_bn_number)
    print("Best BN Accuracy:", best_bn_accuracy)

    final_output = compare_all_cpts(bn_number=best_bn_number)
    print(final_output)

    gt_bn = normalize_bn(read_json(GT_FILE))
    best_bn = get_bn(proposed_bn_filename, bn_number=best_bn_number)
    nor_best_bn =normalize_bn(best_bn)

    TARGET_NODES = {
        "Network_Manipulation",
        "Physical_Anomaly",
        "Program_Anomaly",
        "Execution_Integrity",
        "Deviation_in_Response",
        "Deviation_in_Dispatch",
        "Root_Causes",
    }

    compute_average_cpt_kl(gt_bn, nor_best_bn, target_nodes=TARGET_NODES)
    compute_average_cpt_rmse(gt_bn, nor_best_bn, target_nodes=TARGET_NODES)
    compute_average_cpt_hellinger(gt_bn, nor_best_bn, target_nodes=TARGET_NODES)

    store_restart_final_bn(
        restart_count=restart_count,
        best_bn_number=best_bn_number,
        best_bn_accuracy=best_bn_accuracy,
        best_bn=best_bn,
        filename=restart_final_bn_filename
    )

    print("###------------------------------###")
    print(f"\nRestart {restart_count} finished.")
    print("###------------------------------###")
    
    restart_count += 1


###------------------------------------------
best_restart, best_bn, accuracy, failures = (
    get_best_restart_bn()
)

print("\n===================================")
print("BEST RESTART")
print("===================================")

print("Restart:", best_restart)
print("Accuracy:", accuracy)
print("Failures:", failures)

print("###------------------------------###")
print("\nPipeline finished.")
print("###------------------------------###")
