import os
import json
import copy
import pandas as pd
import argparse

from config.settings import *
from utils.json_utils import *
from utils.bn_io import *
from utils.file_utils import *

from evaluation.automatic_bn_reasoning_old import run_evaluation as initial_run_evaluation
from agents.bn_evaluator import run_evaluation, store_analysis
from agents.bn_generator_reflexion import generate_and_select_best_candidate
from evaluation.bn_validator import compute_average_cpt_hellinger, compute_average_cpt_kl, compute_average_cpt_rmse, get_best_bn_number, compare_all_cpts


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

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


def get_best_restart_bn(filename=RESTART_FINAL_BN_FILE):

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
                TEST_CSV
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
    filename=BN_ANALYSIS_FILE
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


def create_constrained_train(
    train_csv,
    failures,
    successes,
    ratio=5
):
    """
    Create a temporary training dataset with the specified
    SUCCESS:FAILURE ratio.

    Parameters
    ----------
    train_csv : str
        Path to the original training CSV.
    failures : list
        Failure results returned by initial_run_evaluation().
    successes : list
        Success results returned by initial_run_evaluation().
    ratio : int
        Desired success-to-failure ratio.
    output_csv : str
        Output CSV filename.

    Returns
    -------
    str
        Path to the constrained training CSV.
    """

    output_csv = f"constrained_train_{ratio}to1.csv"

    df = pd.read_csv(train_csv)
    df.columns = df.columns.str.strip()

    # Scenario IDs from the evaluation results
    failure_ids = set(failures["Scenario"])
    success_ids = set(successes["Scenario"])

    # Select corresponding rows from the original training CSV
    failure_rows = df[df["Scenario #"].isin(failure_ids)]
    success_rows = df[df["Scenario #"].isin(success_ids)]

    # Determine the maximum dataset satisfying the desired ratio
    n_failure = min(len(failure_rows), len(success_rows) // ratio)
    n_success = n_failure * ratio

    failure_rows = failure_rows.sample(
        n=n_failure,
        random_state=42
    )

    success_rows = success_rows.sample(
        n=n_success,
        random_state=42
    )

    constrained_df = (
        pd.concat([success_rows, failure_rows], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    constrained_df.to_csv(output_csv, index=False)

    print(
        f"Created {output_csv}: "
        f"{len(success_rows)} successes, "
        f"{len(failure_rows)} failures "
        f"(ratio {ratio}:1)"
    )

    return output_csv


# ----------------------------------------
# Global variables
# ----------------------------------------

train_csv = TRAIN_CSV

TARGET_NODES_FOR_VALIDATION = {
        "Network_Manipulation",
        "Physical_Anomaly",
        "Program_Anomaly",
        "Execution_Integrity",
        "Deviation_in_Response",
        "Deviation_in_Dispatch",
        "Root_Causes",
    }


EXPECTED_CHANGED_CPTS = {
        "Execution_Integrity",
        "Root_Causes"
    }


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--SFR",
    type=int,
    default=None,
    help="Success:Failure ratio (e.g., 8). Omit for normal experiment."
)

args = parser.parse_args()

use_ratio_constraint = args.SFR is not None
SFR = args.SFR

# ----------------------------------------
# Create constrained training set
# ----------------------------------------

flawed_bn = safe_json_loads(FLAWED_BN_FILE)

if use_ratio_constraint:
    failures, successes, flawed_accuracy, _ = initial_run_evaluation(flawed_bn, TRAIN_CSV)
    working_train_csv = create_constrained_train(
            train_csv=TRAIN_CSV,
            failures=failures,
            successes=successes,
            ratio=SFR
    )

    print(pd.read_csv(TRAIN_CSV).shape, pd.read_csv(working_train_csv).shape)
    train_csv = working_train_csv ### -------------------------- CRUCIAL step for ration constrained experiments
    print(pd.read_csv(train_csv).shape, pd.read_csv(working_train_csv).shape)


# -----------------------------
# CLEAR OLD FILES
# -----------------------------

clear_files([
    BN_ANALYSIS_FILE,
    PROPOSED_BN_FILE,
    DANGER_REPORT_FILE,
    FAILURE_PARAMETER_FILE,
    RESTART_FINAL_BN_FILE,
])


restart_count = 0
flawed_bn = safe_json_loads(read_file(FLAWED_BN_FILE))


# ----------------------------------------
# Main orchestration loop
# ----------------------------------------

while restart_count < MAX_RESTARTS:

    print(f"\n==============================")
    print(f"PIPELINE RESTART #{restart_count}")
    print(f"==============================")
    
    current_temperature = BASE_TEMPERATURE + 0.1 * restart_count

    # -----------------------------
    # CLEAR OLD FILES
    # -----------------------------
    clear_files([
        BN_ANALYSIS_FILE,
        PROPOSED_BN_FILE,
        DANGER_REPORT_FILE,
        FAILURE_PARAMETER_FILE
    ])

    # -----------------------------
    # STEP 1: INITIAL BN
    # -----------------------------
    bn_number = 0

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

    initial_accuracy_threshold = (
        flawed_accuracy +
        INITIAL_IMPROVEMENT_RATIO * (1.0 - flawed_accuracy)
    )

    print(f"Flawed BN Accuracy: {100*flawed_accuracy:.2f}%")
    print(f"Initial Acceptance Threshold: {100*initial_accuracy_threshold:.2f}%")

    for retry in range(MAX_INITIAL_RETRIES):
        
        print(f"\nInitial Generation Attempt {retry}")

        if retry > 0:
            remove_analysis(bn_number, BN_ANALYSIS_FILE)

        evaluation_output = run_evaluation(flawed_bn, bn_number=bn_number, temperature=current_temperature, dataset_file=train_csv)
        store_analysis(bn_number, evaluation_output)

        new_bn = generate_and_select_best_candidate(CONTEXT_AGENT_FILE, PROPOSED_BN_FILE, BN_ANALYSIS_FILE, train_csv, temperature=current_temperature) 
        
        failures, successes, accuracy, _ = initial_run_evaluation(new_bn, train_csv)

        print(f"Accuracy = {100*accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_bn = copy.deepcopy(new_bn)

        if accuracy >= initial_accuracy_threshold:
            print("Initial BN accepted.")
            break


    if best_accuracy >= initial_accuracy_threshold:
        print(f"Accepted initial BN ({100*best_accuracy:.2f}%).")
    else:
        print(
            f"Threshold ({100*initial_accuracy_threshold:.2f}%) "
            f"not reached after {MAX_INITIAL_RETRIES} attempts. "
            f"Using best generated BN ({100*best_accuracy:.2f}%)."
        )
    
    
    remove_analysis(bn_number, BN_ANALYSIS_FILE)
    bn_number += 1 
    store_new_bn(bn_number, best_bn)

    print("\nInitial BN generated")

    
    # -----------------------------
    # STEP 2: CONTINUOUS ITERATIVE REFINEMENT
    # -----------------------------

    solved = False
    i = 1


    while i <= MAX_ITER:

        print(f"\n===== ITERATION {i+1} =====")

        best_retry_bn = None
        best_retry_accuracy = float("-inf")

        ### ------------------------------
        # Evaluate
        ### ------------------------------
        prev_bn_json = find_proposed_bn(
            bn_number,
            PROPOSED_BN_FILE
        )

        ### ------------------------------
        # Initial evaluation to get failures for reflexion and analysis
        ### ------------------------------
        failures, successes, accuracy, results = initial_run_evaluation(
            prev_bn_json, train_csv
        )

        evaluation_output = run_evaluation(prev_bn_json, bn_number=bn_number, temperature=current_temperature, dataset_file=train_csv)
        
        store_analysis(bn_number, evaluation_output)

        new_bn = generate_and_select_best_candidate(
                CONTEXT_AGENT_FILE,
                PROPOSED_BN_FILE,
                BN_ANALYSIS_FILE,
                train_csv,
                temperature=current_temperature
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
            remove_bn(bn_number, PROPOSED_BN_FILE)

            new_bn = generate_and_select_best_candidate(
                CONTEXT_AGENT_FILE,
                PROPOSED_BN_FILE,
                BN_ANALYSIS_FILE,
                train_csv,
                temperature=current_temperature
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
                remove_bn(bn_number, PROPOSED_BN_FILE)

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
    # STEP 3: VALIDATION
    # -----------------------------
    
    best_bn_number, best_bn_accuracy = get_best_bn_number(PROPOSED_BN_FILE, train_csv=train_csv)
    print("\n\nBest BN Number:", best_bn_number)
    print("Best BN Accuracy:", best_bn_accuracy)

    final_output = compare_all_cpts(bn_number=best_bn_number, expected_changed_cpts=EXPECTED_CHANGED_CPTS)
    print(final_output)

    gt_bn = normalize_bn(read_json(GROUND_TRUTH_BN_FILE))
    best_bn = get_bn(PROPOSED_BN_FILE, bn_number=best_bn_number)
    nor_best_bn =normalize_bn(best_bn)

    compute_average_cpt_kl(gt_bn, nor_best_bn, target_nodes=TARGET_NODES_FOR_VALIDATION)
    compute_average_cpt_rmse(gt_bn, nor_best_bn, target_nodes=TARGET_NODES_FOR_VALIDATION)
    compute_average_cpt_hellinger(gt_bn, nor_best_bn, target_nodes=TARGET_NODES_FOR_VALIDATION)

    store_restart_final_bn(
        restart_count=restart_count,
        best_bn_number=best_bn_number,
        best_bn_accuracy=best_bn_accuracy,
        best_bn=best_bn,
        filename=RESTART_FINAL_BN_FILE
    )

    print("###------------------------------###")
    print(f"\nRestart {restart_count} finished.")
    print("###------------------------------###")
    
    restart_count += 1


### -----------------------------
# FINAL EVALUATION (on TEST set)
### -----------------------------

best_restart, best_restart_bn, accuracy, failures = (
    get_best_restart_bn()
)

print("\n===================================")
print("BEST RESTART")
print("===================================")

print("Restart:", best_restart)
print("Accuracy:", accuracy)
print("Failures:", failures)

final_output = compare_all_cpts(bn_number=None, expected_changed_cpts=EXPECTED_CHANGED_CPTS, passed_bn=best_restart_bn)
print(final_output)

gt_bn = normalize_bn(read_json(GROUND_TRUTH_BN_FILE))
nor_best_bn = normalize_bn(best_restart_bn)

compute_average_cpt_kl(gt_bn, nor_best_bn, target_nodes=TARGET_NODES_FOR_VALIDATION)
compute_average_cpt_rmse(gt_bn, nor_best_bn, target_nodes=TARGET_NODES_FOR_VALIDATION)
compute_average_cpt_hellinger(gt_bn, nor_best_bn, target_nodes=TARGET_NODES_FOR_VALIDATION)

print("###------------------------------###")
print("\nPipeline finished.")
print("###------------------------------###")
