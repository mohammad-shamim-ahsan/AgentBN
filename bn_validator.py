import json
from collections import Counter

from automatic_bn_reasoning_old import run_evaluation

GT_FILE = "BN_gt.json"
PROPOSED_FILE = "last_proposed_bn.jsonl"
OUT_FILE = "cpt_comparison_analysis.json"

# -----------------------------
# FILE READERS
# -----------------------------
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bn(path, bn_number=None):
    last_record = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if bn_number is not None and record.get("bn_number") == bn_number:
                return record["bn"]

            last_record = record

    if bn_number is not None:
        raise ValueError(f"BN #{bn_number} not found.")

    if last_record is None:
        raise ValueError("No BN records found.")

    return last_record["bn"]


def normalize_bn(bn_obj):
    if "bn" in bn_obj and "nodes" in bn_obj["bn"]:
        return {n["name"]: n for n in bn_obj["bn"]["nodes"]}

    if "nodes" in bn_obj:
        return {n["name"]: n for n in bn_obj["nodes"]}

    return bn_obj


def get_best_bn_number(filename="last_proposed_bn.jsonl", train_csv="combined_train_scenarios.csv"):
    best_bn_number = None
    best_bn_accuracy = None
    best_failure_count = float("inf")

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            bn_number = record.get("bn_number")
            bn_json = record.get("bn")

            if not bn_json:
                continue

            failures, successes, accuracy, results = run_evaluation(
                bn_json, train_csv
            )

            failure_count = len(failures)

            if failure_count <= best_failure_count:
                best_failure_count = failure_count
                best_bn_number = bn_number
                best_bn_accuracy = accuracy

    return best_bn_number, best_bn_accuracy


# -----------------------------
# MAIN COMPARISON
# -----------------------------
def cpt_values_identical(gt_node, proposed_node, tol=1e-9):
    gt_values = gt_node.get("cpt", {}).get("values")
    proposed_values = proposed_node.get("cpt", {}).get("values")

    if gt_values is None or proposed_values is None:
        return False

    if len(gt_values) != len(proposed_values):
        return False

    for gt_row, proposed_row in zip(gt_values, proposed_values):
        if len(gt_row) != len(proposed_row):
            return False

        for g, p in zip(gt_row, proposed_row):
            if abs(g - p) > tol:
                return False

    return True


def evaluate_agent_cpt_change_policy(
    all_cpts,
    expected_changed_cpts,
    actual_changed_cpts
):
    all_cpts = set(all_cpts)
    expected_changed_cpts = set(expected_changed_cpts)
    actual_changed_cpts = set(actual_changed_cpts)

    matched_expected = actual_changed_cpts & expected_changed_cpts
    unexpected_changed = actual_changed_cpts - expected_changed_cpts
    missed_expected = expected_changed_cpts - actual_changed_cpts

    unflawed_cpts = all_cpts - expected_changed_cpts
    untouched_unflawed = unflawed_cpts - actual_changed_cpts

    expected_change_recall = (
        len(matched_expected) / len(expected_changed_cpts)
        if expected_changed_cpts else 1.0
    )

    unexpected_change_rate = (
        len(unexpected_changed) / len(unflawed_cpts)
        if unflawed_cpts else 0.0
    )

    unflawed_preservation_rate = (
        len(untouched_unflawed) / len(unflawed_cpts)
        if unflawed_cpts else 1.0
    )

    if expected_change_recall == 1.0 and unexpected_change_rate == 0.0:
        verdict = "excellent"

    elif (
        expected_change_recall == 1.0
        and unflawed_preservation_rate >= 0.80
        and unexpected_change_rate <= 0.20
    ):
        verdict = "very_good"

    elif expected_change_recall > 0 and unexpected_change_rate <= 0.20:
        verdict = "good"

    elif expected_change_recall > 0 and unexpected_change_rate <= 0.50:
        verdict = "fair"

    else:
        verdict = "poor"

    return {
        "verdict": verdict,

        "total_cpts": len(all_cpts),

        "expected_changed_count": len(expected_changed_cpts),
        "expected_changed_cpts": sorted(expected_changed_cpts),

        "actual_changed_count": len(actual_changed_cpts),
        "actual_changed_cpts": sorted(actual_changed_cpts),

        "matched_expected_count": len(matched_expected),
        "matched_expected_cpts": sorted(matched_expected),

        "missed_expected_count": len(missed_expected),
        "missed_expected_cpts": sorted(missed_expected),

        "unexpected_changed_count": len(unexpected_changed),
        "unexpected_changed_cpts": sorted(unexpected_changed),

        "unflawed_cpt_count": len(unflawed_cpts),
        "untouched_unflawed_count": len(untouched_unflawed),
        "untouched_unflawed_cpts": sorted(untouched_unflawed),

        "expected_change_recall": round(expected_change_recall, 4),
        "unexpected_change_rate": round(unexpected_change_rate, 4),
        "unflawed_preservation_rate": round(unflawed_preservation_rate, 4),
    }


def compare_all_cpts(bn_number=None):
    gt_bn = normalize_bn(read_json(GT_FILE))
    proposed_bn = normalize_bn(get_bn(PROPOSED_FILE, bn_number=bn_number))

    EXPECTED_CHANGED_CPTS = {
        "Execution_Integrity",
        "Root_Causes"
    }

    non_identical_cpts = []
    missing_in_proposed = []

    for node_name in sorted(gt_bn.keys()):
        if node_name not in proposed_bn:
            missing_in_proposed.append(node_name)
            continue

        if not cpt_values_identical(gt_bn[node_name], proposed_bn[node_name]):
            non_identical_cpts.append(node_name)

    agent_change_report = evaluate_agent_cpt_change_policy(
        all_cpts=set(gt_bn.keys()),
        expected_changed_cpts=EXPECTED_CHANGED_CPTS,
        actual_changed_cpts=set(non_identical_cpts)
    )

    final_output = {
        "bn_number": bn_number,

        "non_identical_cpt_count": len(non_identical_cpts),
        "non_identical_cpts": sorted(non_identical_cpts),

        "missing_in_proposed": sorted(missing_in_proposed),

        "agent_change_report": agent_change_report
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved CPT identity comparison to {OUT_FILE}")
    print("BN number:", bn_number)
    print("Agent change verdict:", agent_change_report["verdict"])
    print("Expected changed CPTs:", agent_change_report["expected_changed_cpts"])
    print("Actual changed CPTs:", agent_change_report["actual_changed_cpts"])
    print("Unexpected changed CPTs:", agent_change_report["unexpected_changed_cpts"])

    return final_output


###------------------------------
import numpy as np

EPS = 1e-12


def kl_divergence(p, q):
    """
    Compute KL(P || Q) for two discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)

    return np.sum(p * np.log(p / q))


def load_bn(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def build_node_dict(bn_json):
    return {node["name"]: node for node in bn_json["nodes"]}


def compute_average_cpt_kl(gt_bn, prop_bn, target_nodes=None):

    total_kl = 0.0
    total_columns = 0

    print("=" * 70)

    for name in sorted(gt_bn.keys()):

        if target_nodes and name not in target_nodes:
            continue

        if name not in prop_bn:
            raise ValueError(f"Missing node '{name}'.")

        gt_values = np.asarray(gt_bn[name]["cpt"]["values"], dtype=float)
        prop_values = np.asarray(prop_bn[name]["cpt"]["values"], dtype=float)

        if gt_values.shape != prop_values.shape:
            raise ValueError(
                f"CPT shape mismatch for {name}: "
                f"{gt_values.shape} vs {prop_values.shape}"
            )

        node_kl = 0.0

        for col in range(gt_values.shape[1]):
            kl = kl_divergence(
                gt_values[:, col],
                prop_values[:, col]
            )

            node_kl += kl
            total_kl += kl
            total_columns += 1

        print(f"{name:<30} {node_kl / gt_values.shape[1]:.8f}")

    print("=" * 70)

    avg_kl = total_kl / total_columns

    print(f"Overall Average CPT KL = {avg_kl:.8f}")

    return avg_kl

def compute_average_cpt_rmse(gt_bn, prop_bn, target_nodes=None):

    total_squared_error = 0.0
    total_parameters = 0

    print("=" * 70)

    for name in sorted(gt_bn.keys()):

        if target_nodes and name not in target_nodes:
            continue

        gt_values = np.asarray(gt_bn[name]["cpt"]["values"], dtype=float)
        prop_values = np.asarray(prop_bn[name]["cpt"]["values"], dtype=float)

        if gt_values.shape != prop_values.shape:
            raise ValueError(f"CPT shape mismatch for {name}")

        node_rmse = np.sqrt(np.mean((gt_values - prop_values) ** 2))

        total_squared_error += np.sum((gt_values - prop_values) ** 2)
        total_parameters += gt_values.size

        print(f"{name:<30} RMSE = {node_rmse:.8f}")

    print("=" * 70)

    overall_rmse = np.sqrt(total_squared_error / total_parameters)

    print(f"Overall CPT RMSE = {overall_rmse:.8f}")

    return overall_rmse

def hellinger_distance(p, q):

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def compute_average_cpt_hellinger(gt_bn, prop_bn, target_nodes=None):

    total = 0.0
    total_columns = 0

    print("=" * 70)

    for name in sorted(gt_bn.keys()):

        if target_nodes and name not in target_nodes:
            continue

        gt_values = np.asarray(gt_bn[name]["cpt"]["values"], dtype=float)
        prop_values = np.asarray(prop_bn[name]["cpt"]["values"], dtype=float)

        node_total = 0.0

        for col in range(gt_values.shape[1]):

            h = hellinger_distance(
                gt_values[:, col],
                prop_values[:, col]
            )

            node_total += h
            total += h
            total_columns += 1

        print(f"{name:<30} Hellinger = {node_total / gt_values.shape[1]:.8f}")

    print("=" * 70)

    overall = total / total_columns

    print(f"Overall Average Hellinger = {overall:.8f}")

    return overall
###------------------------------


### ------------------------------
if __name__ == "__main__":

    train_csv="combined_train_scenarios.csv"

    best_bn_number, best_bn_accuracy = get_best_bn_number(
        "last_proposed_bn.jsonl", train_csv=train_csv
    )

    print("Best BN Number:", best_bn_number)
    print("Best BN Accuracy:", best_bn_accuracy)

    # final_output = compare_all_cpts(bn_number=best_bn_number)
    # print(final_output)

    gt_bn = normalize_bn(read_json(GT_FILE))
    prop_bn = normalize_bn(get_bn(PROPOSED_FILE, bn_number=best_bn_number))

    TARGET_NODES = {
        "Network_Manipulation",
        "Physical_Anomaly",
        "Program_Anomaly",
        "Execution_Integrity",
        "Deviation_in_Response",
        "Deviation_in_Dispatch",
        "Root_Causes",
    }

    compute_average_cpt_kl(
        gt_bn,
        prop_bn,
        target_nodes=TARGET_NODES
    )

    compute_average_cpt_rmse(
        gt_bn,
        prop_bn,
        target_nodes=TARGET_NODES
    )

    compute_average_cpt_hellinger(
        gt_bn,
        prop_bn,
        target_nodes=TARGET_NODES
    )
