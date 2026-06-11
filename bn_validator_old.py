import json
from collections import Counter

from automatic_bn_reasoning import run_evaluation

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


# -----------------------------
# CPT HELPERS
# -----------------------------
def is_evidence_node(node_name):
    return node_name.startswith("GPTN_") or node_name.startswith("LPTN_")


def flatten(values):
    return [x for row in values for x in row]


def same_shape(a, b):
    return (
        len(a) == len(b)
        and all(len(row_a) == len(row_b) for row_a, row_b in zip(a, b))
    )


def column_sums(values):
    if not values:
        return []

    rows = len(values)
    cols = len(values[0])

    return [
        round(sum(values[r][c] for r in range(rows)), 6)
        for c in range(cols)
    ]


def classify_error(abs_error):
    close_threshold = 0.05
    moderate_threshold = 0.15

    if abs_error <= close_threshold:
        return "close"
    elif abs_error <= moderate_threshold:
        return "moderate"
    return "severe"


def deterministic_verdict(result):
    if not result["same_cpt_shape"]:
        return "invalid"

    if not result["valid_probability_range"]:
        return "invalid"

    if not result["columns_sum_to_one"]:
        return "invalid"

    total = result.get("num_proposed_parameters", 0)

    if total == 0:
        return "invalid"

    close_count = result.get("close_count", 0)
    moderate_count = result.get("moderate_count", 0)
    severe_count = result.get("severe_count", 0)

    close_weight = 1
    moderate_weight = 3
    severe_weight = 6

    impact_score = (
        close_count * close_weight +
        moderate_count * moderate_weight +
        severe_count * severe_weight
    )

    impact_ratio = impact_score / total

    impact_threshold = 3.0

    if impact_ratio >= impact_threshold:
        return "poor"

    if impact_ratio >= impact_threshold/2:
        return "fair"

    if impact_ratio >= impact_threshold/5:
        return "good"

    return "excellent"


# -----------------------------
# CPT COMPARISON
# -----------------------------
def compare_cpt_only(gt_node, proposed_node):
    gt_values = gt_node["cpt"]["values"]
    proposed_values = proposed_node["cpt"]["values"]

    shape_ok = same_shape(gt_values, proposed_values)

    gt_flat = flatten(gt_values)
    proposed_flat = flatten(proposed_values)

    diffs = []

    if shape_ok:
        pairs = zip(gt_flat, proposed_flat)
    else:
        pairs = zip(gt_flat[:min(len(gt_flat), len(proposed_flat))],
                    proposed_flat[:min(len(gt_flat), len(proposed_flat))])

    for i, (g, p) in enumerate(pairs):
        abs_error = abs(g - p)
        signed_error = p - g

        diffs.append({
            "parameter_index": i,
            "ground_truth_value": round(g, 6),
            "proposed_value": round(p, 6),
            "absolute_error": round(abs_error, 6),
            "signed_error": round(signed_error, 6),
            "severity": classify_error(abs_error)
        })

    valid_probability_range = all(0 <= x <= 1 for x in proposed_flat)

    proposed_sums = column_sums(proposed_values)

    columns_sum_to_one = all(
        abs(s - 1.0) <= 1e-6 for s in proposed_sums
    )

    result = {
        "same_cpt_shape": shape_ok,
        "num_ground_truth_parameters": len(gt_flat),
        "num_proposed_parameters": len(proposed_flat),
        "valid_probability_range": valid_probability_range,
        "proposed_column_sums": proposed_sums,
        "columns_sum_to_one": columns_sum_to_one,
        "max_abs_error": round(max((d["absolute_error"] for d in diffs), default=0), 6),
        "mean_abs_error": round(
            sum(d["absolute_error"] for d in diffs) / len(diffs), 6
        ) if diffs else None,
        "close_count": sum(d["severity"] == "close" for d in diffs),
        "moderate_count": sum(d["severity"] == "moderate" for d in diffs),
        "severe_count": sum(d["severity"] == "severe" for d in diffs),
        "parameter_differences": diffs
    }

    result["verdict"] = deterministic_verdict(result)

    return result


# -----------------------------
# OVERALL SUMMARY
# -----------------------------
def build_overall_analysis(results):
    verdict_counts = Counter(
        r.get("verdict", "invalid")
        for r in results.values()
    )

    total_nodes = len(results)
    total_parameters = sum(
        r.get("num_proposed_parameters", 0)
        for r in results.values()
    )

    total_severe = sum(r.get("severe_count", 0) for r in results.values())
    total_moderate = sum(r.get("moderate_count", 0) for r in results.values())
    total_close = sum(r.get("close_count", 0) for r in results.values())

    invalid_nodes = [
        node for node, r in results.items()
        if r.get("verdict") == "invalid"
    ]

    poor_nodes = [
        node for node, r in results.items()
        if r.get("verdict") == "poor"
    ]

    fair_nodes = [
        node for node, r in results.items()
        if r.get("verdict") == "fair"
    ]

    good_nodes = [
        node for node, r in results.items()
        if r.get("verdict") == "good"
    ]

    excellent_nodes = [
        node for node, r in results.items()
        if r.get("verdict") == "excellent"
    ]

    node_error_counts = {
        node: {
            "verdict": r.get("verdict", "unknown"),
            "close_count": r.get("close_count", 0),
            "moderate_count": r.get("moderate_count", 0),
            "severe_count": r.get("severe_count", 0),
        }
        for node, r in results.items()
    }

    if invalid_nodes:
        overall_verdict = "invalid" # If any node is invalid, the whole BN is invalid
    elif verdict_counts.get("excellent", 0) == total_nodes:
        overall_verdict = "excellent" # All nodes are excellent
    elif verdict_counts.get("poor", 0) == 0 and verdict_counts.get("fair", 0) == 0:
        overall_verdict = "good" # No poor or fair nodes, but not all excellent
    elif verdict_counts.get("poor", 0) == 0:
        overall_verdict = "fair" # No poor nodes, but some fair nodes
    else:
        overall_verdict = "poor" # At least one poor node

    return {
        "overall_verdict": overall_verdict,
        "total_compared_nodes": total_nodes,
        "total_compared_parameters": total_parameters,
        "total_close_parameters": total_close,
        "total_moderate_parameters": total_moderate,
        "total_severe_parameters": total_severe,
        "node_verdict_counts": dict(verdict_counts),
        "invalid_nodes": invalid_nodes,
        "poor_nodes": poor_nodes,
        "fair_nodes": fair_nodes,
        "good_nodes": good_nodes,
        "excellent_nodes": excellent_nodes,
        "node_error_counts": node_error_counts
    }


# -----------------------------
# MAIN COMPARISON
# -----------------------------
def compare_all_cpts(bn_number=None):
    gt_bn = normalize_bn(read_json(GT_FILE))
    proposed_bn = normalize_bn(get_bn(PROPOSED_FILE, bn_number=bn_number))

    results = {}

    for node_name in sorted(gt_bn.keys()):
        if is_evidence_node(node_name):
            continue

        if node_name not in proposed_bn:
            results[node_name] = {
                "node": node_name,
                "status": "missing_in_proposed",
                "same_cpt_shape": False,
                "valid_probability_range": False,
                "columns_sum_to_one": False,
                "verdict": "invalid"
            }
            continue

        comparison = compare_cpt_only(gt_bn[node_name], proposed_bn[node_name])
        comparison["node"] = node_name
        results[node_name] = comparison

    final_output = {
        "bn_number": bn_number,
        "excluded_nodes": "Nodes starting with GPTN_ or LPTN_",
        "node_level_cpt_analysis": results,
        "overall_cpt_analysis": build_overall_analysis(results)
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved deterministic CPT comparison to {OUT_FILE}")
    print("Overall verdict:", final_output["overall_cpt_analysis"]["overall_verdict"])

    return final_output


def get_best_bn_number(filename="last_proposed_bn.jsonl"):
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
                bn_json
            )

            failure_count = len(failures)

            if failure_count <= best_failure_count:
                best_failure_count = failure_count
                best_bn_number = bn_number
                best_bn_accuracy = accuracy

    return best_bn_number, best_bn_accuracy

### ------------------------------
if __name__ == "__main__":
    best_bn_number, best_bn_accuracy = get_best_bn_number(
        "last_proposed_bn.jsonl"
    )
    print("Best BN Number:", best_bn_number)
    print("Best BN Accuracy:", best_bn_accuracy)

    compare_all_cpts(bn_number=best_bn_number)
