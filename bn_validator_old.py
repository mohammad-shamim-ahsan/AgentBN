from openai import OpenAI
import json

client = OpenAI(api_key="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")

GT_FILE = "BN_gt.json"
PROPOSED_FILE = "last_proposed_bn.jsonl"
OUT_FILE = "cpt_comparison_analysis.json"
MODEL = "gpt-5.4"

def llm(prompt, temperature=0.2, max_tokens=4000):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return response.output[0].content[0].text.strip()

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

            if bn_number is not None:
                if record.get("bn_number") == bn_number:
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

def flatten(values):
    out = []
    for row in values:
        out.extend(row)
    return out

def column_sums(values):
    if not values:
        return []

    rows = len(values)
    cols = len(values[0])

    return [
        sum(values[r][c] for r in range(rows))
        for c in range(cols)
    ]

def compare_cpt_only(gt_node, proposed_node):
    gt_values = gt_node["cpt"]["values"]
    proposed_values = proposed_node["cpt"]["values"]

    gt_flat = flatten(gt_values)
    proposed_flat = flatten(proposed_values)

    diffs = []

    for i, (g, p) in enumerate(zip(gt_flat, proposed_flat)):
        abs_error = abs(g - p)

        if abs_error <= 0.05:
            severity = "close"
        elif abs_error <= 0.20:
            severity = "moderate"
        else:
            severity = "severe"

        diffs.append({
            "parameter_index": i,
            "absolute_error": round(abs_error, 6),
            "signed_error": round(p - g, 6),
            "severity": severity
        })

    return {
        "same_cpt_shape": (
            len(gt_values) == len(proposed_values)
            and all(len(a) == len(b) for a, b in zip(gt_values, proposed_values))
        ),
        "num_ground_truth_parameters": len(gt_flat),
        "num_proposed_parameters": len(proposed_flat),
        "valid_probability_range": all(0 <= x <= 1 for x in proposed_flat),
        "proposed_column_sums": column_sums(proposed_values),
        "columns_sum_to_one": all(
            abs(s - 1.0) <= 1e-6 for s in column_sums(proposed_values)
        ),
        "max_abs_error": round(max((d["absolute_error"] for d in diffs), default=0), 6),
        "mean_abs_error": round(
            sum(d["absolute_error"] for d in diffs) / len(diffs), 6
        ) if diffs else None,
        "close_count": sum(d["severity"] == "close" for d in diffs),
        "moderate_count": sum(d["severity"] == "moderate" for d in diffs),
        "severe_count": sum(d["severity"] == "severe" for d in diffs),
        "parameter_differences": diffs
    }

def make_prompt(node_name, cpt_comparison):
    return f"""
You are comparing only CPT parameters of a proposed Bayesian Network node against the ground-truth node.

Do not discuss structure, edges, semantics, parents, or states unless CPT shape is affected.
Focus only on CPT validity and CPT parameter accuracy.

Node: {node_name}

Computed CPT comparison:
{json.dumps(cpt_comparison, indent=2)}

Use these thresholds:
- close: absolute error <= 0.05
- moderate: 0.05 < absolute error <= 0.20
- severe: absolute error > 0.20

Return valid JSON only with this schema:

{{
  "node": "{node_name}",
  "same_cpt_shape": true,
  "valid_probability_range": true,
  "columns_sum_to_one": true,
  "max_abs_error": 0.0,
  "mean_abs_error": 0.0,
  "close_count": 0,
  "moderate_count": 0,
  "severe_count": 0,
  "cpt_accuracy_assessment": "...",
  "main_cpt_issues": ["..."],
  "verdict": "excellent|good|fair|poor|invalid"
}}
""".strip()

def is_evidence_node(node_name):
    return node_name.startswith("GPTN_") or node_name.startswith("LPTN_")

def compare_all_cpts():
    gt_bn = normalize_bn(read_json(GT_FILE))
    proposed_bn = normalize_bn(get_bn(PROPOSED_FILE, bn_number=8))

    results = {}

    for node_name in sorted(gt_bn.keys()):
        if is_evidence_node(node_name):
            continue

        if node_name not in proposed_bn:
            results[node_name] = {
                "status": "missing_in_proposed",
                "main_cpt_issues": ["Node missing from proposed BN, so CPT cannot be compared."],
                "verdict": "invalid"
            }
            continue

        cpt_comparison = compare_cpt_only(gt_bn[node_name], proposed_bn[node_name])

        prompt = make_prompt(node_name, cpt_comparison)
        raw = llm(prompt)

        try:
            results[node_name] = json.loads(raw)
        except json.JSONDecodeError:
            results[node_name] = {
                "status": "llm_output_invalid_json",
                "raw_output": raw
            }

    overall_prompt = f"""
        You are evaluating the proposed Bayesian Network CPT quality.

        Use only the node-level CPT verdicts below.
        Do not discuss evidence nodes.

        Node-level CPT analyses:
        {json.dumps(results, indent=2)}

        Return valid JSON only:

        {{
            "overall_verdict": "excellent|good|fair|poor|invalid",
            "summary": "...",
            "main_reasons": ["..."],
            "node_verdict_counts": {{
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0,
                "invalid": 0
            }}
        }}
        """.strip()

    raw_overall = llm(overall_prompt)

    try:
        overall_analysis = json.loads(raw_overall)
    except json.JSONDecodeError:
        overall_analysis = {
            "status": "llm_output_invalid_json",
            "raw_output": raw_overall
        }

    final_output = {
        "node_level_cpt_analysis": results,
        "overall_cpt_analysis": overall_analysis
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved CPT-only analysis to {OUT_FILE}")

if __name__ == "__main__":
    compare_all_cpts()
