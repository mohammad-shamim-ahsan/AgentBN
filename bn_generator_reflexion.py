from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import json

client = OpenAI(api_key="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")

def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    return response.output[0].content[0].text.strip()

###-----------------------------
bn_analysis_filename="bn_analysis.json"
max_records=3
proposed_bn_filename="last_proposed_bn.jsonl"

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
    
full_context = read_file("context_gen_agent.txt")
scenario_dataset = read_file("final_validated_dataset.csv")
failure_report = read_file("flawed_failure_results.json")
prompt_template_text = read_file("ref_prompt.txt")
CONSTRAINTS = {
    "correctness": [
        "All CPT entries must be valid probabilities (0 ≤ p ≤ 1)",
        "Each CPT column must sum to 1 (±1e-6 tolerance)"
    ],
    "formatting": [
        "Output must be valid JSON",
        "Must follow schema: {node: {parents: [...], cpt: {...}}}"
    ]
}

### -----------------------------
def read_analysis_memory(filename=bn_analysis_filename, max_records=max_records):
    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)

                # Keep only useful fields for refinement
                filtered_record = {
                    "bn_number": record.get("bn_number"),
                    "failure_text": record.get("failure_text"),
                    "success_text": record.get("success_text"),
                    "analysis": record.get("analysis")
                }

                records.append(filtered_record)

            except:
                continue

    # Keep only recent records
    records = records[-max_records:]

    return records

# def format_analysis_memory(records):
#     formatted = []

#     for r in records:
#         formatted.append(
#             f"""
# BN #{r['bn_number']}:

# SUMMARY:
# {json.loads(r['analysis'])['summary']}

# CPT ISSUES:
# {json.dumps(json.loads(r['analysis'])['cpt_issues'], indent=2)}

# FIXES:
# {json.dumps(json.loads(r['analysis'])['fixes'], indent=2)}
# """
#         )

#     return "\n\n---\n\n".join(formatted)

def format_analysis_memory(records):
    formatted = []

    for r in records:

        cpt_report = r.get("cpt_danger_report", {})

        # handle stringified JSON if needed
        if isinstance(cpt_report, str):
            try:
                cpt_report = json.loads(cpt_report)
            except Exception:
                cpt_report = {}

        overall_summary = cpt_report.get(
            "overall_summary",
            "No overall summary available."
        )

        dangerous_cpts = cpt_report.get(
            "dangerous_cpts",
            []
        )

        formatted.append(
            f"""
BN #{r.get('bn_number', 'unknown')}:

FAILURE COUNT:
{r.get('failure_count', 'unknown')}

SUCCESS COUNT:
{r.get('success_count', 'unknown')}

OVERALL SUMMARY:
{overall_summary}

DANGEROUS CPTS:
{json.dumps(dangerous_cpts, indent=2)}
"""
        )

    return "\n\n---\n\n".join(formatted)

# def read_last_bn(filename=proposed_bn_filename, bn_number=1):
#     try:
#         with open(filename, "r", encoding="utf-8") as f:
#             for line in f:
#                 record = json.loads(line)
#                 if record["bn_number"] == bn_number:
#                     return record["bn"]
#     except:
#         return None
    
# def format_last_bn(bn_json):
#     if not bn_json:
#         return "No previous BN available."

#     return json.dumps(bn_json, indent=2)

def read_bn_memory(filename=proposed_bn_filename, max_records=max_records):
    records = []

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)

                    filtered_record = {
                        "bn_number": record.get("bn_number"),
                        "bn": record.get("bn")
                    }

                    records.append(filtered_record)

                except:
                    continue

    except:
        return []

    # Keep only recent records
    records = records[-max_records:]

    return records


def format_bn_memory(records):
    if not records:
        return "No previous BNs available."

    formatted = []

    for r in records:
        formatted.append(
            f"""
BN #{r['bn_number']}:

{json.dumps(r['bn'], indent=2)}
"""
        )

    return "\n\n---\n\n".join(formatted)

def format_constraints():
    return "\n".join(
        f"{k.upper()}:\n- " + "\n- ".join(v)
        for k, v in CONSTRAINTS.items()
    )

def generate_refined_bn(full_context, scenario_dataset, failure_report, last_bn_number=1, proposed_bn_filename=proposed_bn_filename, bn_analysis_filename=bn_analysis_filename, max_records=max_records):
    analysis_records = read_analysis_memory(bn_analysis_filename, max_records)
    analysis_memory = format_analysis_memory(analysis_records)

    # last_bn = read_last_bn(proposed_bn_filename, bn_number=last_bn_number)
    # previous_bn = format_last_bn(last_bn)

    # prompt_gen_template = PromptTemplate(
    #     input_variables=["full_context", "analysis_memory", "previous_bn"],
    #     template=prompt_template_text
    # )

    # prompt = prompt_gen_template.format(
    #     full_context=full_context,
    #     analysis_memory=analysis_memory,
    #     previous_bn=previous_bn,
    #     constraints=format_constraints()
    # )

    bn_records = read_bn_memory(
        proposed_bn_filename,
        max_records
    )

    previous_bns = format_bn_memory(bn_records)

    prompt_gen_template = PromptTemplate(
        input_variables=[
            "full_context",
            "scenario_dataset",
            "failure_report",
            "analysis_memory",
            "previous_bns"
        ],
        template=prompt_template_text
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        scenario_dataset=scenario_dataset,
        failure_report=failure_report,
        analysis_memory=analysis_memory,
        previous_bns=previous_bns,
        constraints=format_constraints()
    )

    return llm(prompt)

def store_new_bn(bn_number, bn_new):
    record = {
        "bn_number": bn_number,
        "bn": bn_new
    }

    with open("last_proposed_bn.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    last_bn_number = 7
    bn_new = generate_refined_bn(full_context, scenario_dataset, failure_report, last_bn_number, proposed_bn_filename, bn_analysis_filename, max_records)
    print(bn_new)
    
    bn_new = json.loads(bn_new)
    new_bn_number = 8
    store_new_bn(new_bn_number, bn_new)
