from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import json

client = OpenAI(api_key="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")

def llm(prompt, temperature=0.2, max_tokens=4000):
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
    
full_context = read_file("context_agent.txt")
flawed_bn = read_file("flawed_BN_0.json")
original_success_report = read_file("merged_flawed_success_train.json")
original_failure_report = read_file("merged_flawed_failure_train.json")
prompt_template_text = read_file("ref_prompt.txt")

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
                    "failure_count": record.get("failure_count"),
                    "success_count": record.get("success_count"),
                    "failure_scenarios_text": record.get("failure_scenarios_text"),
                    "success_scenarios_text": record.get("success_scenarios_text"),
                    "cpt_danger_report": record.get("cpt_danger_report"),
                    # "analysis": record.get("analysis")
                }

                records.append(filtered_record)

            except:
                continue

    # Keep only recent records
    records = records[-max_records:]

    return records

def format_analysis_memory(records):
    formatted = []

    for r in records:
        cpt_report = r.get("cpt_danger_report", {})

        if isinstance(cpt_report, str):
            try:
                cpt_report = json.loads(cpt_report)
            except Exception:
                cpt_report = {}

        formatted.append(
            f"""
BN #{r.get('bn_number', 'unknown')}:

FAILURE COUNT:
{r.get('failure_count', 'unknown')}

SUCCESS COUNT:
{r.get('success_count', 'unknown')}

FAILURE SCENARIOS:
{r.get('failure_scenarios_text', 'No failure scenarios available.')}

SUCCESS SCENARIOS:
{r.get('success_scenarios_text', 'No success scenarios available.')}

REPORTED RISK LEVEL:
{cpt_report.get('reported_risk_level', 'unknown')}

DANGEROUS CPTS:
{json.dumps(cpt_report.get('dangerous_cpts', []), indent=2)}

OVERALL SUMMARY:
{cpt_report.get('overall_summary', 'No overall summary available.')}
"""
        )

    return "\n\n---\n\n".join(formatted)

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

def generate_refined_bn(
    full_context,
    flawed_bn,
    original_success_report,
    original_failure_report,
    last_bn_number=1,
    proposed_bn_filename=proposed_bn_filename,
    bn_analysis_filename=bn_analysis_filename,
    max_records=max_records
):
    analysis_records = read_analysis_memory(bn_analysis_filename, max_records)
    analysis_memory = format_analysis_memory(analysis_records)

    bn_records = read_bn_memory(proposed_bn_filename, max_records)
    previous_bns = format_bn_memory(bn_records)

    prompt_gen_template = PromptTemplate(
        input_variables=[
            "full_context",
            "flawed_bn",
            "original_success_report",
            "original_failure_report",
            "previous_bns",
            "analysis_memory"
        ],
        template=prompt_template_text
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        flawed_bn=flawed_bn,
        original_success_report=original_success_report,
        original_failure_report=original_failure_report,
        previous_bns=previous_bns,
        analysis_memory=analysis_memory
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
    last_bn_number = 1

    bn_new = generate_refined_bn(
        full_context=full_context,
        flawed_bn=flawed_bn,
        original_success_report=original_success_report,
        original_failure_report=original_failure_report,
        last_bn_number=last_bn_number,
        proposed_bn_filename=proposed_bn_filename,
        bn_analysis_filename=bn_analysis_filename,
        max_records=max_records
    )

    print(bn_new)
    bn_new = json.loads(bn_new)
    new_bn_number = last_bn_number + 1

    store_new_bn(new_bn_number, bn_new)
