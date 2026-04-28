from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import json

client = OpenAI(api_key="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")

# -------------------------------
# 🔌 Unified LLM Call
# -------------------------------
def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    return response.output[0].content[0].text.strip()

def read_analysis_memory(filename="bn_analysis.json", max_records=5):
    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue

    # Keep only most recent N records (important to avoid prompt explosion)
    records = records[-max_records:]

    return records

def format_analysis_memory(records):
    formatted = []

    for r in records:
        formatted.append(
            f"""
BN #{r['bn_number']}:

SUMMARY:
{json.loads(r['analysis'])['summary']}

CPT ISSUES:
{json.dumps(json.loads(r['analysis'])['cpt_issues'], indent=2)}

FIXES:
{json.dumps(json.loads(r['analysis'])['fixes'], indent=2)}
"""
        )

    return "\n\n---\n\n".join(formatted)

def read_last_bn(filename="proposed_bn.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None
    
def format_last_bn(bn_json):
    if not bn_json:
        return "No previous BN available."

    return json.dumps(bn_json, indent=2)

# -------------------------------
# 2️⃣ Read Single Context File
# -------------------------------
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

full_context = read_file("context_gen_agent.txt")
prompt_template_text = read_file("ref_prompt.txt")

prompt_gen_template = PromptTemplate(
    input_variables=["full_context", "analysis_memory", "previous_bn"],
    template=prompt_template_text
)

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

def format_constraints():
    return "\n".join(
        f"{k.upper()}:\n- " + "\n- ".join(v)
        for k, v in CONSTRAINTS.items()
    )

def draft_model(full_context):
    analysis_records = read_analysis_memory()
    analysis_memory = format_analysis_memory(analysis_records)

    last_bn = read_last_bn()
    previous_bn = format_last_bn(last_bn)

    prompt = prompt_gen_template.format(
        full_context=full_context,
        analysis_memory=analysis_memory,
        previous_bn=previous_bn,
        constraints=format_constraints()
    )

    return llm(prompt)

bn_new = draft_model(full_context)
print(bn_new)

bn_new = json.loads(bn_new)
bn_number = 2
record = {
    "bn_number": bn_number,
    "bn": bn_new
}

with open("last_proposed_bn.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
