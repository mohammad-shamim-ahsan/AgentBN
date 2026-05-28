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

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

proposed_bn_filename="last_proposed_bn.jsonl"
full_context = read_file("context_gen_agent.txt")
scenario_dataset = read_file("final_validated_dataset.csv")
failure_report = read_file("flawed_failure_results.json")
prompt_template_text = read_file("gen_prompt.txt")

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

# -------------------------------
# 1️⃣ Draft Stage (generation only)
# -------------------------------

def draft_model(full_context, scenario_dataset, failure_report, prompt_template_text):
    prompt_gen_template = PromptTemplate(
        input_variables=[
            "full_context",
            "scenario_dataset",
            "failure_report",
        ],
        template=prompt_template_text
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        scenario_dataset=scenario_dataset,
        failure_report=failure_report,
        constraints=format_constraints()
    )

    return llm(prompt)

def generate_bn(full_context, scenario_dataset, failure_report, prompt_template_text):
    # 1️⃣ Draft
    draft = draft_model(full_context, scenario_dataset, failure_report, prompt_template_text)
    return draft

def store_bn_proposal(bn_new, bn_number, filename=proposed_bn_filename):
    record = {
        "bn_number": bn_number,
        "bn": bn_new
    }

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

### -------------------------------
if __name__ == "__main__":
    bn_proposal = generate_bn(full_context, scenario_dataset, failure_report, prompt_template_text)
    print(bn_proposal)
    bn_json = json.loads(bn_proposal)
    store_bn_proposal(bn_json, bn_number=1)
