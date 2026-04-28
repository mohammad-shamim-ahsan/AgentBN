from langchain_core.prompts import PromptTemplate
from openai import OpenAI

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

# -------------------------------
# 2️⃣ Read Single Context File
# -------------------------------
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

full_context = read_file("context_gen_agent.txt")

# -------------------------------
# 3️⃣ Prompt Template (from file)
# -------------------------------
prompt_template_text = read_file("gen_prompt.txt")

prompt_gen_template = PromptTemplate(
    input_variables=["full_context"],
    template=prompt_template_text
)

# -------------------------------
# 0️⃣ Explicit Constraints (single source of truth)
# -------------------------------

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

def draft_model(full_context):
    prompt = prompt_gen_template.format(
        full_context=full_context,
        constraints=format_constraints()
    )

    return llm(prompt)

# -------------------------------
# 4️⃣ Orchestration (HARD GATE ADDED)
# -------------------------------

def subtask_A(full_context):
    # 1️⃣ Draft
    draft = draft_model(full_context)
    return draft

# -------------------------------
# 5️⃣ Run
# -------------------------------
bn_proposal = subtask_A(full_context)
print("================== Final BN Proposal ==========================")
print(bn_proposal)

import json

bn_json = json.loads(bn_proposal)

with open("proposed_bn.json", "w") as f:
    json.dump(bn_json, f, indent=2)
