from langchain.prompts import PromptTemplate
from openai import OpenAI

client = OpenAI(api_key="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")

import json

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

full_context = read_file("Context for Gen-Agent.txt")

# -------------------------------
# 3️⃣ Prompt Template (from file)
# -------------------------------
prompt_template_text = read_file("gen_prompt_A_2.txt")

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
    ],
    "structure": [
        "All nodes must include a CPT",
        "Parent-child relationships must be consistent"
    ],
    "redundancy": [
        "No duplicate CPT entries",
        "No repeated nodes or repeated configurations"
    ],
    "completeness": [
        "All variables in the Bayesian network must have CPTs",
        "All parent configurations must be fully covered"
    ]
}

def format_constraints():
    return "\n".join(
        f"{k.upper()}:\n- " + "\n- ".join(v)
        for k, v in CONSTRAINTS.items()
    )

# -------------------------------
# 1️⃣ Draft Model
# -------------------------------
def draft_model(full_context):
    prompt = prompt_gen_template.format(
        full_context=full_context,
        constraints=format_constraints()
    )
    return llm(prompt)

# -------------------------------
# 2️⃣ Critic Model
# -------------------------------
def critic_model(draft):
    prompt = f"""
You are a STRICT FAILURE DETECTOR for Bayesian Networks.

Return ONLY valid JSON. No markdown. No explanations.

---

CONSTRAINTS:
{format_constraints()}

---

INPUT:
{draft}

---

RULES:
1. Only report violations
2. Do NOT include correct nodes
3. If no violations exist, return empty list

---

OUTPUT FORMAT:

{{
  "violations": [
    {{
      "node": "string",
      "issue": "string"
    }}
  ]
}}

OR

{{ "violations": [] }}
"""
    return llm(prompt, temperature=0.0)

# -------------------------------
# 3️⃣ Safe Violation Parsing
# -------------------------------
def extract_violations(critique_raw):
    try:
        data = json.loads(critique_raw)
        v = data.get("violations", [])
        return v if isinstance(v, list) else []
    except:
        return [{"node": "UNKNOWN", "issue": "Malformed critic output"}]

# -------------------------------
# 4️⃣ Refiner Model
# -------------------------------
def refiner_model(draft, violations):
    prompt = f"""
You are a CORRECTION ENGINE for Bayesian Networks.

Fix ONLY the listed violations.

---

ORIGINAL OUTPUT:
{draft}

---

VIOLATIONS:
{json.dumps(violations, indent=2)}

---

HARD RULES:
- Fix ONLY listed violations
- Do NOT modify correct parts
- Preserve structure
- Ensure CPT columns sum to 1
- Output MUST be valid JSON

---

OUTPUT:
Return FULL corrected JSON only.
"""
    return llm(prompt, temperature=0.2)

# -------------------------------
# 5️⃣ JSON Validator (Hard Gate)
# -------------------------------
def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False

# -------------------------------
# 6️⃣ Orchestration (ITERATIVE HARD GATE)
# -------------------------------
def subtask_A(full_context, max_iters=3):

    draft = draft_model(full_context)

    print("========== Initial Draft ==========")
    print(draft)

    for i in range(max_iters):

        print(f"\n====== Iteration {i+1} ======")

        # Critic
        critique_raw = critic_model(draft)
        print("Critic Output:")
        print(critique_raw)

        violations = extract_violations(critique_raw)
        print("Parsed Violations:", violations)

        # ✅ STOP CONDITION
        if len(violations) == 0:
            print("✅ No violations. Stopping.")
            break

        # Refine
        refined = refiner_model(draft, violations)

        print("Refined Output:")
        print(refined)

        # 🚨 Safety: ensure valid JSON
        if not is_valid_json(refined):
            print("⚠️ Refiner produced invalid JSON. Stopping.")
            break

        # 🚨 Stability check (no change)
        if refined.strip() == draft.strip():
            print("⚠️ No change after refinement. Stopping.")
            break

        draft = refined  # iterate

    return draft

# -------------------------------
# 7️⃣ Run
# -------------------------------
bn_proposal = subtask_A(full_context)
print("\n================ Final BN Proposal ================")
print(bn_proposal)

