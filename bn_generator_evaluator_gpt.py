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

###
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# -----------------------------
# Build structure
# -----------------------------
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import json

with open("proposed_bn.json", "r") as f:
    bn_json = json.load(f)

def build_model(bn):
    edges = bn["edges"]
    model = DiscreteBayesianNetwork(edges)

    cpds = []

    for node in bn["nodes"]:
        name = node["name"]
        states = node["states"]
        parents = node.get("parents", [])

        cpt = node["cpt"]

        # -----------------------------
        # Build state_names safely
        # -----------------------------
        state_names = {name: states}

        if parents:
            parent_order = cpt["parent_state_order"]
            for p in parents:
                state_names[p] = parent_order[p]

            evidence_card = [len(parent_order[p]) for p in parents]

        else:
            evidence_card = None

        # -----------------------------
        # Root node
        # -----------------------------
        if not parents:
            values = cpt["values"]

            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=values,
                state_names=state_names
            )

        # -----------------------------
        # Child node
        # -----------------------------
        else:
            values = cpt["values"]

            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=values,
                evidence=parents,
                evidence_card=evidence_card,
                state_names=state_names
            )

        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()

    return model

# -----------------------------
def run_inference(model, query, evidence=None):
    infer = VariableElimination(model)

    result = infer.query(
        variables=[query],
        evidence=evidence or {}
    )

    return result

# -----------------------------
# RUN
# -----------------------------
model = build_model(bn_json)
evidence = {
    "GPTN_1": "Not_Found",
    "GPTN_2": "Not_Found",
    "GPTN_3": "Not_Found",
    "GPTN_5": "Found",
    "LPTN_1_i": "Found",
    "LPTN_1_ii": "Found",
    "LPTN_1_iii": "Found",
    "LPTN_1_iv": "Found",
    "LPTN_1_v": "Found",
    "LPTN_1_vi": "Found",
    "LPTN_1_viii": "Found",
    "LPTN_1_ix": "Found",
    "LPTN_1_x": "Found"
}

result = run_inference(
    model,
    query="Root_Causes",
    evidence=evidence
)

print(result)

###

# ### --- Evaluator Agent
# bn_json = subtask_A(full_context)
# reports = failure_reports

# # --- STEP 1 — VALIDATE BN
# def validate_bn(bn):
#     required_keys = ["nodes", "edges"]
#     return all(k in bn for k in required_keys)

# # --- STEP 2 — SUBTASK A (Failure Coverage Evaluation)
# def evaluate_failure_coverage(bn, reports):

#     prompt = f"""
# You are evaluating a Bayesian Network for failure coverage.

# TASK:
# Check whether the BN can represent ALL failure scenarios in the reports.

# BN:
# {json.dumps(bn, indent=2)}

# Reports:
# {reports}

# Return:
# - missing failure modes
# - weak dependencies
# - incorrect probabilistic assumptions

# Be strict and precise.
# """
#     return call_llm(prompt, temperature=0.2)

# # - STEP 3 — SUBTASK B (Consistency Check)
# def evaluate_consistency(bn, reports):

#     prompt = f"""
# You are evaluating whether the Bayesian Network is consistent with previously correctly reasoned cases.

# BN:
# {json.dumps(bn, indent=2)}

# Reports:
# {reports}

# Identify:
# - contradictions
# - overconfident CPTs
# - incorrect independence assumptions
# """
#     return call_llm(prompt, temperature=0.1)

# # - BN TOOL INFERENCE
# from pgmpy.models import BayesianNetwork
# from pgmpy.inference import VariableElimination

# def run_bn_inference(bn):

#     model = BayesianNetwork()

#     # edges
#     for e in bn["edges"]:
#         model.add_edge(e[0], e[1])

#     # NOTE: CPT mapping depends on your encoding
#     # You must adapt this to your CPT structure

#     infer = VariableElimination(model)

#     return infer

# # - SUBTASK C (Feedback Fusion)
# def generate_feedback(eval_a, eval_b, inference_results):

#     prompt = f"""
# You are a Bayesian Network critic and optimizer.

# Combine all signals:

# Evaluation A:
# {eval_a}

# Evaluation B:
# {eval_b}

# Inference Results:
# {inference_results}

# Generate structured feedback:

# {
#   "issues": [...],
#   "fixes": [...],
#   "priority": "high/medium/low"
# }
# """
#     return call_llm(prompt, temperature=0.2)

# # - FULL AGENT 2 PIPELINE
# def agent2_pipeline(bn, reports):

#     if not validate_bn(bn):
#         raise ValueError("Invalid BN format")

#     print("=== Subtask A ===")
#     eval_a = evaluate_failure_coverage(bn, reports)

#     print("=== Subtask B ===")
#     eval_b = evaluate_consistency(bn, reports)

#     print("=== BN Inference ===")
#     inference_results = "RUN_INFERENCE_HERE"  # plug pgmpy output

#     print("=== Subtask C ===")
#     feedback = generate_feedback(eval_a, eval_b, inference_results)

#     return feedback

