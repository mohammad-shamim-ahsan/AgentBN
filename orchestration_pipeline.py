import os
import json
import re

from bn_generator import generate_bn, store_bn_proposal
from bn_generator_evaluator import find_proposed_bn, run_evaluation, store_analysis
from bn_generator_reflexion import generate_refined_bn, store_new_bn

MAX_ITER = 3
bn_analysis_filename = "bn_analysis.json"
max_records = 3
proposed_bn_filename = "last_proposed_bn.jsonl"

# -----------------------------
# SAFE JSON LOADER
# -----------------------------
def safe_json_loads(text):
    if not text or not text.strip():
        return None

    text = text.strip()

    # remove markdown code fences if present
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# -----------------------------
# CLEAR OLD FILES
# -----------------------------
for filename in [bn_analysis_filename, proposed_bn_filename]:
    if os.path.exists(filename):
        open(filename, "w").close()
        print(f"Cleared: {filename}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# STEP 1: INITIAL BN
# -----------------------------
bn_number = 0
full_context = read_file("context_gen_agent.txt")
gen_prompt_template_text = read_file("gen_prompt.txt")

bn_text = generate_bn(full_context, gen_prompt_template_text)

bn_json = safe_json_loads(bn_text)
if bn_json is None:
    raise ValueError("Initial BN generation failed: invalid JSON")

store_bn_proposal(bn_json, bn_number, proposed_bn_filename)

print("\nInitial BN generated")

# -----------------------------
# STEP 2 & 3: LOOP
# -----------------------------
for i in range(MAX_ITER):
    print(f"\n===== ITERATION {i+1} =====")

    ### Evaluate
    prev_bn_json = find_proposed_bn(bn_number, proposed_bn_filename)

    failure_text, success_text, analysis, results = run_evaluation(prev_bn_json)

    store_analysis(bn_number, failure_text, success_text, analysis, results)

    print("\nEvaluation completed. Analysis stored.")

    # STOP CONDITION: no failures
    if not failure_text or not failure_text.strip():
        print(f"\nNo failure cases found for BN {bn_number}. Stopping iterations.")
        break

    ### Reflexion (BN refinement)
    new_bn = generate_refined_bn(
        full_context,
        bn_number,
        proposed_bn_filename,
        bn_analysis_filename,
        max_records
    )

    new_bn_json = safe_json_loads(new_bn)

    if new_bn_json is None:
        print("\nInvalid JSON from LLM during refinement.")
        print("Raw output (truncated):")
        print(new_bn[:1000])
        print("\nStopping iteration safely.")
        break

    bn_number += 1
    store_new_bn(bn_number, new_bn_json)
    print(f"\nRefinement completed. BN#{bn_number} stored.")

    if bn_number >= MAX_ITER:
        print(f"\nReached maximum iterations ({MAX_ITER}). Storing Analysis and Stopping.")
        
        prev_bn_json = find_proposed_bn(bn_number, proposed_bn_filename)
        failure_text, success_text, analysis, results = run_evaluation(prev_bn_json)
        store_analysis(bn_number, failure_text, success_text, analysis, results)
        break

print(f"\nPipeline completed. Final BN number: {bn_number}")
