# ======================================================
# FILES
# ======================================================

BN_ANALYSIS_FILE = "workspace/bn_analysis.json"

PROPOSED_BN_FILE = "workspace/last_proposed_bn.jsonl"

TRAIN_CSV = "datasets/combined_train_scenarios.csv"

TEST_CSV = "datasets/combined_test_scenarios.csv"

RESTART_FINAL_BN_FILE = "workspace/restart_final_bns.jsonl"


# ======================================================
# INPUT FILES
# ======================================================

CONTEXT_AGENT_FILE = "prompts/context_agent.txt"

GEN_PROMPT_FILE = "prompts/gen_prompt.txt"

REF_PROMPT_FILE = "prompts/ref_prompt.txt"

FLAWED_BN_FILE = "datasets/flawed_BN_0.json"

GROUND_TRUTH_BN_FILE = "datasets/BN_gt.json"


# ======================================================
# EXPERIMENT SETTINGS
# ======================================================

MAX_RESTARTS = 3

BASE_TEMPERATURE = 0.2

MAX_ITER = 3

MAX_INITIAL_RETRIES = 3

INITIAL_IMPROVEMENT_RATIO = 0.30

MAX_NO_IMPROVEMENT_RETRIES = 3

# ======================================================
# WORKSPACE
# ======================================================

DANGER_REPORT_FILE = "workspace/dangerous_cpt_report.json"

FAILURE_PARAMETER_FILE = "workspace/failure_parameter_statistics.json"

ACTIVATION_TRACE_FILE = "workspace/activation_trace.csv"

CPT_COMPARISON_FILE = "workspace/cpt_comparison_analysis.json"