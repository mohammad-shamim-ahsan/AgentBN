from pathlib import Path

# ======================================================
# EXPERIMENT
# ======================================================

EXPERIMENT = "sachs"

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "datasets" / EXPERIMENT

WORKSPACE_DIR = ROOT / "workspace" / EXPERIMENT
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_DIR = ROOT / "prompts"
CONTEXT_DIR = PROMPT_DIR / "contexts"


# ======================================================
# INPUT FILES
# ======================================================

GROUND_TRUTH_BN_FILE = DATASET_DIR / "BN_gt.json"

FLAWED_BN_FILE = DATASET_DIR / "flawed_BN_0.json"

TRAIN_CSV = DATASET_DIR / "combined_train_scenarios.csv"

TEST_CSV = DATASET_DIR / "combined_test_scenarios.csv"


# ======================================================
# PROMPTS
# ======================================================

CONTEXT_AGENT_FILE = PROMPT_DIR / EXPERIMENT / "context_agent.txt"

REF_PROMPT_FILE = PROMPT_DIR / "ref_prompt.txt"

SCENARIO_GEN_PROMPT_FILE = PROMPT_DIR / "scenario_gen_prompt.txt"


# ======================================================
# WORKSPACE
# ======================================================

BN_ANALYSIS_FILE = WORKSPACE_DIR / "bn_analysis.json"

PROPOSED_BN_FILE = WORKSPACE_DIR / "last_proposed_bn.jsonl"

RESTART_FINAL_BN_FILE = WORKSPACE_DIR / "restart_final_bns.jsonl"

DANGER_REPORT_FILE = WORKSPACE_DIR / "dangerous_cpt_report.json"

FAILURE_PARAMETER_FILE = WORKSPACE_DIR / "failure_parameter_statistics.json"

ACTIVATION_TRACE_FILE = WORKSPACE_DIR / "activation_trace.csv"

CPT_COMPARISON_FILE = WORKSPACE_DIR / "cpt_comparison_analysis.json"


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
# TARGET NODE
# ======================================================

if EXPERIMENT == "der":
    TARGET_NODE = "Root_Causes"

    TARGET_NODES_FOR_VALIDATION = {
        "Network_Manipulation",
        "Physical_Anomaly",
        "Program_Anomaly",
        "Execution_Integrity",
        "Deviation_in_Response",
        "Deviation_in_Dispatch",
        "Root_Causes",
    }

    EXPECTED_CHANGED_CPTS = {
        "Execution_Integrity",
        "Root_Causes",
    }

elif EXPERIMENT == "lung_cancer":
    TARGET_NODE = "either"
    
    EVIDENCE_NODES = {
        "asia",
        "smoke",
        "xray",
        "dysp",
    }

    TARGET_NODES_FOR_VALIDATION = {
        "tub",
        "lung",
        "bronc",
        "either",
        "xray",
        "dysp",
    }

    EXPECTED_CHANGED_CPTS = {
        "xray"
    }

elif EXPERIMENT == "sachs":
    TARGET_NODE = ""
    
    EVIDENCE_NODES = {
        
    }

    TARGET_NODES_FOR_VALIDATION = {
        
    }

    EXPECTED_CHANGED_CPTS = {
        
    }

else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}") 


# ======================================================
# EVALUATION HYPERPARAMTERS
# ======================================================

MIN_CONFIDENCE = 0.50
MIN_MARGIN = 0.20


# ======================================================
# FORMATTING RETRIALS HYPERPARAMTERS
# ======================================================

MAX_FORMAT_RETRIES = 3
MAX_REPAIR_RETRIES = 2