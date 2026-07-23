import os
import json
import argparse
from utils.pgmpy_tool import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--benchmark",
    type=str,
    default="alarm",
    choices=["alarm", "lung_cancer", "der"],
    help="Benchmark to run."
)

args = parser.parse_args()

# Pass benchmark to settings.py
os.environ["BENCHMARK"] = args.benchmark

from config.settings import *

# --------------------------------------------------
# Load BN
# --------------------------------------------------
with open(GROUND_TRUTH_BN_FILE, "r") as f:
    bn_json = json.load(f)

model = build_model(bn_json)

# --------------------------------------------------
# Find relevant nodes
# --------------------------------------------------
relevant_nodes = get_target_evidence_paths(
    model,
    EVIDENCE_NODES,
    verbose=True
)