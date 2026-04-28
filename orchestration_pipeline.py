import json
from datetime import datetime

import bn_generator as gen
import bn_generator_reflexion as ref
import bn_generator_evaluator as evaler


# -----------------------------
# LOAD FILES
# -----------------------------
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


context_gen = read_file("context_gen_agent.txt")
context_eval = read_file("context_eval_agent.txt")

gen_prompt = read_file("gen_prompt.txt")
ref_prompt = read_file("ref_prompt.txt")
eval_prompt = read_file("eval_prompt.txt")


# -----------------------------
# SETTINGS
# -----------------------------
NUM_ITERATIONS = 3

