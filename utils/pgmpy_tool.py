from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np

from config.settings import *


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


def run_inference(model, query, evidence=None):
    infer = VariableElimination(model)

    result = infer.query(
        variables=[query],
        evidence=evidence or {}
    )

    return result


def call_bn_inference(model, df):

    results = []

    for _, row in df.iterrows():

        evidence = {}

        for col in df.columns:

            if col in ["Scenario #", "Ground Truth"]:
                continue

            val = row[col]

            if pd.isna(val):
                continue

            evidence[col] = str(val).strip()

        # print(f"\nRunning inference for Scenario: {row['Scenario #']}")
        # print("Evidence:", evidence)

        result = run_inference(
            model,
            query=TARGET_NODE,
            evidence=evidence
        )

        pred_idx = np.argmax(result.values)
        pred_state = result.state_names[TARGET_NODE][pred_idx]
        confidence = result.values[pred_idx]

        probs = sorted(result.values, reverse=True)
        max_prob = probs[0]
        second_prob = probs[1]
        margin = max_prob - second_prob

        posterior_probs = {
            state: float(prob)
            for state, prob in zip(
                result.state_names[TARGET_NODE],
                result.values
            )
        }

        # Success only if:
        # 1. Predicted class matches Ground Truth
        # 2. Confidence >= 50%
        # 3. Margin >= 20%
        is_success = (
            pred_state == row["Ground Truth"]
            and confidence >= MIN_CONFIDENCE
            and margin >= MIN_MARGIN
        )

        results.append({
            "Scenario": row["Scenario #"],
            "Prediction": pred_state,
            "Confidence": confidence,
            "Ground Truth": row["Ground Truth"],
            "Posterior": posterior_probs,
            "Success": is_success
        })

    return results
