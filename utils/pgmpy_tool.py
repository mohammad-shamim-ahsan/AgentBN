from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
from collections import deque

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


def get_target_evidence_paths(model, evidence_nodes, verbose=False):
    """
    Returns:

    1. All nodes lying on directed paths between TARGET_NODE and
       each evidence node.

    2. All additional ancestors required to determine the active
       CPT columns of those path nodes.

    3. All directed target-evidence paths, represented in the
       forward direction of the Bayesian Network.
    """

    if verbose:
        print("\n" + "=" * 70)
        print("Finding Target-Evidence Paths")
        print(f"Target Node: {TARGET_NODE}")
        print("=" * 70)

    # --------------------------------------------------
    # Build adjacency lists
    # --------------------------------------------------
    children = {
        node: list(model.successors(node))
        for node in model.nodes()
    }

    parents = {
        node: list(model.predecessors(node))
        for node in model.nodes()
    }

    relevant_nodes = {TARGET_NODE}

    # Store all discovered directed paths
    path_list = []

    # --------------------------------------------------
    # DFS: enumerate all directed paths
    # --------------------------------------------------
    def dfs_all_paths(current, destination, adjacency, path):

        if current == destination:
            yield path
            return

        for nxt in adjacency.get(current, []):

            if nxt in path:
                continue

            yield from dfs_all_paths(
                nxt,
                destination,
                adjacency,
                path + [nxt]
            )

    # --------------------------------------------------
    # Process every evidence node
    # --------------------------------------------------
    for evidence in evidence_nodes:

        if evidence == TARGET_NODE:
            continue

        if verbose:
            print(f"\nEvidence: {evidence}")

        # ==================================================
        # Case 1:
        # TARGET ---> ... ---> EVIDENCE
        # ==================================================
        paths = list(
            dfs_all_paths(
                TARGET_NODE,
                evidence,
                children,
                [TARGET_NODE]
            )
        )

        if paths:

            if verbose:
                print("  Direction : TARGET -> EVIDENCE")
                print(f"  Paths Found : {len(paths)}")

            for i, path in enumerate(paths, 1):

                if verbose:
                    print(f"    Path {i}: {' -> '.join(path)}")

                relevant_nodes.update(path)

                # Already in BN-forward direction
                path_list.append(path)

            continue

        # ==================================================
        # Case 2:
        # EVIDENCE ---> ... ---> TARGET
        # ==================================================
        paths = list(
            dfs_all_paths(
                TARGET_NODE,
                evidence,
                parents,
                [TARGET_NODE]
            )
        )

        if paths:

            if verbose:
                print("  Direction : EVIDENCE -> TARGET")
                print(f"  Paths Found : {len(paths)}")

            for i, path in enumerate(paths, 1):

                # Reverse so path is stored in BN-forward direction
                forward_path = list(reversed(path))

                if verbose:
                    print(
                        f"    Path {i}: "
                        f"{' -> '.join(forward_path)}"
                    )

                relevant_nodes.update(path)
                path_list.append(forward_path)

        else:
            if verbose:
                print("  No directed path found.")

    # --------------------------------------------------
    # Add parent closure
    # --------------------------------------------------
    path_nodes = set(relevant_nodes)
    added_parent_nodes = set()

    nodes_to_process = list(relevant_nodes)

    while nodes_to_process:

        node = nodes_to_process.pop()

        for parent in parents.get(node, []):

            if parent not in relevant_nodes:
                relevant_nodes.add(parent)
                added_parent_nodes.add(parent)
                nodes_to_process.append(parent)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print(f"Path Nodes ({len(path_nodes)})")

        for node in sorted(path_nodes):
            print(f"  {node}")

        print(
            "\nAdditional Parent Nodes Required for CPT Activation "
            f"({len(added_parent_nodes)})"
        )

        if added_parent_nodes:
            for node in sorted(added_parent_nodes):
                print(f"  {node}")
        else:
            print("  None")

        print("\n" + "-" * 70)
        print(f"Final Relevant Nodes ({len(relevant_nodes)})")

        for node in sorted(relevant_nodes):
            print(f"  {node}")

        print("\n" + "-" * 70)
        print(f"Directed Paths ({len(path_list)})")

        for i, path in enumerate(path_list, 1):
            print(f"  P{i}: {' -> '.join(path)}")

        print("=" * 70)

    return relevant_nodes, path_nodes, added_parent_nodes, path_list


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
