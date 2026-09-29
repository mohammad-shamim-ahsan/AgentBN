import os
import argparse

import numpy as np
import pandas as pd
import random

from itertools import product
from pgmpy.inference import VariableElimination


# ==================================================
# Arguments
# ==================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--benchmark",
    type=str,
    default="alarm",
    help="Benchmark dataset",
)

parser.add_argument(
    "--iterations",
    type=int,
    default=20,
    help="Number of Batch EM iterations",
)

parser.add_argument(
    "--oracle-size",
    type=int,
    default=5,
    help="Number of CPTs included in the oracle set",
)

parser.add_argument(
    "--num-subsets",
    type=int,
    default=4,
    help="Number of independent training subsets",
)

args = parser.parse_args()

os.environ["BENCHMARK"] = args.benchmark

EM_MAX_ITER = args.iterations
ORACLE_SIZE = args.oracle_size
NUM_SUBSETS = args.num_subsets


# ==================================================
# Project imports
# ==================================================
from utils.pgmpy_tool import *
from utils.bn_io import *
from config.settings import *


# EM convergence tolerance: commonly used to stop EM
# when the change in log-likelihood between consecutive
# iterations becomes sufficiently small.
#
# We do not use a convergence tolerance because the
# baseline is evaluated at a fixed number of EM
# iterations for consistent and reproducible comparison.
#
# EM_TOL = 1e-6


# ==================================================
# Experiment
# ==================================================
print("\n" + "=" * 60)
print(f"BATCH EM BASELINE — BENCHMARK: {BENCHMARK.upper()}")
print(f"EM ITERATIONS: {EM_MAX_ITER}")
print(f"NUMBER OF TRAINING SUBSETS: {NUM_SUBSETS}")
print("=" * 60)


# ==================================================
# Step 1: Load deployed/flawed BN
# ==================================================
bn_json = load_bn(FLAWED_BN_FILE)
model = build_model(bn_json)

print("\n=== Flawed BN loaded ===")
print(f"Number of nodes: {len(model.nodes())}")
print(f"Number of edges: {len(model.edges())}")
print(f"Number of CPTs:  {len(model.get_cpds())}")


# ==================================================
# Step 2: Load and split training data
# ==================================================
full_train_data = pd.read_csv(TRAIN_CSV)

print("\n=== Original training data ===")
print(f"Number of scenarios: {len(full_train_data)}")
print(f"Columns: {list(full_train_data.columns)}")

# Deterministically shuffle the complete refinement set.
full_train_data = full_train_data.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

# Split into approximately equal subsets.
train_subsets = np.array_split(
    full_train_data,
    NUM_SUBSETS,
)

print("\n=== Batch EM training subsets ===")

for subset_id, subset in enumerate(
    train_subsets,
    start=1,
):
    print(
        f"Subset {subset_id}: "
        f"{len(subset)} scenarios"
    )


# ==================================================
# Step 3: Determine CPTs available for EM updates
# ==================================================

# Actual flawed CPTs from benchmark settings.
FLAWED_CPTS = list(EXPECTED_CHANGED_CPTS)

if ORACLE_SIZE == 0:

    # Standard Batch EM:
    # all CPTs are available for parameter learning.
    EM_CPTS = list(model.nodes())
    ORACLE_CPTS = None

    print("\n=== EM configuration ===")
    print("Mode: Standard Batch EM")
    print(f"CPTs available for update: {len(EM_CPTS)}")

else:

    # Oracle-restricted Batch EM:
    # all actually flawed CPTs are always included.
    if ORACLE_SIZE < len(FLAWED_CPTS):
        raise ValueError(
            f"Oracle size must be 0 or at least "
            f"{len(FLAWED_CPTS)}."
        )

    if ORACLE_SIZE > len(model.nodes()):
        raise ValueError(
            f"Oracle size cannot exceed "
            f"{len(model.nodes())}."
        )

    candidate_cpts = [
        variable
        for variable in model.nodes()
        if variable not in FLAWED_CPTS
    ]

    # Fixed seed ensures that the same oracle CPT set
    # is used across all training subsets.
    random.seed(42)

    ORACLE_CPTS = FLAWED_CPTS + random.sample(
        candidate_cpts,
        ORACLE_SIZE - len(FLAWED_CPTS),
    )

    EM_CPTS = ORACLE_CPTS

    print("\n=== EM configuration ===")
    print("Mode: Oracle-restricted Batch EM")
    print(f"Oracle size: {ORACLE_SIZE}")
    print(f"Oracle CPTs: {ORACLE_CPTS}")


# ==================================================
# Archived implementation: pgmpy built-in Batch EM
# ==================================================
#
# The built-in pgmpy ExpectationMaximization estimator was
# initially tested for this baseline. It was not used in the
# final implementation because its E-step expands latent-state
# configurations, which became computationally infeasible for
# the partially observed ALARM scenarios.
#
# This block is retained only for archival/reference purposes.
#
# from pgmpy.estimators import ExpectationMaximization
#
# init_cpds = {
#     cpd.variable: cpd.copy()
#     for cpd in model.get_cpds()
# }
#
# latent_card = {
#     variable: model.get_cpds(variable).variable_card
#     for variable in latent_nodes
# }
#
# state_names = {}
#
# for cpd in model.get_cpds():
#     for variable, states in cpd.state_names.items():
#         state_names[variable] = list(states)
#
# em = ExpectationMaximization(
#     model=model,
#     data=train_data,
#     state_names=state_names,
# )
#
# learned_cpds = em.get_parameters(
#     latent_card=latent_card,
#     init_cpds=init_cpds,
#     max_iter=1,
#     atol=1e-8,
#     n_jobs=1,
#     batch_size=1,
#     show_progress=True,
# )
#
# ==================================================


# ==================================================
# Step 4: Compute expected sufficient statistics
# ==================================================
def compute_expected_counts(
    model,
    inference,
    train_data,
    variable,
):
    cpd = model.get_cpds(variable)

    variable_states = cpd.state_names[variable]
    parent_variables = list(cpd.variables[1:])

    family = [variable] + parent_variables

    parent_state_lists = [
        cpd.state_names[parent]
        for parent in parent_variables
    ]

    expected_counts = np.zeros_like(
        cpd.get_values(),
        dtype=float,
    )

    for _, row in train_data.iterrows():

        evidence = {
            observed_variable: row[observed_variable]
            for observed_variable in train_data.columns
        }

        query_variables = [
            family_variable
            for family_variable in family
            if family_variable not in evidence
        ]

        # Infer the posterior over unobserved
        # variables in the CPT family.
        if query_variables:
            family_posterior = inference.query(
                variables=query_variables,
                evidence=evidence,
                show_progress=False,
            )
        else:
            family_posterior = None

        parent_combinations = (
            product(*parent_state_lists)
            if parent_variables
            else [()]
        )

        for col_idx, parent_states in enumerate(
            parent_combinations
        ):

            parent_assignment = dict(
                zip(parent_variables, parent_states)
            )

            # Skip parent configurations inconsistent
            # with the observed evidence.
            inconsistent = any(
                parent in evidence
                and evidence[parent] != state
                for parent, state
                in parent_assignment.items()
            )

            if inconsistent:
                continue

            for row_idx, variable_state in enumerate(
                variable_states
            ):

                # Skip child states inconsistent
                # with an observed child.
                if (
                    variable in evidence
                    and evidence[variable] != variable_state
                ):
                    continue

                assignment = {
                    variable: variable_state,
                    **parent_assignment,
                }

                if family_posterior is None:
                    probability = 1.0

                else:
                    posterior_assignment = {
                        family_variable: state
                        for family_variable, state
                        in assignment.items()
                        if family_variable in query_variables
                    }

                    probability = family_posterior.get_value(
                        **posterior_assignment
                    )

                expected_counts[
                    row_idx,
                    col_idx,
                ] += probability

    return expected_counts


# ==================================================
# Step 5: Update CPT from expected counts
# ==================================================
def update_cpd_from_counts(
    cpd,
    expected_counts,
):

    column_totals = expected_counts.sum(axis=0)

    if np.any(column_totals == 0):
        raise ValueError(
            f"Zero expected count for at least one "
            f"parent configuration in CPT "
            f"{cpd.variable}."
        )

    new_values = expected_counts / column_totals

    updated_cpd = TabularCPD(
        variable=cpd.variable,
        variable_card=cpd.variable_card,
        values=new_values,
        evidence=list(cpd.variables[1:]),
        evidence_card=list(cpd.cardinality[1:]),
        state_names=cpd.state_names,
    )

    return updated_cpd


# ==================================================
# Step 6: Compute observed-data log-likelihood
# ==================================================
def compute_log_likelihood(
    model,
    train_data,
):

    inference = VariableElimination(model)

    log_likelihood = 0.0
    observed_variables = list(train_data.columns)

    for _, row in train_data.iterrows():

        evidence = {}

        for variable in observed_variables:

            posterior = inference.query(
                variables=[variable],
                evidence=evidence,
                show_progress=False,
            )

            probability = posterior.get_value(
                **{variable: row[variable]}
            )

            if probability <= 0.0:
                return -np.inf

            log_likelihood += np.log(
                probability
            )

            evidence[variable] = row[variable]

    return log_likelihood


# ==================================================
# Step 7: Convert learned model to BN JSON
# ==================================================
def update_bn_json_from_model(
    bn_json,
    model,
):

    bn_new = {
        "edges": bn_json["edges"],
        "nodes": [],
    }

    for node in bn_json["nodes"]:

        node_new = node.copy()
        node_new["cpt"] = node["cpt"].copy()

        cpd = model.get_cpds(
            node["name"]
        )

        node_new["cpt"]["values"] = (
            cpd.get_values().tolist()
        )

        bn_new["nodes"].append(
            node_new
        )

    return bn_new


# ==================================================
# Step 8: Run Batch EM independently per subset
# ==================================================
for subset_id, subset_df in enumerate(
    train_subsets,
    start=1,
):

    print("\n" + "=" * 60)
    print(
        f"BATCH EM — SUBSET "
        f"{subset_id}/{NUM_SUBSETS}"
    )
    print("=" * 60)

    # ----------------------------------------------
    # Prepare this training subset
    # ----------------------------------------------
    train_data = subset_df.copy()

    train_data = train_data.drop(
        columns=["Scenario #"]
    )

    train_data = train_data.rename(
        columns={
            "Ground Truth": TARGET_NODE
        }
    )

    print("\n=== EM training data ===")
    print(
        f"Number of scenarios: "
        f"{len(train_data)}"
    )
    print(
        f"Shape: {train_data.shape}"
    )
    print(
        f"Columns: "
        f"{list(train_data.columns)}"
    )

    # ----------------------------------------------
    # Restart from SAME original flawed BN
    # ----------------------------------------------
    bn_json = load_bn(
        FLAWED_BN_FILE
    )

    initial_model = build_model(
        bn_json
    )

    print("\n=== Flawed BN reloaded ===")
    print(
        f"Number of nodes: "
        f"{len(initial_model.nodes())}"
    )
    print(
        f"Number of edges: "
        f"{len(initial_model.edges())}"
    )
    print(
        f"Number of CPTs:  "
        f"{len(initial_model.get_cpds())}"
    )

    # ----------------------------------------------
    # Verify observed states against BN states
    # ----------------------------------------------
    for variable in train_data.columns:

        data_states = set(
            train_data[
                variable
            ].unique()
        )

        bn_states = set(
            initial_model
            .get_cpds(variable)
            .state_names[variable]
        )

        # A subset does not necessarily contain
        # every possible state. Therefore, only
        # require observed states to be valid BN
        # states.
        if not data_states.issubset(
            bn_states
        ):
            raise ValueError(
                f"State mismatch for "
                f"{variable}: "
                f"data states = "
                f"{sorted(data_states)}, "
                f"BN states = "
                f"{sorted(bn_states)}"
            )

    print(
        "\n✓ All observed variable states "
        "are valid BN states."
    )

    # ----------------------------------------------
    # Identify observed and latent variables
    # ----------------------------------------------
    observed_nodes = set(
        train_data.columns
    )

    latent_nodes = (
        set(initial_model.nodes())
        - observed_nodes
    )

    initial_model.latents = (
        latent_nodes
    )

    print(
        "\n=== Observed and latent variables ==="
    )
    print(
        f"Observed variables: "
        f"{len(observed_nodes)}"
    )
    print(
        f"Latent variables:   "
        f"{len(latent_nodes)}"
    )

    # ----------------------------------------------
    # Initialize independent Batch EM run
    # ----------------------------------------------
    current_model = (
        initial_model.copy()
    )

    parameter_change_history = []

    previous_log_likelihood = (
        compute_log_likelihood(
            model=current_model,
            train_data=train_data,
        )
    )

    print(
        "\n=== Running Batch EM ==="
    )

    # ----------------------------------------------
    # EM iterations
    # ----------------------------------------------
    for iteration in range(
        1,
        EM_MAX_ITER + 1,
    ):

        inference = VariableElimination(
            current_model
        )

        updated_cpds = []

        # ==========================================
        # E-step + M-step
        # ==========================================
        for variable in EM_CPTS:

            cpd = (
                current_model
                .get_cpds(variable)
            )

            expected_counts = (
                compute_expected_counts(
                    model=current_model,
                    inference=inference,
                    train_data=train_data,
                    variable=variable,
                )
            )

            # Each scenario contributes total
            # probability mass 1 to each CPT.
            if not np.isclose(
                expected_counts.sum(),
                len(train_data),
            ):
                raise ValueError(
                    f"Expected-count total "
                    f"mismatch for "
                    f"{variable}: "
                    f"{expected_counts.sum()} "
                    f"!= {len(train_data)}"
                )

            updated_cpd = (
                update_cpd_from_counts(
                    cpd=cpd,
                    expected_counts=(
                        expected_counts
                    ),
                )
            )

            updated_cpds.append(
                updated_cpd
            )

        # ==========================================
        # Construct theta^(t+1)
        # ==========================================
        next_model = (
            current_model.copy()
        )

        # Replace only the CPTs selected
        # for EM updates.
        #
        # For Standard Batch EM this
        # replaces all CPTs.
        for updated_cpd in updated_cpds:

            old_cpd = (
                next_model.get_cpds(
                    updated_cpd.variable
                )
            )

            next_model.remove_cpds(
                old_cpd
            )

            next_model.add_cpds(
                updated_cpd
            )

        if not next_model.check_model():
            raise ValueError(
                f"Batch EM model failed "
                f"validation for subset "
                f"{subset_id} at "
                f"iteration {iteration}."
            )

        # ==========================================
        # Log-likelihood
        # ==========================================
        current_log_likelihood = (
            compute_log_likelihood(
                model=next_model,
                train_data=train_data,
            )
        )

        log_likelihood_change = (
            current_log_likelihood
            - previous_log_likelihood
        )

        # ==========================================
        # Measure parameter convergence
        # ==========================================
        max_change = 0.0
        max_change_variable = None

        for variable in (
            current_model.nodes()
        ):

            old_values = (
                current_model
                .get_cpds(variable)
                .get_values()
            )

            new_values = (
                next_model
                .get_cpds(variable)
                .get_values()
            )

            variable_change = np.max(
                np.abs(
                    new_values
                    - old_values
                )
            )

            if (
                variable_change
                > max_change
            ):
                max_change = (
                    variable_change
                )

                max_change_variable = (
                    variable
                )

        parameter_change_history.append(
            max_change
        )

        print(
            f"Subset {subset_id} | "
            f"Iteration {iteration:3d}: "
            f"log-likelihood = "
            f"{current_log_likelihood:.6f}, "
            f"ΔLL = "
            f"{log_likelihood_change:.6f}, "
            f"max change = "
            f"{max_change:.10f} "
            f"({max_change_variable})"
        )

        previous_log_likelihood = (
            current_log_likelihood
        )

        # Move to theta^(t+1)
        current_model = next_model

        # ------------------------------------------
        # Convergence check intentionally disabled.
        # All runs use the same fixed number of
        # iterations.
        # ------------------------------------------
        #
        # if max_change < EM_TOL:
        #     break


    # ==================================================
    # Step 9: Finalize learned BN for this subset
    # ==================================================
    learned_model = current_model

    if not learned_model.check_model():
        raise ValueError(
            f"Final learned BN for "
            f"subset {subset_id} "
            f"is invalid."
        )

    learned_bn_json = (
        update_bn_json_from_model(
            bn_json=bn_json,
            model=learned_model,
        )
    )


    # ==================================================
    # Step 10: Save learned BN
    #
    # bn_number == subset_id
    # ==================================================
    store_new_bn(
        bn_number=EM_MAX_ITER,
        bn_new=learned_bn_json,
        filename=BATCH_EM_BN_FILE,
        overwrite=(subset_id == 1),
        metadata={
            "subset_id": subset_id,
            "train_size": len(train_data),
            "iterations": EM_MAX_ITER,
            "oracle_size": ORACLE_SIZE,
            "oracle_cpts": ORACLE_CPTS,
        },
    )

    print(
        f"\n✓ Learned Batch EM BN "
        f"for subset {subset_id} saved."
    )


    # ==================================================
    # Step 11: Print CPT changes from flawed BN
    # ==================================================
    print(
        "\n=== CPT changes from flawed BN ==="
    )

    for variable in EM_CPTS:

        old_cpd = (
            initial_model
            .get_cpds(variable)
            .get_values()
        )

        new_cpd = (
            learned_model
            .get_cpds(variable)
            .get_values()
        )

        cpt_change = np.max(
            np.abs(
                new_cpd
                - old_cpd
            )
        )

        print(
            f"{variable:<20} "
            f"max change = "
            f"{cpt_change:.10f}"
        )


    # ==================================================
    # Step 12: Print parameter change history
    # ==================================================
    print(
        "\n=== Parameter change history ==="
    )

    for i, change in enumerate(
        parameter_change_history,
        start=1,
    ):
        print(
            f"Iteration {i:3d}: "
            f"{change:.10f}"
        )


    print(
        f"\n=== Batch EM subset "
        f"{subset_id} finished ==="
    )

    print(
        f"Iterations completed:     "
        f"{iteration}"
    )

    print(
        f"Final maximum CPT change: "
        f"{parameter_change_history[-1]:.10f}"
    )

    print(
        f"Largest-changing CPT "
        f"(final iteration): "
        f"{max_change_variable}"
    )

    print(
        "✓ Final learned BN is valid."
    )


# ==================================================
# Final experiment summary
# ==================================================
print("\n" + "=" * 60)
print("ALL BATCH EM SUBSETS FINISHED")
print("=" * 60)

print(
    f"Training subsets: {NUM_SUBSETS}"
)

print(
    f"Total training scenarios: "
    f"{len(full_train_data)}"
)

print(
    f"Learned BNs stored in: "
    f"{BATCH_EM_BN_FILE}"
)

print("\nMapping:")

for subset_id, subset in enumerate(
    train_subsets,
    start=1,
):
    print(
        f"  bn_number={EM_MAX_ITER}, "
        f"subset_id={subset_id} "
        f"({len(subset)} scenarios)"
    )

print("\n✓ Batch EM experiment completed.")
