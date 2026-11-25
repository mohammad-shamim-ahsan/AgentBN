"""Selective ALARM BN refinement via three-point sensitivity analysis.

Place this file at your project root, where utils/ and config/ are importable.
Run: python sensitivity_analysis_refinement.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from utils.pgmpy_tool import build_model, run_inference
from utils.bn_io import store_new_bn
from pgmpy.inference import VariableElimination
from config.settings import *
from itertools import product
import copy

DATA_DIR = Path('datasets') / BENCHMARK
BN_FILE = DATA_DIR / 'flawed_BN_0.json'
SCENARIOS_FILE = DATA_DIR / 'combined_train_scenarios.csv'
OUTPUT_FILE = SENSITIVITY_BN_FILE
QUERY_STATE = 'TRUE'
NUM_SUBSETS = 4
SENSITIVITY_FRACTION = 0.25  # Sample from each learning subset; retain all cases for learning.
SPLIT_SEED = 42
DELTA = 0.2
K = 2  # Maximum number of distinct CPTs allowed to be refined
MAX_ITERATIONS = None  # None means stop only when no new parameters qualify.
PRIOR_PSEUDOCOUNT = 1.0  # Uniform Dirichlet baseline, plus prior-data counts.
EPS = 1e-12


def load_scenarios(df, model, limit=None):
    """Convert a single training subset to evidence scenarios."""
    evidence_columns = [c for c in df.columns if c not in ('Scenario #', 'Ground Truth')]
    if not evidence_columns:
        raise ValueError('No evidence columns found.')
    if TARGET_NODE in evidence_columns:
        raise ValueError('Target cannot be included in evidence.')
    scenarios = []
    for _, row in (df.head(limit) if limit is not None else df).iterrows():
        evidence = {}
        for var in evidence_columns:
            value = str(row[var]).strip()
            if not value:
                continue
            states = model.get_cpds(var).state_names[var]
            if value not in states:
                raise ValueError(f'Invalid state {value!r} for {var}: {states}')
            evidence[var] = value
        scenarios.append(evidence)
    if not scenarios:
        raise ValueError('No evidence scenarios found.')
    return scenarios


def query_probability(model, evidence):
    posterior = run_inference(model, query=TARGET_NODE, evidence=evidence)
    states = posterior.state_names[TARGET_NODE]
    return float(posterior.values[states.index(QUERY_STATE)])


def perturb_cpd(cpd, column, state_index, new_value):
    """Return perturbed CPD, using proportional covariation in one column."""
    table = cpd.get_values().copy()
    original = float(table[state_index, column])
    if original >= 1.0 - EPS:
        raise ValueError('Proportional covariation undefined for x=1.')
    table[:, column] *= (1.0 - new_value) / (1.0 - original)
    table[state_index, column] = new_value
    new_cpd = cpd.copy()
    new_cpd.values = table.reshape(cpd.values.shape)
    return new_cpd


def three_point_sensitivity(model, node, column, state_index, evidence):
    original_cpd = model.get_cpds(node)
    x = float(original_cpd.get_values()[state_index, column])
    p = []
    try:
        for t in (0.0, 0.5, 1.0):
            model.add_cpds(perturb_cpd(original_cpd, column, state_index, t))
            p.append(query_probability(model, evidence))
    finally:
        model.add_cpds(original_cpd)

    p0, phalf, p1 = p

    if not np.all(np.isfinite(p)):
        print(f"Nonfinite posterior: {node}, "
            f"column={column}, state={state_index}")
        print(f"x={x}, p0={p0}, phalf={phalf}, p1={p1}")
        print(f"Evidence: {evidence}", flush=True)
        return float('nan')

    if max(p) - min(p) < np.finfo(float).eps:
        return 0.0
    
    # From p(t) = (alpha*t + beta)/(gamma*t + 1):
    # phalf*(gamma+2) = alpha + 2*beta
    # p1*(gamma+1) = alpha + beta
    # Thus gamma = (2*phalf - p0 - p1)/(p1 - phalf)
    if abs(p1 - phalf) < np.finfo(float).eps:
        print(f"Unstable gamma: {node}, "
            f"column={column}, state={state_index}")
        print(f"x={x}, p0={p0}, phalf={phalf}, p1={p1}")
        print(f"Evidence: {evidence}", flush=True)
        return float('nan')
    
    beta = p0
    gamma = (2.0 * phalf - p0 - p1) / (p1 - phalf)
    alpha = p1 * (gamma + 1.0) - beta
    denominator = (gamma * x + 1.0) ** 2
    
    if denominator < EPS:
        print(f"Near-zero denominator: {node}, "
            f"column={column}, state={state_index}")
        print(f"x={x}, p0={p0}, phalf={phalf}, p1={p1}")
        print(f"gamma={gamma}, denominator={denominator}")
        print(f"Evidence: {evidence}", flush=True)
        return float('nan')
    
    return float((alpha - beta * gamma) / denominator)


def calculate_importance(model, scenarios):
    rows = []
    for cpd in list(model.get_cpds()):
        node = cpd.variable
        table = cpd.get_values()
        for column in range(table.shape[1]):
            # Every admissible CPT entry is tested as a meta parameter.
            # At x=1, proportional covariation is not uniquely defined.
            for state_index in range(table.shape[0]):
                x = float(table[state_index, column])
                if x >= 1.0 - EPS:
                    continue
                
                derivatives = []
                for scenario_idx, evidence in enumerate(scenarios):
                    derivative = three_point_sensitivity(
                        model, node, column, state_index, evidence
                    )

                    if not np.isfinite(derivative):
                        raise ArithmeticError(
                            f"Nonfinite sensitivity for {node}, "
                            f"column={column}, state={state_index}, "
                            f"scenario={scenario_idx}, x={x}, "
                            f"evidence={evidence}"
                        )

                    derivatives.append(derivative)

                rows.append({
                    'node': node,
                    'column': column,
                    'state': cpd.state_names[node][state_index],
                    'parameter_value': x,
                    'importance': float(np.mean(derivatives)),
                    'mean_absolute_sensitivity': float(np.mean(np.abs(derivatives))),
                    'num_scenarios': len(scenarios),
                })
        print(f'Completed CPT: {node}', flush=True)
    
    return pd.DataFrame(rows).sort_values('importance', ascending=False)


def compute_expected_counts(model, inference, train_data, variable):
    """E-step copied from batch_em.py: expected CPT-family counts.

    train_data contains observed columns, including TARGET_NODE obtained
    by renaming the Ground Truth column.
    """

    cpd = model.get_cpds(variable)
    variable_states = cpd.state_names[variable]
    parent_variables = list(cpd.variables[1:])
    family = [variable] + parent_variables
    parent_state_lists = [cpd.state_names[parent] for parent in parent_variables]
    expected_counts = np.zeros_like(cpd.get_values(), dtype=float)

    for _, row in train_data.iterrows():
        evidence = {observed_variable: row[observed_variable]
                    for observed_variable in train_data.columns}
        query_variables = [family_variable for family_variable in family
                           if family_variable not in evidence]
        if query_variables:
            family_posterior = inference.query(
                variables=query_variables,
                evidence=evidence,
                show_progress=False,
            )
        else:
            family_posterior = None

        parent_combinations = (
            product(*parent_state_lists) if parent_variables else [()]
        )
        for col_idx, parent_states in enumerate(parent_combinations):
            parent_assignment = dict(zip(parent_variables, parent_states))
            inconsistent = any(
                parent in evidence and evidence[parent] != state
                for parent, state in parent_assignment.items()
            )
            if inconsistent:
                continue
            for row_idx, variable_state in enumerate(variable_states):
                if variable in evidence and evidence[variable] != variable_state:
                    continue
                assignment = {variable: variable_state, **parent_assignment}
                if family_posterior is None:
                    probability = 1.0
                else:
                    posterior_assignment = {
                        family_variable: state
                        for family_variable, state in assignment.items()
                        if family_variable in query_variables
                    }
                    probability = family_posterior.get_value(**posterior_assignment)
                expected_counts[row_idx, col_idx] += probability

    return expected_counts


def prepare_em_data(df, model):
    """Match batch_em.py's observed-variable handling."""
    data = df.drop(columns=['Scenario #'], errors='ignore').rename(
        columns={'Ground Truth': TARGET_NODE}
    ).copy()
    if data.columns.duplicated().any():
        raise ValueError('Duplicate observed variable columns after renaming.')
    for variable in data.columns:
        allowed = set(model.get_cpds(variable).state_names[variable])
        observed = set(data[variable])
        if not observed.issubset(allowed):
            raise ValueError(f'Invalid states for {variable}: {observed - allowed}')
    return data


def estimate_prior_hyperparameters(model, prior_data):
    inference = VariableElimination(model)
    priors = {}

    for node in model.nodes():
        prior_counts = compute_expected_counts(
            model, inference, prior_data, node
        )

        if not np.isclose(prior_counts.sum(), len(prior_data)):
            raise ValueError(f"Prior expected-count mismatch for {node}")

        priors[node] = PRIOR_PSEUDOCOUNT + prior_counts

    return priors


def refine_selected(model, selected, priors, train_data):
    """Apply Equation (4) to selected meta parameters, covarying other states.

    Fixed Dirichlet hyperparameters are estimated from subset 1
    using the original deployed BN. Expected training counts are
    computed using the current BN. Nonselected CPT columns are untouched.
    """
    selected_by_node = {}
    for row in selected.itertuples(index=False):
        selected_by_node.setdefault(row.node, {}).setdefault(
            int(row.column), set()
        ).add(row.state)

    # Reuse the same E-step inference object for all selected CPT families.
    inference = VariableElimination(model)
    updates = []

    for node, columns in selected_by_node.items():
        cpd = model.get_cpds(node)
        original = cpd.get_values().copy()

        data_counts = compute_expected_counts(
            model, inference, train_data, node
        )

        if not np.isclose(data_counts.sum(), len(train_data)):
            raise ValueError(f"Training expected-count mismatch for {node}")

        alpha = priors[node]
        estimates = (alpha + data_counts) / (
            alpha + data_counts
        ).sum(axis=0, keepdims=True)

        updated = original.copy()
        states = cpd.state_names[node]

        for col, selected_states in columns.items():
            chosen = [
                i for i, state in enumerate(states)
                if state in selected_states
            ]
            others = [
                i for i in range(len(states))
                if i not in chosen
            ]

            if not others:
                # All states in a column were selected: use the full update.
                updated[:, col] = estimates[:, col]
                continue

            chosen_total = float(estimates[chosen, col].sum())
            updated[chosen, col] = estimates[chosen, col]
            remainder = max(0.0, 1.0 - chosen_total)
            old_other_total = float(original[others, col].sum())

            if old_other_total > EPS:
                updated[others, col] = (
                    original[others, col] * remainder / old_other_total
                )
            else:
                # Proportional covariation is undefined for zero remainder mass.
                # Use posterior proportions among nonselected states instead.
                posterior_other_total = float(estimates[others, col].sum())

                if posterior_other_total <= EPS:
                    raise ArithmeticError(
                        f'Cannot normalize {node} column {col}'
                    )

                updated[others, col] = (
                    estimates[others, col] * remainder / posterior_other_total
                )

        new_cpd = cpd.copy()
        new_cpd.values = updated.reshape(cpd.values.shape)
        updates.append(new_cpd)

    # Apply simultaneously, as in the Batch EM M-step.
    for new_cpd in updates:
        model.add_cpds(new_cpd)


def main():
    with BN_FILE.open(encoding='utf-8') as f:
        bn_json = json.load(f)

    full_train_data = pd.read_csv(SCENARIOS_FILE, dtype=str, keep_default_na=False)
    full_train_data.columns = full_train_data.columns.str.strip()
    if 'Ground Truth' not in full_train_data.columns:
        raise ValueError('Ground Truth column is required for parameter refinement.')

    # Same shuffle and four subset memberships as the Batch EM baseline.
    full_train_data = full_train_data.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
    train_subsets = np.array_split(full_train_data, NUM_SUBSETS)
    if NUM_SUBSETS != 4:
        raise ValueError('This experiment reserves subset 1 and refines subsets 2–4; NUM_SUBSETS must be 4.')

    # Subset 1 is reserved exclusively for informative prior estimation.
    # Subsets 2–4 retain the same scenario membership as AgentBN/Batch EM.
    prior_df = train_subsets[0]
    print(f'Prior dataset: subset 1 ({len(prior_df)} scenarios)', flush=True)
    print('Learning datasets: subsets 2, 3, 4', flush=True)

    deployed_model = build_model(bn_json)
    prior_data = prepare_em_data(prior_df, deployed_model)

    priors = estimate_prior_hyperparameters(
        deployed_model, prior_data
    )

    for subset_id in range(2, NUM_SUBSETS + 1):
        subset_df = train_subsets[subset_id - 1]
        model = build_model(bn_json)
        train_data = prepare_em_data(subset_df, model)

        # Fixed, reproducible 25% sample for sensitivity; its observations
        # remain in train_data for parameter learning (no 25/75 partition).
        num_sensitivity = max(1, int(round(len(subset_df) * SENSITIVITY_FRACTION)))
        sensitivity_df = subset_df.sample(
            n=num_sensitivity, random_state=SPLIT_SEED + subset_id
        )
        scenarios = load_scenarios(sensitivity_df, model)
        refined = set()
        refined_cpts = set()
        iteration = 0
        print(f'\n=== Learning subset {subset_id}/{NUM_SUBSETS}: '
              f'{len(prior_data)} shared prior cases (subset 1), '
              f'{len(train_data)} training cases (100%), '
              f'{len(scenarios)} sensitivity scenarios (25%) ===', flush=True)

        while True:
            if MAX_ITERATIONS is not None and iteration >= MAX_ITERATIONS:
                print('Reached configured iteration limit.', flush=True)
                break
            
            importance = calculate_importance(model, scenarios)

            # Select parameters exceeding the importance threshold.
            eligible = importance[importance['importance'] > DELTA]

            # Remaining budget for new CPTs.
            remaining_budget = K - len(refined_cpts)

            # Rank previously unrefined CPTs by maximum parameter importance.
            new_cpt_importance = (
                eligible[~eligible['node'].isin(refined_cpts)]
                .groupby('node')['importance']
                .max()
                .sort_values(ascending=False)
            )

            # Admit new CPTs within the remaining budget.
            new_cpts = set(
                new_cpt_importance.head(remaining_budget).index
            )

            # Previously refined CPTs do not consume additional budget.
            allowed_cpts = refined_cpts | new_cpts

            # Select eligible parameters within allowed CPTs.
            selected = eligible[eligible['node'].isin(allowed_cpts)]

            if selected.empty or all(
                (r.node, int(r.column), r.state) in refined
                for r in selected.itertuples(index=False)
            ):
                print('No new eligible parameters; stopping.', flush=True)
                break
            
            iteration += 1
            print(f'Iteration {iteration}: selected {len(selected)} parameters '
                  f'across {selected.node.nunique()} CPTs', flush=True)
            print('Selected CPTs:', ', '.join(sorted(selected.node.unique())), flush=True)
            for r in selected.itertuples(index=False):
                print(f'  {r.node} | column={r.column} | state={r.state} '
                      f'| importance={r.importance:.6g}', flush=True)
            
            refine_selected(model, selected, priors, train_data)
            model.check_model()
            refined.update(
                (r.node, int(r.column), r.state)
                for r in selected.itertuples(index=False)
            )

            refined_cpts.update(selected['node'].unique())

            assert len(refined_cpts) <= K

            print(
                f'Cumulative refined CPTs ({len(refined_cpts)}/{K}): '
                f'{", ".join(sorted(refined_cpts))}',
                flush=True
            )

        # Store all final models in one file, using the same project utility
        # and record metadata convention as batch_em.py.
        refined_bn_json = copy.deepcopy(bn_json)
        for node in refined_bn_json['nodes']:
            node['cpt']['values'] = model.get_cpds(node['name']).get_values().tolist()
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        store_new_bn(
            bn_number=iteration,
            bn_new=refined_bn_json,
            filename=str(OUTPUT_FILE),
            overwrite=(subset_id == 2),
            metadata={
                'subset_id': subset_id,
                'train_size': len(train_data),
                'prior_subset_id': 1,
                'prior_size': len(prior_data),
                'sensitivity_size': len(scenarios),
                'sensitivity_fraction': SENSITIVITY_FRACTION,
                'iterations': iteration,
                'threshold': DELTA,
                'cpt_budget': K,
                'num_refined_cpts': len(refined_cpts),
                'refined_cpts': sorted(refined_cpts),
            },
        )
        print(f'Saved refined BN for subset {subset_id}: {OUTPUT_FILE}', flush=True)

    print('\nAll three independent refinement runs (subsets 2–4) completed.', flush=True)


#-------------------------------------------------------------------------
if __name__ == '__main__':
    main()
