"""Generate a deployed BN from a ground-truth BN using bounded perturbations."""
import argparse
import copy
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from config.settings import BENCHMARK, TARGET_NODE, MIN_CONFIDENCE, MIN_MARGIN
from utils.pgmpy_tool import build_model, run_inference

K = 2
SENSITIVITY_FRACTION = 0.25
ACCURACY_THRESHOLD = 0.80
DIRICHLET_CONCENTRATION = 50
MAX_PARAMETER_CHANGE = 0.20
PRESERVE_DOMINANT_STATE = True
MAX_ATTEMPTS = 1000
RANDOM_SEED = 42
EPS = 1e-8
DATA_DIR = Path(__file__).resolve().parent / 'datasets' / BENCHMARK


def perturb_cpt(table, rng):
    original = np.asarray(table, dtype=float)
    modified = original.copy()
    for j in range(original.shape[1]):
        p = original[:, j]
        support = p > 0
        if support.sum() < 2:
            continue  # Preserve deterministic columns and structural zeros.
        for _ in range(100):
            q = np.zeros_like(p)
            q[support] = rng.dirichlet(DIRICHLET_CONCENTRATION * p[support])
            if np.max(np.abs(q - p)) > MAX_PARAMETER_CHANGE:
                continue
            if PRESERVE_DOMINANT_STATE and np.argmax(q) != np.argmax(p):
                continue
            modified[:, j] = q
            break
    return modified


def prepare_scenarios(model, path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    if df.empty or 'Ground Truth' not in df:
        raise ValueError('Generation scenarios must contain Ground Truth labels.')
    columns = [c for c in df if c not in ('Scenario #', 'Ground Truth')]
    if not columns or TARGET_NODE in columns:
        raise ValueError('Invalid evidence columns.')
    scenarios = []
    for _, row in df.iterrows():
        evidence = {}
        for c in columns:
            if pd.isna(row[c]) or not row[c].strip():
                continue
            value = row[c].strip()
            if c not in model.nodes() or value not in model.get_cpds(c).state_names[c]:
                raise ValueError(f'Invalid evidence: {c}={value}')
            evidence[c] = value
        label = str(row['Ground Truth']).strip()
        if label not in model.get_cpds(TARGET_NODE).state_names[TARGET_NODE]:
            raise ValueError(f'Invalid ground truth: {label}')
        scenarios.append((evidence, label))
    return scenarios


def evaluate_accuracy(model, scenarios):
    successes = 0
    cache = {}
    for evidence, label in scenarios:
        key = tuple(sorted(evidence.items()))
        if key not in cache:
            result = run_inference(model, query=TARGET_NODE, evidence=evidence)
            values = np.asarray(result.values, dtype=float).ravel()
            if not np.all(np.isfinite(values)) or not np.isclose(values.sum(), 1):
                raise ValueError('Invalid inference posterior.')
            idx = int(np.argmax(values))
            ordered = np.sort(values)
            margin = ordered[-1] - ordered[-2] if len(ordered) > 1 else 1.0
            cache[key] = (result.state_names[TARGET_NODE][idx], values[idx], margin)
        prediction, confidence, margin = cache[key]
        successes += int(prediction == label and confidence >= MIN_CONFIDENCE and margin >= MIN_MARGIN)
    return successes / len(scenarios)


def combination_rounds(eligible, k, rng):
    """Yield each combination once per round, in randomized order.

    For large pools, sample lazily to avoid materializing all combinations.
    """
    total = comb(len(eligible), k)
    round_id = 0
    while True:
        round_id += 1
        if total <= 100_000:
            pool = list(combinations(eligible, k))
            rng.shuffle(pool)
            for selected in pool:
                yield round_id, selected
        else:
            seen = set()
            while len(seen) < total:
                selected = tuple(sorted(map(int, rng.choice(eligible, size=k, replace=False))))
                if selected not in seen:
                    seen.add(selected)
                    yield round_id, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, default=DATA_DIR / 'BN_gt.json')
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'flawed_BN_generated.json')
    parser.add_argument('--k', type=int, default=K)
    parser.add_argument('--scenarios', type=Path, default=DATA_DIR / 'combined_train_scenarios.csv')
    parser.add_argument('--mode', choices=['random', 'high', 'medium', 'low', 'exclude_top_k', 'high_exclude_top_k'], default='random')
    parser.add_argument('--ranking', type=Path, default=DATA_DIR / 'flawed_BN_generated.importance.csv')
    parser.add_argument('--recompute-ranking', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    metadata_path = args.output.with_suffix('.metadata.json')
    if args.output.resolve() == args.gt.resolve() or metadata_path.resolve() == args.gt.resolve():
        raise ValueError('Cannot overwrite the ground-truth BN.')
    if not args.overwrite and (args.output.exists() or metadata_path.exists()):
        raise FileExistsError('Output exists; choose another path or pass --overwrite.')
    gt = json.loads(args.gt.read_text(encoding='utf-8'))
    gt_model = build_model(gt)
    gt_model.check_model()
    scenarios = prepare_scenarios(gt_model, args.scenarios)
    gt_accuracy = evaluate_accuracy(gt_model, scenarios)
    print(f'Ground-truth accuracy: {gt_accuracy:.2%}', flush=True)
    if gt_accuracy < ACCURACY_THRESHOLD:
        raise ValueError('Ground-truth accuracy is already below the acceptance threshold.')
    eligible = [i for i, node in enumerate(gt['nodes'])
                if np.any((np.asarray(node['cpt']['values']) > 0).sum(axis=0) > 1)]
    ranking = None
    ranking_path = args.ranking
    if args.mode != 'random':
        n = max(1, round(len(scenarios) * SENSITIVITY_FRACTION))
        if ranking_path.resolve() in {args.gt.resolve(), args.output.resolve(), metadata_path.resolve(), args.scenarios.resolve()}:
            raise ValueError('Ranking path conflicts with an input or output file.')
        if ranking_path.exists() and not args.recompute_ranking:
            ranking = pd.read_csv(ranking_path)
            print(f'Reusing ground-truth CPT ranking: {ranking_path}', flush=True)
        else:
            from baselines.sensitivity_analysis import calculate_importance
            sample_rng = np.random.default_rng(RANDOM_SEED)
            indices = sample_rng.choice(len(scenarios), size=n, replace=False)
            evidence_sample = [scenarios[int(i)][0] for i in indices]
            importance = calculate_importance(gt_model, evidence_sample)
            ranking = (importance.groupby('node')['importance'].max()
                       .sort_values(ascending=False, kind='stable')
                       .rename('importance').reset_index())
            ranking['rank'] = np.arange(1, len(ranking) + 1)
            ranking['category'] = ''
            for category, indices in zip(('high', 'medium', 'low'),
                                         np.array_split(ranking.index.to_numpy(), 3)):
                ranking.loc[indices, 'category'] = category
            ranking_path.parent.mkdir(parents=True, exist_ok=True)
            ranking.to_csv(ranking_path, index=False)
            print(f'Saved ground-truth CPT ranking: {ranking_path}', flush=True)
        if not {'node', 'importance', 'rank', 'category'}.issubset(ranking.columns):
            raise ValueError('Invalid ranking CSV columns.')
        if ranking['node'].duplicated().any() or set(ranking['node']) != {node['name'] for node in gt['nodes']} or not np.all(np.isfinite(ranking['importance'])) or sorted(ranking['rank']) != list(range(1, len(ranking)+1)):
            raise ValueError('Ranking does not match BN nodes or valid ranks; recompute it.')
        ranking = ranking.sort_values('rank')
        print(ranking.to_string(index=False), flush=True)
        if args.mode == 'high_exclude_top_k':
            names = set(ranking.loc[(ranking['category'] == 'high') & (ranking['rank'] > args.k), 'node'])
        elif args.mode == 'exclude_top_k':
            names = set(ranking.loc[ranking['rank'] > args.k, 'node'])
        else:
            names = set(ranking.loc[ranking['category'] == args.mode, 'node'])
        eligible = [i for i in eligible if gt['nodes'][i]['name'] in names]
    if not 1 <= args.k <= len(eligible):
        raise ValueError('K must be between 1 and the number of perturbable CPTs in the selected mode.')
    rng = np.random.default_rng(RANDOM_SEED)
    best = gt_accuracy
    total_combinations = comb(len(eligible), args.k)
    print(f'Eligible CPTs: {len(eligible)} | Unique combinations: {total_combinations}', flush=True)
    selection_rng = np.random.default_rng(RANDOM_SEED)
    rounds = combination_rounds(eligible, args.k, selection_rng)
    previous_round = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        candidate = copy.deepcopy(gt)
        round_id, selected = next(rounds)
        if round_id != previous_round:
            print(f'=== Combination round {round_id} ===', flush=True)
            previous_round = round_id
        changed = []
        for i in selected:
            node = candidate['nodes'][int(i)]
            original = np.asarray(node['cpt']['values'], dtype=float)
            modified = perturb_cpt(original, rng)
            if np.max(np.abs(modified - original)) > EPS:
                node['cpt']['values'] = modified.tolist()
                changed.append(node['name'])
        if len(changed) != args.k:
            continue
        model = build_model(candidate)
        model.check_model()
        accuracy = evaluate_accuracy(model, scenarios)
        best = min(best, accuracy)
        print(f'Attempt {attempt}: {accuracy:.2%} | {", ".join(sorted(changed))}', flush=True)
        if accuracy < ACCURACY_THRESHOLD:
            metadata = dict(benchmark=BENCHMARK, target=TARGET_NODE, k=args.k,
                            ground_truth_cpt_set=sorted(changed), attempt=attempt,
                            ground_truth_accuracy=gt_accuracy, deployed_accuracy=accuracy,
                            accuracy_threshold=ACCURACY_THRESHOLD, seed=RANDOM_SEED,
                            dirichlet_concentration=DIRICHLET_CONCENTRATION,
                            max_parameter_change=MAX_PARAMETER_CHANGE,
                            preserve_dominant_state=PRESERVE_DOMINANT_STATE,
                            min_confidence=MIN_CONFIDENCE, min_margin=MIN_MARGIN,
                            generation_scenarios=str(args.scenarios), num_scenarios=len(scenarios))
            metadata['selection_mode'] = args.mode
            metadata['combination_round'] = round_id
            metadata['unique_combinations_per_round'] = total_combinations
            metadata['selection_policy'] = 'without_replacement_per_round'
            if ranking is not None:
                metadata['importance_ranking_file'] = str(ranking_path)
                metadata['excluded_top_k'] = ranking.head(args.k)['node'].tolist() if args.mode in ('exclude_top_k', 'high_exclude_top_k') else []
                metadata['selected_cpt_importance'] = ranking.loc[
                    ranking['node'].isin(changed)
                ].to_dict(orient='records')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f'Saved {args.output} and {metadata_path}', flush=True)
            return
    raise RuntimeError(f'No acceptable candidate after {MAX_ATTEMPTS} attempts. Lowest accuracy: {best:.2%}. No deployed BN saved.')


#--------------------------------------------------------------
if __name__ == '__main__':
    main()
