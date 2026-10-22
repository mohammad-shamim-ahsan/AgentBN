import pandas as pd
import random

from pathlib import Path

from automatic_reasoning_simple import run_evaluation, load_bn
from config.settings import *


# ============================================================
# Environment Settings
# ============================================================

print(f"BENCHMARK = {BENCHMARK!r}")

DATASET_DIR = Path("datasets") / BENCHMARK
CSV_DIR = DATASET_DIR / "csv"
CSV_FILES = sorted(CSV_DIR.glob(f"{BENCHMARK}.*.csv"))

if not CSV_FILES:
    raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")


# ============================================================
# Ground-Truth Success Filtering
# ============================================================

def keep_success_cases(csv_file):

    failures, successes, accuracy, results = run_evaluation(
        bn_json,
        csv_file,
        target_node=TARGET_NODE
    )

    df = pd.read_csv(csv_file, dtype=str)

    success_ids = set(
        pd.to_numeric(successes["Scenario"]).astype(int)
    )

    df = df[
        df["Scenario #"].astype(int).isin(success_ids)
    ].copy()

    df = df.reset_index(drop=True)
    df["Scenario #"] = range(1, len(df) + 1)

    df.to_csv(csv_file, index=False)

    print(
        f"{csv_file.name}: "
        f"{len(df)} / {len(results)} scenarios retained"
    )

    return df

def normalize_states(df, bn_json):
    df = df.copy()

    for node in bn_json["nodes"]:
        node_name = node["name"]

        if node_name not in df.columns:
            continue

        valid_states = node["states"]

        # Case-insensitive mapping to exact BN state labels
        state_mapping = {
            str(state).strip().casefold(): str(state)
            for state in valid_states
        }

        normalized = (
            df[node_name]
            .astype("string")
            .str.strip()
            .str.casefold()
            .map(state_mapping)
        )

        # Detect unrecognized states rather than silently accepting them
        invalid = normalized.isna()

        if invalid.any():
            unknown_values = (
                df.loc[invalid, node_name]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"Unknown states for node '{node_name}': "
                f"{unknown_values}. Expected: {valid_states}"
            )

        df[node_name] = normalized.astype(str)

    return df


# Load BNs
bn_json = load_bn(GROUND_TRUTH_BN_FILE)
flawed_bn = load_bn(FLAWED_BN_FILE)

# Load CSV samples
all_samples = pd.concat(
    [pd.read_csv(f, dtype=str) for f in CSV_FILES],
    ignore_index=True
)

# Normalize using exact state labels from the ground-truth BN
all_samples = normalize_states(all_samples, bn_json)

print(f"Combined raw shape: {all_samples.shape}")

# ============================================================
# Experiment B: Random Evidence Selection
# ============================================================

RANDOM_SEED_B = 42

# Same number of evidence nodes as Experiment A
NUM_EVIDENCE_B = len(EVIDENCE_NODES)

# All available ALARM nodes except the primary target
candidate_evidence_nodes = sorted(
    (set(EVIDENCE_NODES) | set(CROSS_TARGET_NODES))
    & set(all_samples.columns)
    - {TARGET_NODE}
)

rng = random.Random(RANDOM_SEED_B)

EVIDENCE_NODES_B = sorted(
    rng.sample(candidate_evidence_nodes, NUM_EVIDENCE_B)
)

# Secondary targets exclude primary target and selected evidence
CROSS_TARGET_NODES_B = sorted(
    set(candidate_evidence_nodes) - set(EVIDENCE_NODES_B)
)

print("\n" + "=" * 60)
print("EXPERIMENT B: RANDOM EVIDENCE")
print("=" * 60)

print(f"Primary target: {TARGET_NODE}")
print(f"Random evidence nodes ({len(EVIDENCE_NODES_B)}):")
print(EVIDENCE_NODES_B)

print(f"Secondary targets ({len(CROSS_TARGET_NODES_B)}):")
print(CROSS_TARGET_NODES_B)

# ============================================================
# Experiment B: Primary Target
# ============================================================

PRIMARY_CSV_B = (
    DATASET_DIR / "combined_test_scenarios_exp_b.csv"
)

primary_df_b = all_samples[
    EVIDENCE_NODES_B + [TARGET_NODE]
].copy()

primary_df_b = primary_df_b.rename(
    columns={TARGET_NODE: "Ground Truth"}
)

primary_df_b["Ground Truth"] = (
    primary_df_b["Ground Truth"].astype(str)
)

primary_df_b = (
    primary_df_b.drop_duplicates().reset_index(drop=True)
)

primary_df_b.insert(
    0, "Scenario #", range(1, len(primary_df_b) + 1)
)

primary_df_b.to_csv(PRIMARY_CSV_B, index=False)

# Retain scenarios correctly inferred by the ground-truth BN
primary_df_b = keep_success_cases(PRIMARY_CSV_B)

print(f"\nExperiment B primary: {len(primary_df_b)} scenarios")

# ============================================================
# Experiment B: Secondary Targets
# ============================================================

secondary_dfs_b = []

for target_index, target in enumerate(
    CROSS_TARGET_NODES_B, start=2
):

    cross_target_csv_b = (
        DATASET_DIR
        / f"combined_test_scenarios_t{target_index}_exp_b.csv"
    )

    cross_df_b = all_samples[
        EVIDENCE_NODES_B + [target]
    ].copy()

    cross_df_b = cross_df_b.rename(
        columns={target: "Ground Truth"}
    )

    cross_df_b["Ground Truth"] = (
        cross_df_b["Ground Truth"].astype(str)
    )

    cross_df_b = (
        cross_df_b.drop_duplicates().reset_index(drop=True)
    )

    cross_df_b.insert(
        0, "Scenario #", range(1, len(cross_df_b) + 1)
    )

    cross_df_b.to_csv(cross_target_csv_b, index=False)

    # Preserve Experiment A's secondary-target filtering
    _, gt_successes_b, _, _ = run_evaluation(
        bn_json,
        cross_target_csv_b,
        target_node=target
    )

    _, flawed_successes_b, _, _ = run_evaluation(
        flawed_bn,
        cross_target_csv_b,
        target_node=target
    )

    success_ids_b = (
        set(pd.to_numeric(gt_successes_b["Scenario"]).astype(int))
        & set(pd.to_numeric(flawed_successes_b["Scenario"]).astype(int))
    )

    cross_df_b = cross_df_b[
        cross_df_b["Scenario #"].isin(success_ids_b)
    ].copy()

    cross_df_b = cross_df_b.reset_index(drop=True)

    cross_df_b["Scenario #"] = range(
        1, len(cross_df_b) + 1
    )

    cross_df_b.to_csv(cross_target_csv_b, index=False)

    print(
        f"{cross_target_csv_b.name} "
        f"({target}): {len(cross_df_b)}"
    )

    cross_df_b["Target"] = target
    secondary_dfs_b.append(cross_df_b)

# ============================================================
# Experiment B: Combined Secondary-Target CSV
# ============================================================

combined_secondary_b = pd.concat(
    secondary_dfs_b,
    ignore_index=True
)

combined_secondary_b = combined_secondary_b.drop(
    columns=["Scenario #"]
)

combined_secondary_b.insert(
    0,
    "Scenario #",
    range(1, len(combined_secondary_b) + 1)
)

COMBINED_SECONDARY_CSV_B = (
    DATASET_DIR
    / "combined_cross_target_test_scenarios_exp_b.csv"
)

combined_secondary_b.to_csv(
    COMBINED_SECONDARY_CSV_B,
    index=False
)

print("\nExperiment B completed.")
print(f"Primary: {len(primary_df_b)} scenarios")
print(
    f"Secondary: {len(combined_secondary_b)} scenarios "
    f"across {len(CROSS_TARGET_NODES_B)} targets"
)
