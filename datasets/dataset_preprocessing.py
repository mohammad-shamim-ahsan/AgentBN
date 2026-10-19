import pandas as pd
from sklearn.model_selection import train_test_split

from automatic_reasoning_simple import run_evaluation, load_bn
from config.settings import *


### ------------------------ Environment Settings Start --------------------

print(f"BENCHMARK = {BENCHMARK!r}")

DATASET_DIR = Path("datasets") / BENCHMARK

TRAIN_CSV = DATASET_DIR / "combined_train_scenarios.csv"
TEST_CSV = DATASET_DIR / "combined_test_scenarios.csv"

# ============================================================
# Select multiple ALARM CSV files
# ============================================================

CSV_DIR = DATASET_DIR / "csv"

CSV_FILES = sorted(CSV_DIR.glob(f"{BENCHMARK}.*.csv"))

# ============================================================
# Load and combine all selected CSVs
# ============================================================

dfs = []

for csv_file in CSV_FILES:
    df_i = pd.read_csv(csv_file, dtype=str)
    dfs.append(df_i)

df = pd.concat(dfs, ignore_index=True)

# Preserve raw rows for independent T2–T5 preprocessing.
all_samples = df.copy()

print(f"Number of CSVs: {len(CSV_FILES)}")
print(f"Combined raw shape: {df.shape}")


# ============================================================
# Keep only target + evidence columns
# ============================================================

columns_to_keep = [TARGET_NODE] + list(EVIDENCE_NODES)

df = df[columns_to_keep]


# ============================================================
# Preserve original BIF state labels (TRUE/FALSE) as strings
# ============================================================

df[TARGET_NODE] = df[TARGET_NODE].astype(str)


# ============================================================
# Rename target column
# ============================================================

df = df.rename(columns={TARGET_NODE: "Ground Truth"})


# ============================================================
# Add Scenario # starting from 1
# ============================================================

df.insert(0, "Scenario #", range(1, len(df) + 1))


# ============================================================
# Ensure exact column order
# ============================================================

df = df[["Scenario #"] + sorted(EVIDENCE_NODES) + ["Ground Truth"]]

print(f"Shape after column selection: {df.shape}")


### ----------------------- Environment Settings End -------------------------


# ============================================================
# Remove duplicate scenarios across ALL selected CSVs
# ============================================================

duplicate_count = df.drop(columns=["Scenario #"]).duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")


# Remove Scenario # before duplicate removal
df = df.drop(columns=["Scenario #"])


# Global duplicate removal
df = df.drop_duplicates().reset_index(drop=True)


# Regenerate Scenario #
df.insert(0, "Scenario #", range(1, len(df) + 1))


print(f"Number of unique scenarios: {len(df)}")
print(df.shape)


# ============================================================
# Train/Test Split
# ============================================================

print("\nGround Truth distribution:")
print(df["Ground Truth"].value_counts())


# 80/20 stratified split
stratify_labels = df["Ground Truth"]

if BENCHMARK == "alarm" and TARGET_NODE == "HYPOVOLEMIA":
    stratify_labels = stratify_labels.map({
        "TRUE": "Hypovolemia",
        "FALSE": "No_Hypovolemia"
    })

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=stratify_labels
)


# Reset indexes
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Regenerate Scenario # separately for each dataset
train_df["Scenario #"] = range(1, len(train_df) + 1)
test_df["Scenario #"] = range(1, len(test_df) + 1)


# ============================================================
# Save combined train/test scenarios before GT filtering
# ============================================================

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)


print(f"\nTotal : {len(df)}")
print(f"Train : {len(train_df)}")
print(f"Test  : {len(test_df)}")


print("\nTrain distribution:")
print(train_df["Ground Truth"].value_counts())


print("\nTest distribution:")
print(test_df["Ground Truth"].value_counts())


# ============================================================
# Load Ground-Truth BN
# ============================================================

bn_json = load_bn(GROUND_TRUTH_BN_FILE)


# ============================================================
# Keep only scenarios successfully inferred by GT BN
# ============================================================

def keep_success_cases(csv_file):

    failures, successes, accuracy, results = run_evaluation(
        bn_json,
        csv_file
    )

    df = pd.read_csv(csv_file, dtype={"Ground Truth": str})

    # Get successful scenario IDs
    success_ids = successes["Scenario"]

    # Keep only successful scenarios
    df = df[df["Scenario #"].isin(success_ids)].copy()

    # Reset and renumber
    df = df.reset_index(drop=True)
    df["Scenario #"] = range(1, len(df) + 1)

    # Overwrite original file
    df.to_csv(csv_file, index=False)

    print(f"\n{csv_file.name}")
    print(f"Original cases : {len(results)}")
    print(f"Success cases  : {len(successes)}")
    print(f"Failure cases  : {len(failures)}")
    print(f"Kept           : {len(df)}")

    return df


# ============================================================
# Filter both combined datasets using Ground-Truth BN
# ============================================================
train_df = keep_success_cases(TRAIN_CSV)
test_df = keep_success_cases(TEST_CSV)


# ============================================================
# Final Dataset Sizes
# ============================================================
print("\nFinal dataset sizes:")
print(f"Train: {len(train_df)}")
print(f"Test : {len(test_df)}")


# ============================================================
# Generate and filter independent cross-target test datasets
# ============================================================

flawed_bn = load_bn(FLAWED_BN_FILE)

for target_index, target in enumerate(CROSS_TARGET_NODES, start=2):
    cross_target_csv = DATASET_DIR / f"combined_test_scenarios_t{target_index}.csv"

    # Use the same evidence variables, with target-specific labels.
    cross_df = all_samples[sorted(EVIDENCE_NODES) + [target]].copy()
    cross_df = cross_df.rename(columns={target: "Ground Truth"})
    cross_df["Ground Truth"] = cross_df["Ground Truth"].astype(str)

    # Independently deduplicate evidence + ground-truth combinations.
    cross_df = cross_df.drop_duplicates().reset_index(drop=True)
    cross_df.insert(0, "Scenario #", range(1, len(cross_df) + 1))
    cross_df.to_csv(cross_target_csv, index=False)

    _, gt_successes, _, _ = run_evaluation(
        bn_json, cross_target_csv, target_node=target
    )
    _, flawed_successes, _, _ = run_evaluation(
        flawed_bn, cross_target_csv, target_node=target
    )

    success_ids = set(gt_successes["Scenario"]) & set(flawed_successes["Scenario"])
    cross_df = cross_df[cross_df["Scenario #"].isin(success_ids)].copy()
    cross_df = cross_df.reset_index(drop=True)
    cross_df["Scenario #"] = range(1, len(cross_df) + 1)
    cross_df.to_csv(cross_target_csv, index=False)
    print(f"{cross_target_csv.name}: {len(cross_df)}")
