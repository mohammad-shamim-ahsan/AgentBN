import pandas as pd
from sklearn.model_selection import train_test_split

from automatic_reasoning_simple import run_evaluation, load_bn
from config.settings import *


### ------------------------ Environment Settings Start --------------------

print(f"BENCHMARK = {BENCHMARK!r}")

DATASET_DIR = Path("datasets") / BENCHMARK

TRAIN_CSV = DATASET_DIR / "combined_train_scenarios.csv"
TEST_CSV = DATASET_DIR / "combined_test_scenarios.csv"


TARGET_NODE = "HYPOVOLEMIA"

EVIDENCE_NODES = {
    "BP",
    "HRBP",
    "HREKG",
    "HRSAT",
    "EXPCO2",
    "CVP",
    "PCWP",
}

df = pd.read_csv(DATASET_DIR / "alarm.0.csv")
print(df.shape)

# Keep only target + evidence columns
columns_to_keep = [TARGET_NODE] + list(EVIDENCE_NODES)

df = df[columns_to_keep]

# Rename target states
df["HYPOVOLEMIA"] = df["HYPOVOLEMIA"].map({
    True: "Hypovolemia",
    False: "No_Hypovolemia"
})

# Rename target column
df = df.rename(columns={"HYPOVOLEMIA": "Ground Truth"})

# Add Scenario # starting from 1
df.insert(0, "Scenario #", range(1, len(df) + 1))

# Ensure exact column order
df = df[
    [
        "Scenario #",
        "BP",
        "HRBP",
        "HREKG",
        "HRSAT",
        "EXPCO2",
        "CVP",
        "PCWP",
        "Ground Truth",
    ]
]

print(df.shape)

### ----------------------- Environment Settings End -------------------------

duplicate_count = df.drop(columns=["Scenario #"]).duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

# Remove Scenario # before checking duplicates
df = df.drop(columns=["Scenario #"])

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Regenerate Scenario #
df.insert(0, "Scenario #", range(1, len(df) + 1))

print(f"Number of unique scenarios: {len(df)}")

print(df.shape)

### --- Splitting

# Check class distribution
print("\nGround Truth distribution:")
print(df["Ground Truth"].value_counts())

# 80/20 stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=df["Ground Truth"]
)

# Reset indexes
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Regenerate Scenario # separately for each dataset
train_df["Scenario #"] = range(1, len(train_df) + 1)
test_df["Scenario #"] = range(1, len(test_df) + 1)

# Save
train_df.to_csv(DATASET_DIR / "combined_train_scenarios.csv", index=False)
test_df.to_csv(DATASET_DIR / "combined_test_scenarios.csv", index=False)

print(f"\nTotal : {len(df)}")
print(f"Train : {len(train_df)}")
print(f"Test  : {len(test_df)}")

print("\nTrain distribution:")
print(train_df["Ground Truth"].value_counts())

print("\nTest distribution:")
print(test_df["Ground Truth"].value_counts())


### ----------------

GT_BN_FILE = DATASET_DIR / "BN_gt.json"

bn_json = load_bn(GT_BN_FILE)


def keep_success_cases(csv_file):

    failures, successes, accuracy, results = run_evaluation(
        bn_json,
        csv_file
    )

    df = pd.read_csv(csv_file)

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


# Filter both datasets using ground-truth BN
train_df = keep_success_cases(TRAIN_CSV)
test_df = keep_success_cases(TEST_CSV)

print("\nFinal dataset sizes:")
print(f"Train: {len(train_df)}")
print(f"Test : {len(test_df)}")
