import json
import random

random.seed(42)

SUCCESS_FILE = "merged_flawed_success_train.json"
FAILURE_FILE = "merged_flawed_failure_train.json"

RATIO = 8  # Success : Failure

OUTPUT_SUCCESS = f"merged_flawed_success_train_{RATIO}to1.json"
OUTPUT_FAILURE = f"merged_flawed_failure_train_{RATIO}to1.json"

with open(SUCCESS_FILE, "r") as f:
    success_data = json.load(f)

with open(FAILURE_FILE, "r") as f:
    failure_data = json.load(f)

num_success = len(success_data)
num_failure = len(failure_data)

# Determine the maximum number of failures that can be used
selected_failure_count = min(num_failure, num_success // RATIO)
selected_success_count = selected_failure_count * RATIO

# Randomly sample
selected_success = random.sample(success_data, selected_success_count)
selected_failure = random.sample(failure_data, selected_failure_count)

# Save
with open(OUTPUT_SUCCESS, "w") as f:
    json.dump(selected_success, f, indent=2)

with open(OUTPUT_FAILURE, "w") as f:
    json.dump(selected_failure, f, indent=2)

print(f"Original: {num_success} successes, {num_failure} failures")
print(f"Selected: {len(selected_success)} successes, {len(selected_failure)} failures")
print(f"Ratio: {len(selected_success) / len(selected_failure):.1f}:1")
