import pandas as pd

SCENARIO_FILE = "Scenarios.csv"

def load_scenarios():
    return pd.read_csv(SCENARIO_FILE)

def scenarios_to_json(df):
    return df.to_dict(orient="records")

scenarios_df = load_scenarios()
scenarios = scenarios_to_json(scenarios_df)
print(scenarios)
