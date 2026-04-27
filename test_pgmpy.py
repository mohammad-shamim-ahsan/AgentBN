from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bn = {
  "nodes": [
    {
      "name": "GPTN_1",
      "states": ["Found", "Not_Found"],
      "parents": [],
      "cpt": {
        "parent_state_order": {},
        "values": [
          [0.2],
          [0.8]
        ]
      }
    },
    {
      "name": "GPTN_2",
      "states": ["Found", "Not_Found"],
      "parents": [],
      "cpt": {
        "parent_state_order": {},
        "values": [
          [0.25],
          [0.75]
        ]
      }
    },
    {
      "name": "GPTN_3",
      "states": ["Found", "Not_Found"],
      "parents": [],
      "cpt": {
        "parent_state_order": {},
        "values": [
          [0.2],
          [0.8]
        ]
      }
    },
    {
      "name": "GPTN_5",
      "states": ["Found", "Not_Found"],
      "parents": [],
      "cpt": {
        "parent_state_order": {},
        "values": [
          [0.5],
          [0.5]
        ]
      }
    },
    {
      "name": "LPTN_1_i",
      "states": ["Found", "Not_Found"],
      "parents": [],
      "cpt": {
        "parent_state_order": {},
        "values": [
          [0.7],
          [0.3]
        ]
      }
    },
    {
      "name": "LPTN_1_ii",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_i"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_i": ["Found", "Not_Found"]
        },
        "values": [
          [0.9, 0.2],
          [0.1, 0.8]
        ]
      }
    },
    {
      "name": "LPTN_1_iii",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_ii"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_ii": ["Found", "Not_Found"]
        },
        "values": [
          [0.9, 0.2],
          [0.1, 0.8]
        ]
      }
    },
    {
      "name": "LPTN_1_iv",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_iii"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_iii": ["Found", "Not_Found"]
        },
        "values": [
          [0.9, 0.2],
          [0.1, 0.8]
        ]
      }
    },
    {
      "name": "LPTN_1_v",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_iv"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_iv": ["Found", "Not_Found"]
        },
        "values": [
          [0.85, 0.25],
          [0.15, 0.75]
        ]
      }
    },
    {
      "name": "LPTN_1_vi",
      "states": ["Found", "Not_Found"],
      "parents": [],
      "cpt": {
        "parent_state_order": {},
        "values": [
          [0.6],
          [0.4]
        ]
      }
    },
    {
      "name": "LPTN_1_viii",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_vi"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_vi": ["Found", "Not_Found"]
        },
        "values": [
          [0.8, 0.3],
          [0.2, 0.7]
        ]
      }
    },
    {
      "name": "LPTN_1_ix",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_viii"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_viii": ["Found", "Not_Found"]
        },
        "values": [
          [0.8, 0.3],
          [0.2, 0.7]
        ]
      }
    },
    {
      "name": "LPTN_1_x",
      "states": ["Found", "Not_Found"],
      "parents": ["LPTN_1_ix"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_ix": ["Found", "Not_Found"]
        },
        "values": [
          [0.8, 0.3],
          [0.2, 0.7]
        ]
      }
    },
    {
      "name": "Network_Manipulation",
      "states": ["Yes", "No"],
      "parents": ["GPTN_1", "GPTN_2"],
      "cpt": {
        "parent_state_order": {
          "GPTN_1": ["Found", "Not_Found"],
          "GPTN_2": ["Found", "Not_Found"]
        },
        "values": [
          [0.97, 0.9, 0.75, 0.02],
          [0.03, 0.1, 0.25, 0.98]
        ]
      }
    },
    {
      "name": "Deviation_in_Response",
      "states": ["Yes", "No"],
      "parents": ["LPTN_1_i", "LPTN_1_ii", "LPTN_1_iii", "LPTN_1_iv", "LPTN_1_v"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_i": ["Found", "Not_Found"],
          "LPTN_1_ii": ["Found", "Not_Found"],
          "LPTN_1_iii": ["Found", "Not_Found"],
          "LPTN_1_iv": ["Found", "Not_Found"],
          "LPTN_1_v": ["Found", "Not_Found"]
        },
        "values": [
          [0.99, 0.95, 0.95, 0.9, 0.95, 0.9, 0.9, 0.8, 0.95, 0.9, 0.9, 0.8, 0.9, 0.8, 0.8, 0.65, 0.95, 0.9, 0.9, 0.8, 0.9, 0.8, 0.8, 0.65, 0.9, 0.8, 0.8, 0.65, 0.8, 0.65, 0.65, 0.5],
          [0.01, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.2, 0.05, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.35, 0.05, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.35, 0.1, 0.2, 0.2, 0.35, 0.2, 0.35, 0.35, 0.5]
        ]
      }
    },
    {
      "name": "Deviation_in_Dispatch",
      "states": ["Yes", "No"],
      "parents": ["LPTN_1_vi", "LPTN_1_viii", "LPTN_1_ix", "LPTN_1_x"],
      "cpt": {
        "parent_state_order": {
          "LPTN_1_vi": ["Found", "Not_Found"],
          "LPTN_1_viii": ["Found", "Not_Found"],
          "LPTN_1_ix": ["Found", "Not_Found"],
          "LPTN_1_x": ["Found", "Not_Found"]
        },
        "values": [
          [0.98, 0.92, 0.92, 0.82, 0.92, 0.82, 0.82, 0.65, 0.92, 0.82, 0.82, 0.65, 0.82, 0.65, 0.65, 0.45],
          [0.02, 0.08, 0.08, 0.18, 0.08, 0.18, 0.18, 0.35, 0.08, 0.18, 0.18, 0.35, 0.18, 0.35, 0.35, 0.55]
        ]
      }
    },
    {
      "name": "Program_Anomaly",
      "states": ["Yes", "No"],
      "parents": ["GPTN_3"],
      "cpt": {
        "parent_state_order": {
          "GPTN_3": ["Found", "Not_Found"]
        },
        "values": [
          [0.9, 0.1],
          [0.1, 0.9]
        ]
      }
    },
    {
      "name": "Execution_Integrity",
      "states": ["Compromised", "Normal"],
      "parents": ["Program_Anomaly", "Deviation_in_Response", "Deviation_in_Dispatch"],
      "cpt": {
        "parent_state_order": {
          "Program_Anomaly": ["Yes", "No"],
          "Deviation_in_Response": ["Yes", "No"],
          "Deviation_in_Dispatch": ["Yes", "No"]
        },
        "values": [
          [0.99, 0.97, 0.96, 0.9, 0.8, 0.45, 0.4, 0.05],
          [0.01, 0.03, 0.04, 0.1, 0.2, 0.55, 0.6, 0.95]
        ]
      }
    },
    {
      "name": "Physical_Anomaly",
      "states": ["Yes", "No"],
      "parents": ["GPTN_5"],
      "cpt": {
        "parent_state_order": {
          "GPTN_5": ["Found", "Not_Found"]
        },
        "values": [
          [0.9, 0.15],
          [0.1, 0.85]
        ]
      }
    },
    {
      "name": "Root_Causes",
      "states": ["System_Fault", "FDI", "Memory_Corruption"],
      "parents": ["Network_Manipulation", "Execution_Integrity", "Physical_Anomaly"],
      "cpt": {
        "parent_state_order": {
          "Network_Manipulation": ["Yes", "No"],
          "Execution_Integrity": ["Compromised", "Normal"],
          "Physical_Anomaly": ["Yes", "No"]
        },
        "values": [
          [0.05, 0.02, 0.15, 0.05, 0.2, 0.95, 0.75, 0.98],
          [0.55, 0.85, 0.75, 0.2, 0.15, 0.03, 0.1, 0.01],
          [0.4, 0.13, 0.1, 0.75, 0.65, 0.02, 0.15, 0.01]
        ]
      }
    }
  ],
  "edges": [
    ["LPTN_1_i", "LPTN_1_ii"],
    ["LPTN_1_ii", "LPTN_1_iii"],
    ["LPTN_1_iii", "LPTN_1_iv"],
    ["LPTN_1_iv", "LPTN_1_v"],
    ["LPTN_1_vi", "LPTN_1_viii"],
    ["LPTN_1_viii", "LPTN_1_ix"],
    ["LPTN_1_ix", "LPTN_1_x"],
    ["GPTN_1", "Network_Manipulation"],
    ["GPTN_2", "Network_Manipulation"],
    ["LPTN_1_i", "Deviation_in_Response"],
    ["LPTN_1_ii", "Deviation_in_Response"],
    ["LPTN_1_iii", "Deviation_in_Response"],
    ["LPTN_1_iv", "Deviation_in_Response"],
    ["LPTN_1_v", "Deviation_in_Response"],
    ["LPTN_1_vi", "Deviation_in_Dispatch"],
    ["LPTN_1_viii", "Deviation_in_Dispatch"],
    ["LPTN_1_ix", "Deviation_in_Dispatch"],
    ["LPTN_1_x", "Deviation_in_Dispatch"],
    ["GPTN_3", "Program_Anomaly"],
    ["Program_Anomaly", "Execution_Integrity"],
    ["Deviation_in_Response", "Execution_Integrity"],
    ["Deviation_in_Dispatch", "Execution_Integrity"],
    ["GPTN_5", "Physical_Anomaly"],
    ["Network_Manipulation", "Root_Causes"],
    ["Execution_Integrity", "Root_Causes"],
    ["Physical_Anomaly", "Root_Causes"]
  ]
}

# -----------------------------
# Build structure
# -----------------------------
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


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

# -----------------------------
def run_inference(model, query, evidence=None):
    infer = VariableElimination(model)

    result = infer.query(
        variables=[query],
        evidence=evidence or {}
    )

    return result

# -----------------------------
# RUN
# -----------------------------
model = build_model(bn)
evidence = {
    "GPTN_1": "Not_Found",
    "GPTN_2": "Not_Found",
    "GPTN_3": "Not_Found",
    "GPTN_5": "Found",
    "LPTN_1_i": "Found",
    "LPTN_1_ii": "Found",
    "LPTN_1_iii": "Found",
    "LPTN_1_iv": "Found",
    "LPTN_1_v": "Found",
    "LPTN_1_vi": "Found",
    "LPTN_1_viii": "Found",
    "LPTN_1_ix": "Found",
    "LPTN_1_x": "Found"
}

result = run_inference(
    model,
    query="Root_Causes",
    evidence=evidence
)

print(result)

