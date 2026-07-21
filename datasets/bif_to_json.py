import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from pgmpy.readwrite import BIFReader


# ============================================================
# BIF -> BN_gt.json
# ============================================================

def convert_bif_to_json(bif_file, output_file):
    bif_file = Path(bif_file)
    output_file = Path(output_file)

    reader = BIFReader(str(bif_file))
    model = reader.get_model()

    edges = [
        [str(parent), str(child)]
        for parent, child in model.edges()
    ]

    nodes = []

    for node in model.nodes():
        node = str(node)
        cpd = model.get_cpds(node)

        if cpd is None:
            raise ValueError(f"No CPT found for node: {node}")

        states = [str(state) for state in cpd.state_names[node]]
        parents = [str(parent) for parent in cpd.variables[1:]]

        node_data = {
            "name": node,
            "states": states
        }

        if parents:
            node_data["parents"] = parents
            node_data["cpt"] = {
                "parent_state_order": {
                    parent: [
                        str(state)
                        for state in cpd.state_names[parent]
                    ]
                    for parent in parents
                },
                "values": cpd.get_values().tolist()
            }
        else:
            node_data["cpt"] = {
                "values": cpd.get_values().reshape(
                    len(states), 1
                ).tolist()
            }

        nodes.append(node_data)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "edges": edges,
                "nodes": nodes
            },
            file,
            indent=2
        )

    print(f"Saved JSON to: {output_file}")


# ============================================================
# Helper functions for XDSL
# ============================================================

def format_probability(value):
    """
    Produce clean numeric text accepted by GeNIe.
    """
    value = float(value)

    if math.isclose(value, round(value), abs_tol=1e-12):
        return str(int(round(value)))

    return format(value, ".15g")


def flatten_cpt_for_xdsl(values):
    """
    JSON stores CPTs as:

        rows    = child states
        columns = parent configurations

    XDSL expects:

        config 1: all child-state probabilities
        config 2: all child-state probabilities
        ...

    Therefore, the matrix must be flattened column by column.
    """
    if not values:
        raise ValueError("CPT values cannot be empty.")

    row_count = len(values)
    column_count = len(values[0])

    for row in values:
        if len(row) != column_count:
            raise ValueError("CPT rows do not have equal lengths.")

    flattened = []

    for column_index in range(column_count):
        for row_index in range(row_count):
            flattened.append(values[row_index][column_index])

    return flattened


def topologically_order_nodes(nodes):
    """
    Return nodes in parent-before-child order.

    GeNIe resolves parent references while reading the <nodes> section.
    Therefore, every parent CPT must be written before any child CPT that
    references it.
    """
    node_map = {node["name"]: node for node in nodes}

    if len(node_map) != len(nodes):
        raise ValueError("Duplicate node names found in the Bayesian network.")

    indegree = {name: 0 for name in node_map}
    children = {name: [] for name in node_map}

    for node in nodes:
        child = node["name"]

        for parent in node.get("parents", []):
            if parent not in node_map:
                raise ValueError(
                    f"Node '{child}' refers to unknown parent '{parent}'."
                )

            indegree[child] += 1
            children[parent].append(child)

    original_order = {
        node["name"]: index
        for index, node in enumerate(nodes)
    }

    ready = sorted(
        (name for name, degree in indegree.items() if degree == 0),
        key=original_order.get
    )

    ordered_names = []

    while ready:
        current = ready.pop(0)
        ordered_names.append(current)

        for child in children[current]:
            indegree[child] -= 1

            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=original_order.get)

    if len(ordered_names) != len(nodes):
        unresolved = [
            name for name, degree in indegree.items()
            if degree > 0
        ]
        raise ValueError(
            "The network contains a cycle or unresolved parent references: "
            + ", ".join(unresolved)
        )

    return [node_map[name] for name in ordered_names]


def compute_node_levels(nodes):
    """
    Assign each node to a graph level for a simple left-to-right layout.
    """
    parent_map = {
        node["name"]: list(node.get("parents", []))
        for node in nodes
    }

    levels = {}

    def determine_level(node_name, visiting=None):
        if node_name in levels:
            return levels[node_name]

        if visiting is None:
            visiting = set()

        if node_name in visiting:
            raise ValueError(
                f"Cycle detected while laying out node: {node_name}"
            )

        visiting.add(node_name)

        parents = parent_map[node_name]

        if not parents:
            level = 0
        else:
            level = 1 + max(
                determine_level(parent, visiting.copy())
                for parent in parents
            )

        levels[node_name] = level
        return level

    for node_name in parent_map:
        determine_level(node_name)

    return levels


def compute_positions(nodes):
    """
    Create GeNIe node rectangles:

        left top right bottom
    """
    levels = compute_node_levels(nodes)

    nodes_by_level = {}

    for node in nodes:
        level = levels[node["name"]]
        nodes_by_level.setdefault(level, []).append(node["name"])

    positions = {}

    node_width = 150
    node_height = 70

    horizontal_gap = 100
    vertical_gap = 55

    start_x = 40
    start_y = 40

    for level in sorted(nodes_by_level):
        level_nodes = nodes_by_level[level]

        for row, node_name in enumerate(level_nodes):
            left = start_x + level * (
                node_width + horizontal_gap
            )
            top = start_y + row * (
                node_height + vertical_gap
            )

            right = left + node_width
            bottom = top + node_height

            positions[node_name] = (
                left,
                top,
                right,
                bottom
            )

    return positions


def indent_xml(element, level=0):
    """
    Add readable indentation without changing XML content.
    """
    indentation = "\n" + level * "\t"

    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "\t"

        for child in element:
            indent_xml(child, level + 1)

        if not child.tail or not child.tail.strip():
            child.tail = indentation

    if level and (
        not element.tail or not element.tail.strip()
    ):
        element.tail = indentation


# ============================================================
# BN_gt.json -> GeNIe XDSL
# ============================================================

def convert_json_to_xdsl(
    json_file,
    xdsl_file,
    network_id=None,
    network_name=None,
):
    json_file = Path(json_file)
    xdsl_file = Path(xdsl_file)

    with json_file.open("r", encoding="utf-8") as file:
        bn = json.load(file)

    if network_id is None:
        network_id = json_file.stem

    if network_name is None:
        network_name = json_file.stem

    nodes = bn["nodes"]
    ordered_nodes = topologically_order_nodes(nodes)

    node_names = {node["name"] for node in nodes}

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    for node in nodes:
        node_name = node["name"]

        for parent in node.get("parents", []):
            if parent not in node_names:
                raise ValueError(
                    f"Node '{node_name}' refers to unknown "
                    f"parent '{parent}'."
                )

        values = node["cpt"]["values"]

        expected_rows = len(node["states"])

        if len(values) != expected_rows:
            raise ValueError(
                f"CPT for '{node_name}' has {len(values)} rows; "
                f"expected {expected_rows}."
            )

        for column_index in range(len(values[0])):
            column_sum = sum(
                float(values[row_index][column_index])
                for row_index in range(len(values))
            )

            if not math.isclose(
                column_sum,
                1.0,
                abs_tol=1e-6
            ):
                raise ValueError(
                    f"CPT column {column_index} for "
                    f"'{node_name}' sums to {column_sum}, not 1."
                )

    # --------------------------------------------------------
    # Root element
    # --------------------------------------------------------

    root = ET.Element(
        "smile",
        {
            "version": "1.0",
            "id": network_id,
            "numsamples": "10000",
            "discsamples": "10000"
        }
    )

    nodes_element = ET.SubElement(root, "nodes")

    # --------------------------------------------------------
    # Probabilistic nodes
    # --------------------------------------------------------
    # GeNIe requires every parent CPT to appear before its children.
    for node in ordered_nodes:
        node_name = node["name"]

        cpt_element = ET.SubElement(
            nodes_element,
            "cpt",
            {"id": node_name}
        )

        for state in node["states"]:
            ET.SubElement(
                cpt_element,
                "state",
                {"id": str(state)}
            )

        parents = node.get("parents", [])

        if parents:
            parents_element = ET.SubElement(
                cpt_element,
                "parents"
            )
            parents_element.text = " ".join(parents)

        flattened_values = flatten_cpt_for_xdsl(
            node["cpt"]["values"]
        )

        probabilities_element = ET.SubElement(
            cpt_element,
            "probabilities"
        )

        probabilities_element.text = " ".join(
            format_probability(value)
            for value in flattened_values
        )

    # --------------------------------------------------------
    # GeNIe visualization metadata
    # --------------------------------------------------------

    extensions_element = ET.SubElement(root, "extensions")

    genie_element = ET.SubElement(
        extensions_element,
        "genie",
        {
            "version": "1.0",
            "app": "GeNIe 5.0",
            "name": network_name
        }
    )

    positions = compute_positions(nodes)

    for node in ordered_nodes:
        node_name = node["name"]

        visual_node = ET.SubElement(
            genie_element,
            "node",
            {"id": node_name}
        )

        name_element = ET.SubElement(
            visual_node,
            "name"
        )
        name_element.text = node_name

        ET.SubElement(
            visual_node,
            "interior",
            {"color": "e5f6f7"}
        )

        ET.SubElement(
            visual_node,
            "outline",
            {"color": "000000"}
        )

        ET.SubElement(
            visual_node,
            "font",
            {
                "color": "000000",
                "name": "Arial",
                "size": "11"
            }
        )

        left, top, right, bottom = positions[node_name]

        position_element = ET.SubElement(
            visual_node,
            "position"
        )
        position_element.text = (
            f"{left} {top} {right} {bottom}"
        )

        ET.SubElement(
            visual_node,
            "barchart",
            {
                "active": "true",
                "width": "128",
                "height": "64"
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    indent_xml(root)

    xml_bytes = ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True
    )

    xdsl_file.parent.mkdir(parents=True, exist_ok=True)

    with xdsl_file.open("wb") as file:
        file.write(xml_bytes)

    print(f"Saved GeNIe XDSL to: {xdsl_file}")


# ============================================================
# BIF -> JSON and optional XDSL
# ============================================================

def convert_bif(
    bif_file,
    json_file,
    xdsl_file=None,
    network_id=None,
    network_name=None,
):
    
    bif_file = Path(bif_file)
    json_file = Path(json_file)

    if xdsl_file is not None:
        xdsl_file = Path(xdsl_file)

    if network_id is None:
        network_id = bif_file.stem

    if network_name is None:
        network_name = bif_file.stem

    convert_bif_to_json(
        bif_file=bif_file,
        output_file=json_file
    )

    if xdsl_file is not None:
        convert_json_to_xdsl(
            json_file=json_file,
            xdsl_file=xdsl_file,
            network_id=network_id,
            network_name=network_name
        )

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    directory = Path("datasets/lung_cancer")

    convert_bif(
        bif_file=directory / "asia.bif",
        json_file=directory / "BN_gt.json",
        xdsl_file=directory / "BN_gt.xdsl",
    )
