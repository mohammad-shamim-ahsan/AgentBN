import os
import json

from config.settings import *


def load_bn(filename):
    with open(filename, "r") as f:
        return json.load(f)
    

def remove_bn(bn_number, filename=PROPOSED_BN_FILE):
    if not os.path.exists(filename):
        print(f"{filename} does not exist.")
        return

    records = []
    removed_count = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if record.get("bn_number") == bn_number:
                    removed_count += 1
                else:
                    records.append(record)

            except json.JSONDecodeError:
                continue

    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(
        f"Removed {removed_count} record(s) "
        f"with bn_number={bn_number}"
    )


def store_new_bn(bn_number, bn_new):
    record = {
        "bn_number": bn_number,
        "bn": bn_new
    }

    with open(PROPOSED_BN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def find_proposed_bn(bn_number, filename=PROPOSED_BN_FILE):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("bn_number") == bn_number:
                return record["bn"]

    raise ValueError(f"No proposed BN found for BN #{bn_number}")


def get_bn(path, bn_number=None):
    last_record = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if bn_number is not None and record.get("bn_number") == bn_number:
                return record["bn"]

            last_record = record

    if bn_number is not None:
        raise ValueError(f"BN #{bn_number} not found.")

    if last_record is None:
        raise ValueError("No BN records found.")

    return last_record["bn"]


def normalize_bn(bn_obj):
    if "bn" in bn_obj and "nodes" in bn_obj["bn"]:
        return {n["name"]: n for n in bn_obj["bn"]["nodes"]}

    if "nodes" in bn_obj:
        return {n["name"]: n for n in bn_obj["nodes"]}

    return bn_obj