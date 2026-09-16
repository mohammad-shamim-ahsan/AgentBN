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


def store_new_bn(
    bn_number,
    bn_new,
    filename=PROPOSED_BN_FILE,
    overwrite=False,
    metadata=None,
):
    record = {
        "bn_number": bn_number,
        "bn": bn_new
    }

    if metadata is not None:
        record["metadata"] = metadata

    if overwrite and os.path.exists(filename):
        records = []

        new_oracle_size = (
            metadata.get("oracle_size")
            if metadata is not None
            else None
        )

        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                old_record = json.loads(line)

                old_oracle_size = (
                    old_record
                    .get("metadata", {})
                    .get("oracle_size")
                )

                same_record = (
                    old_record.get("bn_number") == bn_number
                    and old_oracle_size == new_oracle_size
                )

                if not same_record:
                    records.append(old_record)

        records.append(record)

        with open(filename, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item) + "\n")

    else:
        with open(filename, "a", encoding="utf-8") as f:
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


def get_bn(
    path,
    bn_number=None,
    oracle_size=None,
):
    last_record = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if bn_number is not None:
                if record.get("bn_number") != bn_number:
                    continue

            if oracle_size is not None:
                if (
                    record.get("metadata", {}).get("oracle_size")
                    != oracle_size
                ):
                    continue

            return record["bn"]

            last_record = record

    raise ValueError(
        f"BN not found for "
        f"bn_number={bn_number}, "
        f"oracle_size={oracle_size}."
    )


def normalize_bn(bn_obj):
    if "bn" in bn_obj and "nodes" in bn_obj["bn"]:
        return {n["name"]: n for n in bn_obj["bn"]["nodes"]}

    if "nodes" in bn_obj:
        return {n["name"]: n for n in bn_obj["nodes"]}

    return bn_obj


def read_all_analysis_records(filename=PROPOSED_BN_FILE):
    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except Exception:
                continue

    return records


def get_best_bn_number(filename=PROPOSED_BN_FILE):
    records = read_all_analysis_records(filename)

    if not records:
        raise ValueError("No analysis records found.")

    best_record = min(
        records,
        key=lambda r: r.get("failure_count", float("inf"))
    )

    return best_record["bn_number"]


def get_analysis_record(bn_number, filename=BN_ANALYSIS_FILE):
    records = read_all_analysis_records(filename)

    for record in records:
        if record.get("bn_number") == bn_number:
            return record

    raise ValueError(f"No analysis record found for BN #{bn_number}")
