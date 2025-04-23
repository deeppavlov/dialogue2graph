# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/csv_converter.py

import csv
from typing import List, Dict, Any


def convert_annotations_to_csv(
    data_list: List[Dict[str, Any]], output_csv: str
) -> None:
    """
    Convert the list of data to a CSV focusing on the "annotation" -> "annotation" keys.
    Each row is a key, each column is one dialogue index.
    """
    all_keys = set()
    for item in data_list:
        ann_obj = item.get("annotation", {})
        if ann_obj is not None and "annotation" in ann_obj:
            for k in ann_obj["annotation"].keys():
                all_keys.add(k)

    all_keys = list(all_keys)
    all_keys.sort()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header_row = ["key"] + [f"dialog_{i + 1}" for i in range(len(data_list))]
        writer.writerow(header_row)

        for key in all_keys:
            row = [key]
            for item in data_list:
                ann_obj = item.get("annotation", {})
                if ann_obj and "annotation" in ann_obj:
                    val = ann_obj["annotation"].get(key, "")
                else:
                    val = ""
                row.append(val)
            writer.writerow(row)
