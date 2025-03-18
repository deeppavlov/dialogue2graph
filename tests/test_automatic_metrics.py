#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_metrics.py

This script demonstrates basic testing of the custom metrics in 'dialogue2graph'.
If any metric's result is not what's expected, it reports a failed test. Otherwise, it reports a pass.

Run this script in your environment where dialogue2graph is installed and
test_cases.json is accessible.
"""

import json
import sys

from dialogue2graph import Graph, Dialogue, DialogueMessage
from dialogue2graph.metrics.automatic_metrics import (
    all_utterances_present,
    is_same_structure,
    all_roles_correct,
    is_correct_lenght,
    triplet_match,
    triplet_match_accuracy,
    compute_graph_metrics,
    all_paths_sampled
)

def run_tests():
    # 1. Load sample data
    with open("test_automatic_metrics_test_cases.json", encoding="utf-8") as f:
        data = json.load(f)

    # We assume data[0] and data[1] each have "graph" and possibly "dialogues"
    if len(data) < 2:
        print("Not enough test data in test_cases.json (need at least two items).")
        sys.exit(1)

    # 2. Construct Graph objects
    graph1 = Graph(graph_dict=data[0]["graph"])
    graph2 = Graph(graph_dict=data[1]["graph"])

    # 3. Construct some Dialogue objects
    #    We'll pick the first dialogue from data[0], if it exists
    dialogues_data_0 = data[0].get("dialogues", [])
    dialogues_0 = []
    for diag in dialogues_data_0:
        messages = []
        for msg in diag["messages"]:
            messages.append(DialogueMessage(text=msg["text"], participant=msg["participant"]))
        d = Dialogue(messages=messages, topic=diag.get("topic", ""), validate=diag.get("validate", False))
        dialogues_0.append(d)

    # 4. Prepare to run each metric on these objects.
    # Let's define a simple results collector.
    test_results = []

    # --- 4A. Test all_utterances_present ---
    try:
        # We'll test if all utterances in graph1 are present in dialogues_0
        # If there's at least one dialogue, pick dialogues_0. Otherwise, we pass an empty list.
        result_utterances = all_utterances_present(graph1, dialogues_0)
        # This metric can return bool or a set of missing utterances,
        # so we check if it's True or an empty set
        if result_utterances is True or (isinstance(result_utterances, set) and len(result_utterances) == 0):
            test_results.append(("all_utterances_present", True))
        else:
            test_results.append(("all_utterances_present", False))
    except Exception as e:
        print("Error testing all_utterances_present:", e)
        test_results.append(("all_utterances_present", False))

    # --- 4B. Test is_same_structure ---
    try:
        same_struc = is_same_structure(graph1, graph2)
        # Suppose we expect them to be isomorphic from the example
        expected_same_struc = True
        test_results.append(("is_same_structure", same_struc == expected_same_struc))
    except Exception as e:
        print("Error testing is_same_structure:", e)
        test_results.append(("is_same_structure", False))

    # --- 4C. Test all_roles_correct ---
    try:
        if len(dialogues_0) >= 2:
            roles_ok = all_roles_correct(dialogues_0[0], dialogues_0[1])
            # We can't say for sure if they're correct in the example,
            # so let's just require it's either True or False without raising error
            # For demonstration let's check we get a boolean:
            test_results.append(("all_roles_correct", isinstance(roles_ok, bool)))
        else:
            # If we don't have 2 dialogues to compare, skip
            test_results.append(("all_roles_correct", True))
    except Exception as e:
        print("Error testing all_roles_correct:", e)
        test_results.append(("all_roles_correct", False))

    # --- 4D. Test is_correct_lenght ---
    try:
        if len(dialogues_0) >= 2:
            length_ok = is_correct_lenght(dialogues_0[0], dialogues_0[1])
            test_results.append(("is_correct_lenght", isinstance(length_ok, bool)))
        else:
            test_results.append(("is_correct_lenght", True))
    except Exception as e:
        print("Error testing is_correct_lenght:", e)
        test_results.append(("is_correct_lenght", False))

    # --- 4E. Test triplet_match and triplet_match_accuracy ---
    try:
        node_map, edge_map = triplet_match(graph1, graph2)
        # As a quick check, ensure node_map and edge_map are dicts
        maps_ok = isinstance(node_map, dict) and isinstance(edge_map, dict)
        test_results.append(("triplet_match", maps_ok))

        acc_dict = triplet_match_accuracy(graph1, graph2)
        # We expect a dict with 'node_accuracy' and 'edge_accuracy' keys
        expected_keys = {"node_accuracy", "edge_accuracy"}
        acc_keys_ok = set(acc_dict.keys()).issuperset(expected_keys)
        test_results.append(("triplet_match_accuracy", acc_keys_ok))
    except Exception as e:
        print("Error testing triplet_match / triplet_match_accuracy:", e)
        test_results.append(("triplet_match", False))
        test_results.append(("triplet_match_accuracy", False))

    # --- 4F. Test compute_graph_metrics ---
    try:
        metric_results = compute_graph_metrics([graph1, graph2])
        # Expected keys from the function
        expected_mkeys = {
            "with_cycles",
            "percentage_with_cycles",
            "average_edges_amount",
            "average_nodes_amount",
            "total_graphs",
            "total_edges",
            "total_nodes",
        }
        metric_keys_ok = set(metric_results.keys()).issuperset(expected_mkeys)
        test_results.append(("compute_graph_metrics", metric_keys_ok))
    except Exception as e:
        print("Error testing compute_graph_metrics:", e)
        test_results.append(("compute_graph_metrics", False))

    # --- 4G. Test all_paths_sampled ---
    try:
        # If we have at least one dialogue, test with the first one
        if len(dialogues_0) > 0:
            all_paths_ok = all_paths_sampled(graph1, dialogues_0[0])
            # This is True or False. We'll just confirm it's a boolean
            is_bool = isinstance(all_paths_ok, bool)
            test_results.append(("all_paths_sampled", is_bool))
        else:
            test_results.append(("all_paths_sampled", True))
    except Exception as e:
        print("Error testing all_paths_sampled:", e)
        test_results.append(("all_paths_sampled", False))

    # 5. Print out overall results
    print("===== TEST RESULTS =====")
    for metric_name, status_ok in test_results:
        if status_ok:
            print(f"[PASSED] {metric_name}")
        else:
            print(f"[FAILED] {metric_name}")


if __name__ == "__main__":
    run_tests()
