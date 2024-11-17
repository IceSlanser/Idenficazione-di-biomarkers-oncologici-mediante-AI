import re
from typing import Any, List, Tuple, Iterable

import numpy as np
from graphviz import Source
from numpy.core.multiarray import ndarray
from scipy.stats import entropy
from xgboost import to_graphviz, XGBClassifier


# Build decision tree with graphviz
def build_tree(clf_xgb: XGBClassifier) -> None:
    print("Building decision tree...")
    # Get a tree's info
    booster = clf_xgb.get_booster()
    tree_dump = booster.get_dump(with_stats=True)[0]
    nodes = [node for node in tree_dump.split('\n') if node]

    # Compute new updated source
    node_info = get_node_info(nodes)
    graph_data = to_graphviz(booster, num_trees=0)
    updated_source = update_graph_source(graph_data, node_info)

    # Save and view the decision tree
    new_graph_data = Source('\n'.join(updated_source))
    new_graph_data.view(filename='../test/backups/XGBTree_with_custom_info')


def get_node_info(nodes: list) -> dict:
    print("Memorizing nodes information...")
    node_info = {}
    for raw_node in nodes:
        # Get node id
        id_match = re.match(r'^\s*(\d+)', raw_node)
        node_id = id_match.group(1)
        # Check for leaf node
        if "leaf=" in raw_node:
            # Prepare match
            cover_match = re.search(r'cover=(\d+)', raw_node)

            # Get value
            cover_count = int(cover_match.group(1))

            # Save leaf's info
            node_info[node_id] = {
                'cover': cover_count,
            }

        # Check for node
        else:
            # Prepare matches
            yes_match = re.search(r'yes=(\d+)', raw_node)
            no_match = re.search(r'no=(\d+)', raw_node)
            cover_match = re.search(r'cover=(\d+)', raw_node)

            # Get values
            yes_count = int(yes_match.group(1))
            no_count = int(no_match.group(1))
            cover_count = int(cover_match.group(1))

            # Save node's info
            node_info[node_id] = {
                'yes': yes_count,
                'no': no_count,
                'cover': cover_count,
            }
    return node_info


# Compute old graph's node info
def compute_graph_node_info(node_info: dict, node_id: str) -> tuple[int, list[float], float]:
    # Find node's yes_distribution
    yes_child_node = node_info.get(str(node_info[node_id].get('yes')))
    yes_cover = yes_child_node.get('cover')

    # Find node's no_distribution
    no_child_node = node_info.get(str(node_info[node_id].get('no')))
    no_cover = no_child_node.get('cover')

    # Compute total distribution
    total_cover = yes_cover + no_cover
    sample_distribution = [round((yes_cover / total_cover) * 100, 1), round((no_cover / total_cover) * 100, 1)]

    # Compute entropy
    ent = entropy(sample_distribution)

    return total_cover,sample_distribution, ent

# Update graph_data with new source
def update_graph_source(old_graph_data: Any, node_info:dict) -> list:
    print("Updating source code...")
    # Update graph data's source
    updated_source = []
    nodes = [node for node in old_graph_data.source.split('\n') if node]
    for raw_node in nodes:
        if 'label=' in raw_node:

            # Check for leaf node
            if 'leaf=' in raw_node:
                # Prepare matches
                id_match = re.match(r'^\s*(\d+)', raw_node)
                label_match = re.search(r'label="([^"]+)"', raw_node)

                # Get information
                node_id = id_match.group(1)
                label = label_match.group(1)
                samples = node_info[node_id]['cover']

                # Compute logit value into probability
                value_match = re.search(r'leaf=(-?\d+\.\d+)', label)
                leaf_value = float(value_match.group(1))
                probability = 1 / (1 + np.exp(-leaf_value))
                status = "Ill" if probability > 0.5 else "Healthy"

                # Updated label
                new_label = (
                    f'label="{label}\\n'
                    f'samples: {samples}\\n'
                    f'status: {status}"'
                )
                raw_node = f'{node_id} [ {new_label}, shape="box", style="filled, rounded", fillcolor="#e48038" ]'

            # Check for node
            elif '->' not in raw_node:
                # Prepare matches
                id_match = re.match(r'^\s*(\d+)', raw_node)
                label_match = re.search(r'label="([^"]+)"', raw_node)

                # Get information
                node_id = id_match.group(1)
                label = label_match.group(1)
                samples, dist, ent = compute_graph_node_info(node_info, node_id)

                # Updated label
                new_label = (
                    f'label="{label}\\n'
                    f'samples: {samples}\\n'
                    f'distribution: {dist}\\n'
                    f'entropy: {ent:.2f}"'
                )
                raw_node = f'{node_id} [ {new_label}, shape="box", style="filled, rounded", fillcolor="#78cbe" ]'
        updated_source.append(raw_node)
    return updated_source
