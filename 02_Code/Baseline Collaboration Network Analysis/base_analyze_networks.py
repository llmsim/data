
import argparse
import csv
from collections import Counter, defaultdict, deque
from math import comb
from pathlib import Path
from random import Random

import numpy as np
from openpyxl import load_workbook


DEFAULT_FOLDER = Path(r"F:\workspace1")
EDGE_FILES = {
    "mission": "edge_data_mission(2).csv",
    "diffusion": "edge_data_diffusion(2).csv",
}


def read_nodes(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.values
        header = next(rows, ())
        if "Id" not in header or "Type" not in header:
            raise ValueError(f"{path.name}: expected Id and Type columns.")
        id_column = header.index("Id")
        type_column = header.index("Type")
        nodes = {}
        for row_number, row in enumerate(rows, start=2):
            if all(value is None for value in row):
                continue
            raw_id = row[id_column]
            org_type = row[type_column]
            node_id = int(str(raw_id).strip())
            if node_id in nodes or not isinstance(org_type, str) or not org_type.strip():
                raise ValueError(f"{path.name}:{row_number}: duplicate Id or missing Type.")
            nodes[node_id] = org_type.strip()
    finally:
        workbook.close()
    if len(nodes) != 146:
        raise ValueError(f"{path.name}: expected 146 unique nodes, found {len(nodes)}.")
    return nodes


def read_graph(path, node_ids):
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    adjacency = np.zeros((len(index), len(index)), dtype=float)
    loops = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"Source", "Target"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: expected Source and Target columns.")
        for row in reader:
            a, b = int(row["Source"]), int(row["Target"])
            if a not in index or b not in index:
                raise ValueError(f"{path.name}:{reader.line_num}: unknown endpoint ({a}, {b}).")
            if row.get("Type", "Undirected").strip().lower() != "undirected":
                raise ValueError(f"{path.name}:{reader.line_num}: expected an undirected tie.")
            if a == b:
                loops.add(a)
            else:
                # Repeated rows and reversed copies refer to the same unweighted tie.
                adjacency[index[a], index[b]] = adjacency[index[b], index[a]] = 1
    return adjacency, loops


def louvain_score(adjacency, seed):

    graph = adjacency.copy()
    total_degree = graph.sum()
    rng = Random(seed)
    while True:
        size = len(graph)
        degrees = graph.sum(axis=1)
        community_degrees = degrees.copy()
        communities = list(range(size))
        neighbors = [np.flatnonzero(row) for row in graph]
        order = list(range(size))
        rng.shuffle(order)

        while True:
            moved = False
            for i in order:
                current = communities[i]
                weights = defaultdict(float)
                for j in neighbors[i]:
                    if j != i:
                        weights[communities[j]] += graph[i, j]
                community_degrees[current] -= degrees[i]
                best = current
                best_gain = weights[current] - degrees[i] * community_degrees[current] / total_degree
                for community, weight in weights.items():
                    gain = weight - degrees[i] * community_degrees[community] / total_degree
                    if gain > best_gain + 1e-12:
                        best, best_gain = community, gain
                communities[i] = best
                community_degrees[best] += degrees[i]
                moved |= best != current
            if not moved:
                break

        _, labels = np.unique(communities, return_inverse=True)
        count = int(labels.max()) + 1
        if count == size:
            internal_fraction = np.trace(graph) / total_degree
            expected_fraction = ((degrees / total_degree) ** 2).sum()
            return float(internal_fraction - expected_fraction)
        # The diagonal of the contracted graph stores twice the internal edge weight.
        membership = np.eye(count)[labels]
        graph = membership.T @ graph @ membership


def graph_metrics(adjacency):

    n = len(adjacency)
    neighbors = [np.flatnonzero(row) for row in adjacency]
    degrees = adjacency.sum(axis=1)
    edges = int(degrees.sum() // 2)
    possible_pairs = comb(n, 2)
    local_clustering = [
        adjacency[np.ix_(near, near)].sum() / (len(near) * (len(near) - 1))
        if len(near) >= 2 else 0.0
        for near in neighbors
    ]
    reachable_pairs = distance_sum = 0
    unseen = set(range(n))
    component_sizes = []
    for source in range(n):
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for target in neighbors[node]:
                if target not in distances:
                    distances[target] = distances[node] + 1
                    queue.append(target)
        if source in unseen:
            component_sizes.append(len(distances))
            unseen.difference_update(distances)
        # Count each unordered pair once.
        for target, distance in distances.items():
            if target > source:
                reachable_pairs += 1
                distance_sum += distance

    component_sizes.sort(reverse=True)
    return {
        "nodes": n,
        "edges": edges,
        "density": edges / possible_pairs if possible_pairs else 0.0,
        "average_degree": float(degrees.mean()) if n else 0.0,
        "average_local_clustering": sum(local_clustering) / n if n else 0.0,
        "connected_components": len(component_sizes),
        "largest_component_nodes": max(component_sizes, default=0),
        "component_sizes": ", ".join(map(str, component_sizes)),
        "possible_pairs": possible_pairs,
        "reachable_pairs": reachable_pairs,
        "reachable_pairs_pct": percentage(reachable_pairs, possible_pairs),
        "average_shortest_path_reachable_pairs": (
            distance_sum / reachable_pairs if reachable_pairs else None
        ),
    }


def percentage(count, total):
    return 100 * count / total if total else 0.0


def isolate_metrics(group, active, loops):
    isolates = group - active
    loop_only = isolates & loops
    return {
        "isolates": len(isolates),
        "isolation_rate_pct": percentage(len(isolates), len(group)),
        "absent_from_edge_csv": len(isolates - loop_only),
        "self_loop_only": len(loop_only),
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.8f}" if isinstance(value, (float, np.floating)) else value
                for key, value in row.items()
            })


def analyze(folder):
    nodes = read_nodes(folder / "node_data(2).xlsx")
    node_ids = sorted(nodes)
    node_set = set(node_ids)
    groups = defaultdict(set)
    for node_id, org_type in nodes.items():
        groups[org_type].add(node_id)

    full_rows, active_rows = [], []
    active_sets, loop_sets = {}, {}
    for scenario, filename in EDGE_FILES.items():
        graph, loops = read_graph(folder / filename, node_ids)
        active_index = np.flatnonzero(graph.sum(axis=1))
        active = {node_ids[i] for i in active_index}
        active_sets[scenario] = active
        loop_sets[scenario] = loops

        subgraph = graph[np.ix_(active_index, active_index)]
        # Nodes of degree zero contribute zero to modularity, so reuse the active partition.
        modularity = max(louvain_score(subgraph, seed) for seed in range(100)) if active else None
        full_rows.append({
            "scenario": scenario,
            **graph_metrics(graph),
            "modularity": modularity,
        })
        active_rows.append({
            "scenario": scenario,
            **graph_metrics(subgraph),
            "active_nodes_pct_of_all": percentage(len(active), len(nodes)),
            "modularity": modularity,
        })

    isolate_rows = []
    for org_type, group in [*groups.items(), ("Total", node_set)]:
        row = {"organization_type": org_type, "total_nodes": len(group)}
        for scenario in EDGE_FILES:
            metrics = isolate_metrics(group, active_sets[scenario], loop_sets[scenario])
            row.update({f"{scenario}_{name}": value for name, value in metrics.items()})
        row["isolated_in_both"] = len(group - active_sets["mission"] - active_sets["diffusion"])
        isolate_rows.append(row)

    for filename, rows in (
        ("01_full_network_metrics.csv", full_rows),
        ("02_active_network_metrics.csv", active_rows),
        ("03_isolates_by_organization_type.csv", isolate_rows),
    ):
        path = folder / filename
        write_csv(path, rows)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="Input and output folder.")
    args = parser.parse_args()
    try:
        analyze(args.folder)
    except (OSError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
