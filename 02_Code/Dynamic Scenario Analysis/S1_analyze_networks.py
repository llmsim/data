
import argparse
import csv
from collections import defaultdict, deque
from math import comb
from pathlib import Path
from random import Random

import numpy as np
from openpyxl import load_workbook


DEFAULT_FOLDER = Path(r"F:\workspace1")
SCENARIOS = ("mission", "diffusion")
CENTRALITY_NAMES = ("closeness", "betweenness", "eigenvector")
SCORE_DECIMALS = 12


def find_input(folder, stem, extensions):
    names = [
        f"{stem}{suffix}{extension}"
        for suffix in ("(3)", "", "(2)")
        for extension in extensions
    ]
    for name in names:
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{folder}: expected one of {', '.join(names)}.")


def read_nodes(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            return parse_nodes(csv.reader(source), path)

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        return parse_nodes(workbook.active.values, path)
    finally:
        workbook.close()


def parse_nodes(rows, path):
    header = [str(value or "").strip().lower() for value in next(rows, ())]
    if "id" not in header:
        raise ValueError(f"{path.name}: expected an Id column.")
    nodes = {}
    for row_number, row in enumerate(rows, start=2):
        values = dict(zip(header, row))
        if all(value is None or str(value).strip() == "" for value in row):
            continue
        try:
            node_id = int(str(values.get("id")).strip())
        except ValueError as exc:
            raise ValueError(f"{path.name}:{row_number}: Id must be an integer.") from exc
        if node_id in nodes:
            raise ValueError(f"{path.name}:{row_number}: duplicate Id {node_id}.")
        label = str(values.get("label") or "").strip()
        org_type = str(values.get("type") or "").strip()
        nodes[node_id] = (label, org_type)
    if not nodes:
        raise ValueError(f"{path.name}: no nodes found.")
    return nodes


def read_graph(path, node_ids, excluded_ids):
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    adjacency = np.zeros((len(index), len(index)), dtype=float)
    excluded_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"Source", "Target"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: expected Source and Target columns.")
        for row in reader:
            a, b = int(row["Source"]), int(row["Target"])
            if a in excluded_ids or b in excluded_ids:
                excluded_rows += 1
                continue
            if a not in index or b not in index:
                raise ValueError(f"{path.name}:{reader.line_num}: unknown endpoint ({a}, {b}).")
            if row.get("Type", "Undirected").strip().lower() != "undirected":
                raise ValueError(f"{path.name}:{reader.line_num}: expected an undirected tie.")
            if a != b:
                # Repeated rows and reversed copies refer to the same unweighted tie.
                adjacency[index[a], index[b]] = adjacency[index[b], index[a]] = 1
    print(f"{path.name}: removed {excluded_rows} rows involving the exiting institution")
    return adjacency


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


def shortest_path_metrics(adjacency):
    n = len(adjacency)
    neighbors = [np.flatnonzero(row).tolist() for row in adjacency]
    closeness = np.zeros(n)
    betweenness = np.zeros(n)
    ordered_pairs = distance_sum = 0
    unseen = set(range(n))
    component_sizes = []
    for source in range(n):
        distances = [-1] * n
        distances[source] = 0
        path_counts = [0] * n
        path_counts[source] = 1
        predecessors = [[] for _ in range(n)]
        order = []
        queue = deque([source])
        while queue:
            node = queue.popleft()
            order.append(node)
            next_distance = distances[node] + 1
            for target in neighbors[node]:
                if distances[target] == -1:
                    distances[target] = next_distance
                    queue.append(target)
                if distances[target] == next_distance:
                    path_counts[target] += path_counts[node]
                    predecessors[target].append(node)

        reached = len(order) - 1
        total_distance = sum(distances[node] for node in order)
        ordered_pairs += reached
        distance_sum += total_distance
        if reached:
            closeness[source] = reached / total_distance * reached / (n - 1)
        if source in unseen:
            component_sizes.append(len(order))
            unseen.difference_update(order)

        dependency = [0.0] * n
        for node in reversed(order):
            for parent in predecessors[node]:
                dependency[parent] += path_counts[parent] / path_counts[node] * (1 + dependency[node])
            if node != source:
                betweenness[node] += dependency[node]

    # Each undirected pair was visited in both directions, as was the denominator.
    if n > 2:
        betweenness /= (n - 1) * (n - 2)
    component_sizes.sort(reverse=True)
    average_path = distance_sum / ordered_pairs if ordered_pairs else None
    return ordered_pairs // 2, average_path, component_sizes, closeness, betweenness


def eigenvector_centrality(adjacency):
    if not np.any(adjacency):
        return np.zeros(len(adjacency))
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    dominant = eigenvectors[:, np.isclose(eigenvalues, eigenvalues[-1], rtol=1e-12, atol=1e-12)]
    # Project the all-ones vector onto the leading eigenspace, including spectral ties.
    scores = dominant @ dominant.sum(axis=0)
    scores = np.maximum(scores, 0.0)
    return scores / np.linalg.norm(scores)


def graph_metrics(adjacency, reachable_pairs, average_path, component_sizes):
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
        "average_shortest_path_reachable_pairs": average_path,
    }


def percentage(count, total):
    return 100 * count / total if total else 0.0


def ranked_nodes(scenario, metric, values, node_ids, nodes, degrees):
    scores = [round(float(value), SCORE_DECIMALS) for value in values]
    order = sorted(range(len(node_ids)), key=lambda i: (-scores[i], node_ids[i]))
    previous_score = None
    for position, i in enumerate(order, start=1):
        score = scores[i]
        if score != previous_score:
            rank = position
            previous_score = score
        node_id = node_ids[i]
        label, org_type = nodes[node_id]
        yield {
            "scenario": scenario,
            "metric": metric,
            "rank": rank,
            "node_id": node_id,
            "label": label,
            "organization_type": org_type,
            "degree": int(degrees[i]),
            "score": score,
        }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.{SCORE_DECIMALS}f}" if isinstance(value, (float, np.floating)) else value
                for key, value in row.items()
            })


def analyze(folder, node_file=None):
    if node_file is not None:
        node_path = node_file if node_file.is_absolute() else folder / node_file
    else:
        attached_path = folder / "node_data(3)(1).xlsx"
        node_path = attached_path if attached_path.is_file() else find_input(folder, "node_data", (".xlsx", ".csv"))
    nodes = read_nodes(node_path)
    if len(nodes) != 146:
        raise ValueError(f"{node_path.name}: this exit experiment requires 146 original nodes, found {len(nodes)}.")
    print(f"Node file: {node_path} ({len(nodes)} unique nodes)")
    exclusions = {"mission": "Northrop Grumman", "diffusion": "Sierra Space"}
    network_rows = []
    centrality_rows = []
    for scenario in SCENARIOS:
        node_ids = sorted(nodes)
        excluded_name = exclusions[scenario]
        if excluded_name:
            matches = [
                node_id for node_id, (label, _) in nodes.items()
                if label.casefold() == excluded_name.strip().casefold()
            ]
            if len(matches) != 1:
                raise ValueError(f"{scenario}: expected one institution named {excluded_name!r}, found {len(matches)}.")
            node_ids.remove(matches[0])
            print(f"{scenario}: excluding {excluded_name} (ID {matches[0]}), using {len(node_ids)} nodes")
        edge_path = find_input(folder, f"edge_data_{scenario}", (".csv",))
        print(f"Reading {node_path.name} and {edge_path.name} ({scenario})")
        graph = read_graph(edge_path, node_ids, set(nodes) - set(node_ids))
        degrees = graph.sum(axis=1)
        active_index = np.flatnonzero(degrees)
        subgraph = graph[np.ix_(active_index, active_index)]
        # Isolates change neither reachable-pair distances nor the modularity objective.
        modularity = max(louvain_score(subgraph, seed) for seed in range(100)) if len(subgraph) else None
        reachable_pairs, average_path, sizes, closeness, betweenness = shortest_path_metrics(graph)
        scopes = (("full", graph, sizes), ("active", subgraph, [size for size in sizes if size > 1]))
        for scope, adjacency, component_sizes in scopes:
            network_rows.append({
                "scenario": scenario,
                "scope": scope,
                **graph_metrics(adjacency, reachable_pairs, average_path, component_sizes),
                "modularity": modularity,
            })

        centralities = (closeness, betweenness, eigenvector_centrality(graph))
        for metric, values in zip(CENTRALITY_NAMES, centralities):
            centrality_rows.extend(ranked_nodes(scenario, metric, values, node_ids, nodes, degrees))

    for filename, rows in (
        ("sensitivity_network_metrics.csv", network_rows),
        ("sensitivity_node_rankings.csv", centrality_rows),
    ):
        path = folder / filename
        write_csv(path, rows)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="Input and output folder.")
    parser.add_argument("--nodes", type=Path, help="Node filename inside --folder, or an absolute path.")
    args = parser.parse_args()
    try:
        analyze(args.folder, args.nodes)
    except (OSError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
