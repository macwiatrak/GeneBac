from math import sqrt
from typing import Any, Optional

import numpy as np
from torch import Tensor


def visualize_graph_via_networkx(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
    labels: Optional[list[str]] = None,
    node_size_list: Optional[np.ndarray] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    node_size = 800

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    pos = nx.spring_layout(g)
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            "",
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                alpha=data["alpha"],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )
    if node_size_list is not None:
        node_size = [node_size_list[idx] * 800 for idx in pos.keys()]

    nodes = nx.draw_networkx_nodes(
        g, pos, node_size=node_size, node_color="blue", margins=0.1, alpha=0.4
    )
    nodes.set_edgecolor("black")

    if labels is not None:
        labels = {p: labels[p] for p in pos}

    nx.draw_networkx_labels(g, pos, font_size=10, labels=labels)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()
