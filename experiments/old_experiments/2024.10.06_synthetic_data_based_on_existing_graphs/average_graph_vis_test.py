import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_random_graph(n_nodes, n_edges):
    """Сгенерировать случайный граф."""
    G = nx.gnm_random_graph(n_nodes, n_edges)
    return G


def combine_graphs(graphs):
    """Объединить несколько графов и подсчитать толщину рёбер."""
    combined_graph = nx.Graph()
    edge_thickness = defaultdict(int)

    for g in graphs:
        for u, v in g.edges():
            # Увеличиваем счетчик для ребра
            edge_thickness[(u, v)] += 1

    # Добавляем рёбра в комбинированный граф с толщиной
    for edge, count in edge_thickness.items():
        combined_graph.add_edge(edge[0], edge[1], thickness=count)

    return combined_graph


def plot_graph(G):
    """Визуализировать граф с толщиной рёбер."""
    pos = nx.spring_layout(G)
    thickness = nx.get_edge_attributes(G, "thickness")

    # Извлекаем толщину для рёбер
    widths = [
        thickness[edge] * 2 for edge in G.edges()
    ]  # Увеличив масштаб для лучшей видимости

    nx.draw(G, pos, with_labels=True, node_color="lightblue", width=widths)
    plt.title("Объединенный граф с толщиной рёбер")
    plt.show()


# Параметры
n_nodes = 10
n_graphs = 3
n_edges = 15

# Генерация случайных графов
graphs = [generate_random_graph(n_nodes, n_edges) for _ in range(n_graphs)]

# Объединение графов
combined_graph = combine_graphs(graphs)

# Визуализация объединенного графа
plot_graph(combined_graph)
