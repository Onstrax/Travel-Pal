from typing import List, Set, Dict, Tuple, Optional
from graphs import Graph, Edge, Vertex
from itertools import combinations
import itertools
import heapq
import math
import sys

import sys
sys.dont_write_bytecode = True

def find_clusters(graph: Graph, max_walk: int = 15) -> List[List[str]]:
    clusters = []
    visited: Set[str] = set()
    
    # Construir lista de adyacencia solo para arcos <=15 minutos
    adjacency: Dict[str, List[str]] = {}
    for edge in graph.get_all_edges():
        if edge.time_minutes <= max_walk:
            u = edge.origin.name
            v = edge.destination.name
            if u not in adjacency:
                adjacency[u] = []
            adjacency[u].append(v)
            if v not in adjacency:
                adjacency[v] = []
            adjacency[v].append(u)
    
    # Encontrar componentes conectados usando BFS
    for vertex_name in graph.vertices:
        if vertex_name not in visited:
            cluster = []
            queue = [vertex_name]
            visited.add(vertex_name)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                # Agregar vecinos no visitados
                if current in adjacency:
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            clusters.append(cluster)
    
    return clusters

def find_optimal_path(
    graph: Graph,
    cluster: List[str],
    start_point: Optional[str] = None
) -> List[str]:
    """
    Encuentra la ruta óptima que visita todos los nodos del cúmulo exactamente una vez.
    Usa Held-Karp para cúmulos <=20 nodos, Christofides para mayores.
    """
    # Validación inicial
    if not cluster:
        return []
    for node in cluster:
        if node not in graph.vertices:
            raise ValueError(f"El nodo {node} no existe en el grafo")

    # Crear matriz de costos
    size = len(cluster)
    cost_matrix = _build_cost_matrix(graph, cluster)

    if start_point is not None:
        start_index = cluster.index(start_point)
    else:
        start_index = None

    return _re_run_to_minimize(graph, cost_matrix, cluster, start_index)
#     return _held_karp_path(cost_matrix, cluster, start_index)
    # Seleccionar algoritmo según tamaño
#     if size <= 20:
#         return _held_karp_path(cost_matrix, cluster, start_index)
#     else:
#         return _christofides_path(cost_matrix, cluster, start_index)

def _optimal_path_from(graph: Graph, cluster: List[str], start: str):
    # Validación de parámetros
    if start not in cluster:
        raise ValueError("El punto de inicio no está en el cúmulo")
    
    unique_cluster = list(dict.fromkeys(cluster))  # Eliminar duplicados preservando orden
    
    # Verificar que todos los nodos existen en el grafo
    for node in unique_cluster:
        if node not in graph.vertices:
            raise ValueError(f"El nodo '{node}' no existe en el grafo")
    
    # Casos triviales
    if not unique_cluster:
        return []
    if len(unique_cluster) == 1:
        return [start]
    
    # Inicialización
    path = [start]
    remaining = [node for node in unique_cluster if node != start]
    current_node = start
    
    # Construir ruta usando el vecino más cercano
    while remaining:
        nearest = None
        min_time = float('inf')
        
        for candidate in remaining:
            time = graph.get_time(current_node, candidate)
            if time < min_time:
                min_time = time
                nearest = candidate
        
        if nearest is None:
            break  # No debería ocurrir en grafo completo
        
        path.append(nearest)
        remaining.remove(nearest)
        current_node = nearest
    
    return path

def _re_run_to_minimize(graph: Graph, cost_matrix: Dict[Tuple[int, int], float], cluster: List[str], start_index: int | None) -> List[str]:
     min_route: List[str] = []
     min_time: int = math.inf
     if len(cluster) <= 10:
          # print("Running Held-Karp", len(cluster))
          return _optimal_held_karp(graph, cluster, cluster[start_index])
     # print("Running Christofides", len(cluster))
     for _ in range(50):
          route1 = _christofides_path(cost_matrix, cluster, start_index)
          time1 = _get_time_for_cluster(route1, graph)
          if time1 < min_time:
               min_time = time1
               min_route = route1
          route2 = _held_karp_path(cost_matrix, cluster, start_index) if len(cluster) <= 20 else []
          time2 = _get_time_for_cluster(route2, graph)
          if time2 < min_time:
               min_time = time2
               min_route = route2
     return min_route

def _optimal_held_karp(graph: Graph, cluster: List[str], start: str) -> List[str]:
    """
    Encuentra la ruta más rápida que visita todos los nodos del cúmulo exactamente una vez,
    comenzando en el nodo especificado, utilizando el algoritmo de Held-Karp.
    """
    
    if start not in cluster:
        raise ValueError("El nodo de inicio no está en el cúmulo")
    
    # Reordenar el cúmulo para que el inicio sea el primer elemento
    nodes = [start] + [node for node in cluster if node != start]
    n = len(nodes)
    
    # Crear matriz de costos
    cost: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                cost[i][j] = graph.get_time(nodes[i], nodes[j]) or float('inf')
    
    # Algoritmo de Held-Karp modificado para ruta abierta
    dp: Dict = {}
    start_mask = 1 << 0
    dp[(start_mask, 0)] = 0.0
    
    # Generar máscaras en orden de tamaño creciente
    masks = [mask for mask in range(1 << n) if (mask & start_mask)]
    masks.sort(key=lambda x: bin(x).count('1'))
    
    for mask in masks:
        size = bin(mask).count('1')
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            # El nodo inicial solo puede estar en máscaras de tamaño 1
            if j == 0 and size > 1:
                continue
            # Buscar todos los posibles nodos anteriores (i)
            for i in range(n):
                if i == j or not (mask & (1 << i)):
                    continue
                prev_mask = mask ^ (1 << j)
                if prev_mask not in [m for m in masks if bin(m).count('1') == size - 1]:
                    continue
                if (prev_mask, i) in dp:
                    new_cost = dp[(prev_mask, i)] + cost[i][j]
                    if (mask, j) not in dp or new_cost < dp[(mask, j)]:
                        dp[(mask, j)] = new_cost
    
    # Encontrar el mínimo costo en la máscara completa
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    best_j = -1
    for j in range(1, n):  # No puede terminar en el nodo inicial
        if (full_mask, j) in dp and dp[(full_mask, j)] < min_cost:
            min_cost = dp[(full_mask, j)]
            best_j = j
    
    if best_j == -1:
        return []  # No se encontró ruta válida
    
    # Reconstruir la ruta
    path = []
    current_mask = full_mask
    current_j = best_j
    path.append(current_j)
    
    while current_mask != start_mask:
        prev_mask = current_mask ^ (1 << current_j)
        for i in range(n):
            if i == current_j:
                continue
            if (prev_mask, i) in dp and dp[(prev_mask, i)] + cost[i][current_j] == dp[(current_mask, current_j)]:
                current_mask = prev_mask
                current_j = i
                path.append(i)
                break
    
    path.reverse()
    return [nodes[i] for i in path]

def _get_time_for_cluster(cluster: List[str], graph: Graph) -> int:
     times = 0
     for i in range(len(cluster)-1):
          times += graph.get_time(cluster[i], cluster[i+1])
     return times

def _build_cost_matrix(graph: Graph, cluster: List[str]) -> Dict[Tuple[int, int], float]:
    """Construye una matriz de costos entre todos los pares del cúmulo."""
    indices = {node: i for i, node in enumerate(cluster)}
    matrix = {}
    for i, u in enumerate(cluster):
        for j, v in enumerate(cluster):
            if i != j:
                matrix[(i, j)] = graph.get_time(u, v) or math.inf
    return matrix

def _held_karp_path(
    cost_matrix: Dict[Tuple[int, int], float],
    cluster: List[str],
    start_index: Optional[int]
) -> List[str]:
    """Versión corregida para TSP abierto (sin retorno al origen)."""
    n = len(cluster)
    memo = {}

    def dp(mask: int, u: int) -> Tuple[float, Optional[int]]:
        if (mask, u) in memo:
            return memo[(mask, u)]

        # Todos los nodos visitados: costo 0 (no se suma regreso)
        if mask == (1 << n) - 1:
            return 0, None

        min_cost = math.inf
        next_node = None

        for v in range(n):
            if not (mask & (1 << v)):
                new_mask = mask | (1 << v)
                cost, _ = dp(new_mask, v)
                # Solo sumar el costo del arco actual
                total_cost = cost_matrix.get((u, v), math.inf) + cost

                if total_cost < min_cost:
                    min_cost = total_cost
                    next_node = v

        memo[(mask, u)] = (min_cost, next_node)
        return min_cost, next_node

    # Inicialización
    if start_index is not None:
        initial_mask = 1 << start_index
        min_cost, next_node = dp(initial_mask, start_index)
        path = [start_index]
    else:
        min_cost = math.inf
        best_start = 0
        for s in range(n):
            initial_mask = 1 << s
            cost, _ = dp(initial_mask, s)
            if cost < min_cost:
                min_cost = cost
                best_start = s
        initial_mask = 1 << best_start
        _, next_node = dp(initial_mask, best_start)
        path = [best_start]

    # Reconstrucción del camino CORREGIDA
    mask = initial_mask
    current = path[0]
    while next_node is not None:
        path.append(next_node)
        mask |= 1 << next_node
        # Acceder directamente a la entrada completa de memo
        _, next_node = memo.get((mask, next_node), (math.inf, None))  # Corregido

    assert len(path) == n, "El camino no incluye todos los nodos"
    return [cluster[i] for i in path]

def _christofides_path(
    cost_matrix: Dict[Tuple[int, int], float],
    cluster: List[str],
    start_index: Optional[int]
) -> List[str]:
    """Implementación completa de Christofides para TSP métrico abierto."""
    n = len(cluster)
    if n == 0:
        return []
    
    # 1. Construir el Grafo Completo
    graph = {i: {j: cost_matrix[(i,j)] for j in range(n) if i != j} for i in range(n)}
    
    # 2. Encontrar el MST
    mst = _prim_mst(graph, n)
    
    # 3. Encontrar nodos de grado impar en el MST
    odd_degree_nodes = _find_odd_degree_nodes(mst, n)
    
    # 4. Encontrar emparejamiento perfecto mínimo
    min_matching = _min_weight_matching(graph, odd_degree_nodes)
    
    # 5. Combinar MST y emparejamiento
    multi_graph = _combine_graphs(mst, min_matching, n)
    
    # 6. Encontrar circuito euleriano
    eulerian_circuit = _hierholzer(multi_graph)
    
    # 7. Convertir a ruta hamiltoniana
    path = _shortcut_eulerian(eulerian_circuit)
    
    # 8. Ajustar punto de inicio si es necesario
    if start_index is not None:
        path = _adjust_start(path, cluster[start_index], cluster)
    
    return [cluster[i] for i in path]

# ---------------------------
# Funciones auxiliares
# ---------------------------

def _prim_mst(graph: Dict[int, Dict[int, float]], n: int) -> List[Set[int]]:
    """Construye el MST usando el algoritmo de Prim."""
    mst = [set() for _ in range(n)]
    visited = [False] * n
    heap = []
    
    start_node = 0
    heapq.heappush(heap, (0, -1, start_node))
    
    while heap:
        weight, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        if u != -1:
            mst[u].add(v)
            mst[v].add(u)
        
        for neighbor in graph[v]:
            if not visited[neighbor]:
                heapq.heappush(heap, (graph[v][neighbor], v, neighbor))
    
    return mst

def _find_odd_degree_nodes(mst: List[Set[int]], n: int) -> List[int]:
    """Identifica nodos con grado impar en el MST."""
    return [i for i in range(n) if len(mst[i]) % 2 != 0]

def _min_weight_matching(graph: Dict[int, Dict[int, float]], nodes: List[int]) -> Set[Tuple[int, int]]:
    """Aproximación greedy de emparejamiento perfecto mínimo."""
    matching = set()
    nodes = sorted(nodes)
    used = [False] * len(nodes)
    
    for i in range(len(nodes)):
        if not used[i]:
            min_cost = math.inf
            pair = -1
            for j in range(i+1, len(nodes)):
                if not used[j]:
                    u, v = nodes[i], nodes[j]
                    cost = graph[u][v]
                    if cost < min_cost:
                        min_cost = cost
                        pair = j
            if pair != -1:
                matching.add((nodes[i], nodes[pair]))
                used[i] = used[pair] = True
    
    return matching

def _combine_graphs(mst: List[Set[int]], matching: Set[Tuple[int, int]], n: int) -> List[Set[int]]:
    """Combina el MST con las aristas del emparejamiento."""
    combined = [set(s) for s in mst]
    for u, v in matching:
        combined[u].add(v)
        combined[v].add(u)
    return combined

def _hierholzer(graph: List[Set[int]]) -> List[int]:
    """Algoritmo de Hierholzer para encontrar circuito euleriano."""
    circuit = []
    stack = [0]
    while stack:
        current = stack[-1]
        if graph[current]:
            next_node = graph[current].pop()
            graph[next_node].remove(current)
            stack.append(next_node)
        else:
            circuit.append(stack.pop())
    return circuit[::-1]

def _shortcut_eulerian(circuit: List[int]) -> List[int]:
    """Elimina nodos repetidos para obtener ruta hamiltoniana."""
    visited = set()
    path = []
    for node in circuit:
        if node not in visited:
            visited.add(node)
            path.append(node)
    return path

def _adjust_start(path: List[int], start_node: str, cluster: List[str]) -> List[int]:
    """Reordena la ruta para que comience en el nodo especificado."""
    start_index = cluster.index(start_node)
    if path[0] == start_index:
        return path
    try:
        idx = path.index(start_index)
    except ValueError:
        return path
    return path[idx:] + path[:idx]



# Ejemplo de uso
if __name__ == "__main__":
    # Crear grafo de prueba
    g = Graph()
    nodes = ["A", "B", "C", "D"]
    for n in nodes:
        g.add_vertex(Vertex(n, "0 0"))
    
    # Añadir arcos de ejemplo
    g.add_edge(Edge(g.vertices["A"], g.vertices["B"], 10))
    g.add_edge(Edge(g.vertices["B"], g.vertices["C"], 12))
    g.add_edge(Edge(g.vertices["C"], g.vertices["D"], 8))
    g.add_edge(Edge(g.vertices["A"], g.vertices["D"], 20))
    
    cluster = ["A", "B", "C", "D"]
    print("Ruta óptima general:", find_optimal_path(g, cluster))
    print("Ruta comenzando en B:", find_optimal_path(g, cluster, "B"))