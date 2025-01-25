# from __future__ import annotations
from typing import Dict, Set, Optional

import sys
sys.dont_write_bytecode = True

class Vertex:
    def __init__(self, name: str, coords: str):
        self.name = name
        self.coords = coords  # Formato: "lon lat"
    
    def __repr__(self) -> str:
        return f"Vertex({self.name})"
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vertex) and self.name == other.name

class Edge:
    def __init__(self, origin: Vertex, destination: Vertex, time_minutes: float, geometry: list[tuple[float, float]]):
        self.origin = origin
        self.destination = destination
        self.time_minutes = time_minutes
        self.geometry = geometry
    
    def __repr__(self) -> str:
        return f"Edge({self.origin.name} ↔ {self.destination.name}: {self.time_minutes} min)"
    
    @property
    def key(self) -> frozenset:
        return frozenset({self.origin.name, self.destination.name})

class Graph:
    def __init__(self):
        self.vertices: Dict[str, Vertex] = {}  # Acceso rápido por nombre
        self.edges: Dict[frozenset, Edge] = {}  # Clave única por par de nodos
        
    def add_vertex(self, vertex: Vertex) -> None:
        if vertex.name in self.vertices:
            raise ValueError(f"Vertex {vertex.name} already exists")
        self.vertices[vertex.name] = vertex
    
    def add_edge(self, edge: Edge) -> None:
        if edge.origin.name not in self.vertices or edge.destination.name not in self.vertices:
            raise ValueError("Vertices must be added to graph before edges")
        
        # Usamos frozenset para que el orden no importe
        self.edges[edge.key] = edge
    
    def get_time(self, origin_name: str, destination_name: str) -> Optional[float]:
        key = frozenset({origin_name, destination_name})
        edge = self.edges.get(key)
        return edge.time_minutes if edge else None
    
    def get_all_vertices(self) -> Set[Vertex]:
        return set(self.vertices.values())
    
    def get_all_edges(self) -> Set[Edge]:
        return set(self.edges.values())
    
    def __repr__(self) -> str:
        return f"Graph({len(self.vertices)} vertices, {len(self.edges)} edges)"

# Ejemplo de uso
if __name__ == "__main__":
    # Crear grafo
    g = Graph()
    
    # Añadir vértices
    v1 = Vertex("A", "-74.072092 4.710989")
    v2 = Vertex("B", "-75.573487 6.244338")
    g.add_vertex(v1)
    g.add_vertex(v2)
    
    # Añadir arcos (tiempos de ejemplo)
    g.add_edge(Edge(v1, v2, 45.5))
    
    # Consultar tiempos
    print(f"Tiempo entre A y B: {g.get_time('A', 'B')} min")  # 45.5
    print(f"Tiempo entre B y A: {g.get_time('B', 'A')} min")  # 45.5 (mismo resultado)
    
    # Intentar obtener tiempo de nodo no existente
    print(f"Tiempo entre A y C: {g.get_time('A', 'C')}")  # None