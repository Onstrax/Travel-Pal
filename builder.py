import time
import polyline
import requests
import itertools
from typing import Dict
from parser import parse_kml_to_dict
from graphs import Vertex, Graph, Edge

import sys
sys.dont_write_bytecode = True

def build_complete_walking_graph(kml_dict: Dict[str, str], osrm_base_url: str = "http://localhost:5000") -> Graph:
# def build_complete_walking_graph(kml_dict: Dict[str, str], osrm_base_url: str = "http://router.project-osrm.org") -> Graph:
    graph = Graph()
    
    # 1. Añadir todos los vértices al grafo
    for name, coords in kml_dict.items():
        graph.add_vertex(Vertex(name, coords))
    
    # 2. Generar todas las combinaciones únicas de pares
    all_vertices = list(graph.vertices.values())
    total_pairs = len(all_vertices) * (len(all_vertices) - 1) // 2
    print(f"Calculando {total_pairs} pares de rutas...")
    
    # 3. Consultar OSRM para cada par
    for i, (v1, v2) in enumerate(itertools.combinations(all_vertices, 2), 1):
        # Formatear coordenadas para OSRM (lon,lat)
        coords_str = f"{v1.coords.replace(' ', ',')};{v2.coords.replace(' ', ',')}"
        
        try:
            # Hacer la solicitud al servidor local
            response = requests.get(
                f"{osrm_base_url}/route/v1/foot/{coords_str}",
                params={"overview": "full", "steps": "false"}  # ¡Cambiado a 'full'!
            ).json()
            
            if response["code"] == "Ok":
                duration = round(response["routes"][0]["duration"] / 60, 1)
                geometry = polyline.decode(response["routes"][0]["geometry"])  # Decodificar polilínea
                edge = Edge(v1, v2, duration, geometry)
                graph.add_edge(edge)
            else:
                print(f"Error entre {v1.name}-{v2.name}: {response.get('message', 'Sin ruta')}")
                continue
            
        except Exception as e:
            print(f"Error de conexión en {v1.name}-{v2.name}: {str(e)}")
            continue
        
        # Progress tracking
        if i % 10 == 0:
            print(f"Procesados {i}/{total_pairs} pares ({i/total_pairs:.1%})")
        
        # Pequeña pausa para no saturar el servidor
        time.sleep(0.1)
    
    print("Grafo completo generado!")
    return graph

# Uso integrado
if __name__ == "__main__":
    # 1. Parsear KML
    poi_dict = parse_kml_to_dict("POIs3.kml")
    
    # 2. Construir grafo
    walking_graph = build_complete_walking_graph(poi_dict)
    
    # 3. Ejemplo de consulta
    # print("\nEjemplo de tiempos:")
    # sample_nodes = list(walking_graph.vertices.keys())[:3]
    # for a, b in itertools.combinations(sample_nodes, 2):
    #     print(f"{a} -> {b}: {walking_graph.get_time(a, b)} min")
        
    # 'Pontificia Universidad Javeriana', 'Parque Nacional', 
    # 'Universidad de Bogotá Jorge Tadeo Lozano', 'Museo del Oro', 'Museo Botero', 
    # 'Parque de los Periodistas Gabriel García Márquez', 'Universidad de los Andes', 
    # 'Monserrate'
    
    g = "Pontificia Universidad Javeriana"
    f = "Parque Nacional"
    e = "Universidad de Bogotá Jorge Tadeo Lozano"
    c = "Universidad de los Andes"
    d = "Parque de los Periodistas Gabriel García Márquez"
    b = "Museo del Oro"
    a = "Museo Botero"
    h = "Parque de los Hippies"
    i = "Monserrate"
    
    times = 0
    print(f"{a} -> {b}: {walking_graph.get_time(a, b)} min")
    times += walking_graph.get_time(a, b)
    print(f"{b} -> {c}: {walking_graph.get_time(b, c)} min")
    times += walking_graph.get_time(b, c)
    print(f"{c} -> {d}: {walking_graph.get_time(c, d)} min")
    times += walking_graph.get_time(c, d)
    print(f"{d} -> {e}: {walking_graph.get_time(d, e)} min")
    times += walking_graph.get_time(d, e)
    print(f"{e} -> {f}: {walking_graph.get_time(e, f)} min")
    times += walking_graph.get_time(e, f)
    print(f"{f} -> {g}: {walking_graph.get_time(f, g)} min")
    times += walking_graph.get_time(f, g)
    print(f"{g} -> {h}: {walking_graph.get_time(g, h)} min")
    times += walking_graph.get_time(g, h)
    print(f"{h} -> {i}: {walking_graph.get_time(h, i)} min")
    times += walking_graph.get_time(h, i)
    print(f'Total = {times}')