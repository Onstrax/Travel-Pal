from builder import build_complete_walking_graph
from joiner import optimize_cluster_order
from parser import parse_kml_to_dict
from typing import List
import map_generator as MG
import clusters as CL
import time
import math

import sys
sys.dont_write_bytecode = True

def get_times_for_cluster(cluster: List[str], print_bool = True) -> int:
     times = 0
     for i in range(len(cluster)-1):
          if print_bool: print(f"{cluster[i]} -> {cluster[i+1]}: {graph.get_time(cluster[i], cluster[i+1])} min")
          times += graph.get_time(cluster[i], cluster[i+1])
     return times

def get_times_for_route(route: List[List[str]], print_bool = True) -> int:
     total_final = 0
     j = 0
     while j < len(route):
          i = 0
          cumulo = route[j]
          while i<len(cumulo)-1:
               origin = cumulo[i]
               destiny = cumulo[i+1]
               travel_time = graph.get_time(origin, destiny)
               if print_bool: print(f'{origin} --> {destiny} : {travel_time} min')
               total_final += travel_time
               i+=1
          origin = cumulo[i]
          destiny = route[j+1][0] if j < len(route)-1 else "NO MAS CUMULOS"
          travel_time = graph.get_time(origin, destiny)
          if print_bool: print(f'TRASBORDO: {origin} --> {destiny} : {travel_time} min') if destiny != "NO MAS CUMULOS" else print("FIN")
          if travel_time: total_final += travel_time
          j+=1
     return total_final

def path_of(cluster: List[str]) -> str:
     path = ""
     for i in range(len(cluster)):
          path += cluster[i]
          if i != len(cluster)-1:
               path += " -> "
     return path

def main():
     pass

if __name__ == "__main__":
     start = time.time()
     start_point = "Marienplatz"
     start_point = None
     data_dict = parse_kml_to_dict("POIs4.kml")
     graph = build_complete_walking_graph(data_dict)
     end = time.time()
     clusters = CL.find_clusters(graph)
     print(clusters)
     optimal_clusters = []
     starter_cluster = None
     for i, sublist in enumerate(clusters):
        if start_point in sublist:
            starter_cluster = clusters.pop(i)
     if starter_cluster:
          print(f"\nCLUSTER ORIGINAL: {path_of(starter_cluster)}")
          optimized_starter = CL.find_optimal_path(graph, starter_cluster, start_point)
          if len(optimized_starter) == 0:
                    optimized_starter = starter_cluster
          optimal_clusters.append(optimized_starter)
          print(f'RUTA HALLADA: {path_of(optimized_starter)}, {get_times_for_cluster(optimized_starter, False)}')
     for cluster in clusters:
          min_time = math.inf
          best_route = []
          print(f"\nCLUSTER ORIGINAL: {path_of(cluster)}")
          for POI in cluster:
               POI_path = CL.find_optimal_path(graph, cluster, POI)
               if len(POI_path) == 0:
                    POI_path = cluster
               POI_time = get_times_for_cluster(POI_path, False)
               if POI_time < min_time:
                    min_time = POI_time
                    best_route = POI_path
          print(f'RUTA HALLADA: {path_of(best_route)}, {min_time}')
          optimal_clusters.append(best_route)
     print(f"\n\nRUTAS OPTIMIZADAS: {optimal_clusters}")
     end2 = time.time()
     
     # Opciones: 'brute', 'genetic', 'nearest'
     # method = 'genetic'
     answer = []
     best_time = math.inf
     if len(optimal_clusters) <= 10: 
          answer = optimize_cluster_order(optimal_clusters, graph, start_point, "brute")
          best_time = get_times_for_route(answer, False)
     nearest_route = optimize_cluster_order(optimal_clusters, graph, start_point, "nearest")
     genetic_route = optimize_cluster_order(optimal_clusters, graph, start_point, "genetic")
     nearest_time = get_times_for_route(nearest_route, False)
     genetic_time = get_times_for_route(genetic_route, False)
     if nearest_time < genetic_time and nearest_time < best_time:
          print("\n\nBy nearest:")
          answer = nearest_route
          best_time = nearest_time
     elif genetic_time < nearest_time and genetic_time < best_time:
          print("\n\nBy genetic:")
          answer = genetic_route
          best_time = genetic_time
     else:
          print("\n\nBy brute:")
     print(f'RUTA FINAL: {answer}')
     
     print(f'\n\nTIEMPO TOTAL: {best_time} min')
     
     MG.create_kml_real_route(answer, graph, "ruta_optima_kml_real.kml")
     # MG.create_html_real_route(answer, graph, "ruta_optima_html_real.html")
     
     end3 = time.time()
     print(f'\nExecution time \n--> Build: {end-start}, \n--> Clusters: {end2-end}, \n--> Joining: {end3-end2}, \n--> Total: {end3-start}')
     
     import shutil
     shutil.rmtree("__pycache__", ignore_errors=True)