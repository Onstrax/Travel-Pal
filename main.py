from builder import build_complete_walking_graph
from joiner import optimize_cluster_order
from parser import parse_kml_to_dict
from typing import List
from graphs import Graph
import map_generator as MG
import clusters as CL
import shutil
import time
import math

import sys
sys.dont_write_bytecode = True

def get_times_for_cluster(cluster: List[str], graph: Graph, print_bool = True) -> int:
     times = 0
     for i in range(len(cluster)-1):
          if print_bool: print(f"{cluster[i]} -> {cluster[i+1]}: {graph.get_time(cluster[i], cluster[i+1])} min")
          times += graph.get_time(cluster[i], cluster[i+1])
     return times

def get_times_for_route(route: List[List[str]], graph: Graph, print_bool = True) -> int:
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

def main(start_point: str, read_from: str, write_to: str, max_walk: int) -> None:
     start = time.time()
     data_dict = parse_kml_to_dict(read_from)
     graph = build_complete_walking_graph(data_dict)
     end = time.time()
     clusters = CL.find_clusters(graph, max_walk) #15 min default
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
          print(f'RUTA HALLADA: {path_of(optimized_starter)}, {get_times_for_cluster(optimized_starter, graph, False)}')
     for cluster in clusters:
          min_time = math.inf
          best_route = []
          print(f"\nCLUSTER ORIGINAL: {path_of(cluster)}")
          for POI in cluster:
               POI_path = CL.find_optimal_path(graph, cluster, POI)
               if len(POI_path) == 0:
                    POI_path = cluster
               POI_time = get_times_for_cluster(POI_path, graph, False)
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
     # if len(optimal_clusters) <= 10: 
     #      answer = optimize_cluster_order(optimal_clusters, graph, start_point, "brute")
     #      best_time = get_times_for_route(answer, graph, False)
     #      print(f"\nBy brute: {best_time}")
     nearest_route = optimize_cluster_order(optimal_clusters, graph, start_point, "nearest")
     nearest_time = get_times_for_route(nearest_route, graph, False)
     print(f"\nBy nearest: {nearest_time}")
     genetic_route = optimize_cluster_order(optimal_clusters, graph, start_point, "genetic")
     genetic_time = get_times_for_route(genetic_route, graph, False)
     print(f"\nBy genetic: {genetic_time}")
     genetic_time = math.inf
     if nearest_time <= genetic_time and nearest_time <= best_time:
          print("\n\nBy nearest:")
          answer = nearest_route
          best_time = nearest_time
     # if genetic_time <= nearest_time and genetic_time <= best_time:
     #      print("\n\nBy genetic:")
     #      answer = genetic_route
     #      best_time = genetic_time
     # else:
     #      print("\n\nBy brute:")
     print(f'RUTA FINAL: {answer}')
     
     print(f'\n\nTIEMPO TOTAL: {best_time} min')
     end3 = time.time()
     
     MG.create_kml_real_route(answer, graph, write_to)
     # MG.create_html_real_route(answer, graph, "ruta_optima_html_real.html")
     
     print(f'\nExecution time \n--> Build: {end-start}, \n--> Clusters: {end2-end}, \n--> Joining: {end3-end2}, \n--> Total: {end3-start}')
     
     shutil.rmtree("__pycache__", ignore_errors=True)

if __name__ == "__main__":
     read_from = "Viena.kml"
     start_point = "Bandgasse 6" # Viena
     # start_point = "Zichy Jenő u. 22" # Budapest
     # start_point = "U Půjčovny 1353/8" # Praga
     # start_point = None # Madrid
     max_walk = 15
     write_to = "VienaRoutesBrute.kml"
     main(start_point, read_from, write_to, max_walk)