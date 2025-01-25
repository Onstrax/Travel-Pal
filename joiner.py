from typing import List, Optional, Tuple
from graphs import Graph
import numpy as np
import itertools
import random
import math

def brute_force_order(clusters: List[List[str]], graph: Graph, start_point: Optional[str] = None) -> List[List[str]]:
    """
    Encuentra el mejor orden para recorrer los clústeres minimizando el tiempo total entre ellos.
    Considera la inversión de clústeres y un punto de inicio opcional.
    """
    # Preprocesar clústeres: generar todas las orientaciones posibles (original/invertido)
    processed_clusters = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue  # Según enunciado, no hay clústeres vacíos
        elif len(cluster) == 1:
            processed_clusters.append((cluster[0], cluster[0]))  # (start, end)
        else:
            # Considerar ambas orientaciones
            processed_clusters.append((cluster[0], cluster[-1]))  # Original
            processed_clusters.append((cluster[-1], cluster[0]))  # Invertido

    n = len(clusters)
    best_permutation = None
    min_total_time = math.inf

    # Manejar punto de inicio
    start_cluster_indices = []
    if start_point is not None:
        for i, cluster in enumerate(clusters):
            if start_point in {cluster[0], cluster[-1]}:
                start_cluster_indices.append(i)

    # Algoritmo de fuerza bruta optimizado para n <= 15
    for perm in itertools.permutations(range(n)):
        # Verificar restricción de punto de inicio
        if start_point is not None and perm[0] not in start_cluster_indices:
            continue

        total_time = 0.0
        prev_end = None
        
        for cluster_idx in perm:
            original_cluster = clusters[cluster_idx]
            orientation_options = [(original_cluster[0], original_cluster[-1])]
            if len(original_cluster) > 1:
                orientation_options.append((original_cluster[-1], original_cluster[0]))
            
            best_cluster_time = math.inf
            best_start, best_end = None, None
            
            for start, end in orientation_options:
                if prev_end is None:
                    current_time = 0.0
                else:
                    current_time = graph.get_time(prev_end, start) or math.inf
                
                if current_time < best_cluster_time:
                    best_cluster_time = current_time
                    best_start = start
                    best_end = end
            
            if best_cluster_time == math.inf:
                break
            
            total_time += best_cluster_time
            prev_end = best_end
        
        if total_time < min_total_time:
            min_total_time = total_time
            best_permutation = perm

    # Reconstruir la secuencia final con orientaciones óptimas
    final_order = []
    prev_end = None
    for cluster_idx in best_permutation:
        cluster = clusters[cluster_idx]
        options = [
            (cluster, cluster[0], cluster[-1]),
            (cluster[::-1], cluster[-1], cluster[0]) if len(cluster) > 1 else (cluster, cluster[0], cluster[0])
        ]
        
        best_option = None
        min_cost = math.inf
        for option in options:
            if prev_end is None:
                cost = 0.0
            else:
                cost = graph.get_time(prev_end, option[1]) or math.inf
            
            if cost < min_cost:
                min_cost = cost
                best_option = option[0]
                current_end = option[2]
        
        final_order.append(best_option)
        prev_end = current_end

    return final_order

def nearest_neighbor_order(
    clusters: List[List[str]], 
    graph: Graph, 
    start_point: Optional[str] = None
) -> List[List[str]]:
    """
    Ordena clústeres usando la heurística del vecino más cercano.
    - Complejidad: O(n²)
    - Ideal para n > 15.
    """
    clusters = clusters.copy()
    ordered = []
    
    # Precalcular todos los puntos iniciales y finales
    endpoints = [
        (cluster[0], cluster[-1]) if len(cluster) > 1 else (cluster[0], cluster[0])
        for cluster in clusters
    ]
    
    # Manejar punto de inicio
    current_end = None
    if start_point:
        for i, (start, end) in enumerate(endpoints):
            if start_point == start or start_point == end:
                current_end = end if start_point == start else start
                ordered.append(clusters[i] if start_point == start else clusters[i][::-1])
                del clusters[i]
                del endpoints[i]
                break
    
    # Si no hay punto de inicio, comenzar con el primer clúster
    if not current_end and clusters:
        current_end = endpoints[0][1]
        ordered.append(clusters[0])
        del clusters[0]
        del endpoints[0]
    
    # Iterar hasta conectar todos los clústeres
    while clusters:
        min_time = math.inf
        best_cluster = None
        best_orientation = None
        
        for i, (start, end) in enumerate(endpoints):
            # Calcular tiempo para orientación original
            time_original = graph.get_time(current_end, start) or math.inf
            # Calcular tiempo para orientación invertida (si aplica)
            time_inverted = graph.get_time(current_end, end) or math.inf if start != end else math.inf
            
            if time_original < min_time:
                min_time = time_original
                best_cluster = clusters[i]
                best_orientation = "original"
            if time_inverted < min_time:
                min_time = time_inverted
                best_cluster = clusters[i][::-1]  # Invertir el clúster
                best_orientation = "inverted"
        
        if best_cluster:
            ordered.append(best_cluster)
            current_end = best_cluster[-1]
            # Eliminar el clúster seleccionado
            idx = clusters.index(best_cluster if best_orientation == "original" else best_cluster[::-1])
            del clusters[idx]
            del endpoints[idx]
    
    return ordered

# --------------------------
# Algoritmo Genético
# --------------------------

class GeneticClusterOptimizer:
    """
    Optimiza el orden de clústeres usando un algoritmo genético.
    - Parámetros ajustables: población, generaciones, mutación.
    - Complejidad: O(población * generaciones * n²)
    """
    def __init__(
        self, 
        graph: Graph,
        population_size=500,
        generations=10000,
        mutation_rate=0.08,
        elite_size=20
    ):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.start_clusters = []
    
    def optimize(
        self, 
        clusters: List[List[str]], 
        start_point: Optional[str] = None
    ) -> List[List[str]]:
        self.clusters = clusters
        self.start_point = start_point
        self.endpoints = self._precompute_endpoints()
        
        if self.start_point:
            self.start_clusters = [
                cluster for cluster in clusters 
                if self.start_point in {cluster[0], cluster[-1]}
            ]
            if not self.start_clusters:
                raise ValueError("Start point no está en ningún extremo de los clústeres")
        
        # Generar población inicial
        population = [self._generate_valid_individual() for _ in range(self.population_size)]
        
        for _ in range(self.generations):
            # Evaluar aptitud
            fitness = [self._calculate_fitness(ind) for ind in population]
            sorted_pop = [x for _, x in sorted(zip(fitness, population))]
            
            # Seleccionar elite y generar nueva población
            new_population = sorted_pop[:self.elite_size]
            while len(new_population) < self.population_size:
                parent1 = self._select_parent(sorted_pop)
                parent2 = self._select_parent(sorted_pop)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Devolver el mejor individuo
        best_individual = min(population, key=lambda x: self._calculate_fitness(x))
        return best_individual
    
    def _precompute_endpoints(self) -> List[Tuple[str, str]]:
        return [
            (cluster[0], cluster[-1]) if len(cluster) > 1 else (cluster[0], cluster[0])
            for cluster in self.clusters
        ]
        
    def _generate_valid_individual(self) -> List[List[str]]:
        """Genera individuos que cumplen con el start_point."""
        individual = []
        remaining = self.clusters.copy()
        
        # Caso con start_point: forzar primer clúster válido
        if self.start_point:
            # Elegir un clúster inicial que contenga el start_point
            first_cluster = random.choice(self.start_clusters).copy()
            
            # Invertir si el start_point está al final
            if first_cluster[-1] == self.start_point:
                first_cluster = first_cluster[::-1]
                
            individual.append(first_cluster)
            remaining.remove(first_cluster if first_cluster in remaining else first_cluster[::-1])
        
        # Añadir el resto de clústeres aleatoriamente
        while remaining:
            cluster = random.choice(remaining)
            if random.random() < 0.5 and len(cluster) > 1:
                individual.append(cluster[::-1])
            else:
                individual.append(cluster)
            remaining.remove(cluster)
            
        return individual
    
    def _generate_individual(self) -> List[List[str]]:
        # Generar individuo aleatorio, considerando orientaciones
        individual = []
        remaining = self.clusters.copy()
        while remaining:
            cluster = random.choice(remaining)
            if random.random() < 0.5 and len(cluster) > 1:
                individual.append(cluster[::-1])
            else:
                individual.append(cluster)
            remaining.remove(cluster)
        return individual
    
    def _calculate_fitness(self, individual: List[List[str]]) -> float:
        # Tiempo total de la ruta
        total_time = 0.0
        current_end = None
        
        # Penalización masiva si hay start_point y no se cumple
        if self.start_point:
            first_cluster_start = individual[0][0]
            if first_cluster_start != self.start_point:
                return math.inf  # Descarta inmediatamente soluciones inválidas
        
        for cluster in individual:
            start = cluster[0]
            if current_end:
                total_time += self.graph.get_time(current_end, start) or math.inf
            current_end = cluster[-1]
        
        return total_time
    
    def _select_parent(self, population: List[List[List[str]]]) -> List[List[str]]:
        # Selección por torneo
        tournament = random.sample(population, 5)
        return min(tournament, key=lambda x: self._calculate_fitness(x))
    
    def _crossover(self, parent1: List[List[str]], parent2: List[List[str]]) -> List[List[str]]:
        # Cruce OX (Order Crossover)
        # Forzar herencia del clúster inicial si hay start_point
        if self.start_point:
            child = [parent1[0]]  # Heredar primer clúster del padre1
            remaining = [c for c in parent2 if c != parent1[0] and c[::-1] != parent1[0]]
        else:
            child = []
            remaining = parent2.copy()
            
        # Completa con clústeres del padre2 que no estén ya en el hijo
        for cluster in parent1[1:]:
            if cluster not in child and cluster[::-1] not in child:
                child.append(cluster)
                
        return child + [c for c in remaining if c not in child]
        
        # start = random.randint(0, len(parent1)-1)
        # end = random.randint(start, len(parent1)-1)
        # child = parent1[start:end]
        # remaining = [c for c in parent2 if c not in child]
        # return child + remaining
    
    def _mutate(self, individual: List[List[str]]) -> List[List[str]]:
        # Mutación por inversión de orientación o intercambio
        # No mutar el primer clúster si hay start_point
        start = 1 if self.start_point else 0
        
        if random.random() < self.mutation_rate and len(individual) > start:
            idx = random.randint(start, len(individual)-1)
            if len(individual[idx]) > 1:
                individual[idx] = individual[idx][::-1]
                
        if random.random() < self.mutation_rate and len(individual) > start+1:
            idx1, idx2 = random.sample(range(start, len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            
        return individual
        
        # if random.random() < self.mutation_rate:
        #     idx = random.randint(0, len(individual)-1)
        #     if len(individual[idx]) > 1:
        #         individual[idx] = individual[idx][::-1]
        # if random.random() < self.mutation_rate:
        #     idx1, idx2 = random.sample(range(len(individual)), 2)
        #     individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        # return individual


def optimize_cluster_order(
    clusters: List[List[str]], 
    graph: Graph, 
    start_point: Optional[str] = None,
    method: str = "genetic"
) -> List[List[str]]:
    """
    Función unificada para seleccionar el método de optimización.
    - method: "genetic" (precisión) o "nearest" (rapidez).
    """
    if method == "brute":
        return brute_force_order(clusters, graph, start_point)
    elif method == "nearest":
        return nearest_neighbor_order(clusters, graph, start_point)
    elif method == "genetic":
        optimizer = GeneticClusterOptimizer(graph)
        return optimizer.optimize(clusters, start_point)
    else:
        raise ValueError("Método no válido. Opciones: 'brute', 'genetic', 'nearest'")