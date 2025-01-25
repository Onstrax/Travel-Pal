from simplekml import Kml, Style, Color, Icon
from graphs import Graph
import folium

def create_kml_route(optimized_route: list[list[str]], graph: Graph, output_filename: str = "ruta_optima.kml"):
    kml = Kml()
    
    # Estilo para la línea de la ruta
    line_style = Style()
    line_style.linestyle.color = Color.rgb(255, 0, 0)  # Rojo
    line_style.linestyle.width = 3
    
    # Crear una carpeta para la ruta
    route_folder = kml.newfolder(name="Ruta Óptima")
    
    # Coordenadas de todos los puntos en orden
    all_coords = []
    for cluster in optimized_route:
        for point_name in cluster:
            vertex = graph.vertices[point_name]
            lon, lat = vertex.coords.split(",")
            all_coords.append((float(lon), float(lat)))
    
    # Crear la línea continua de la ruta
    line = route_folder.newlinestring(name="Ruta Principal")
    line.coords = all_coords
    line.style = line_style
    
    # Añadir marcadores para cada punto
    for idx, (lon, lat) in enumerate(all_coords):
        point = route_folder.newpoint(name=f"Punto {idx+1}")
        point.coords = [(lon, lat)]
        point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pal2/icon18.png"
    
    kml.save(output_filename)
    print(f"KML generado: {output_filename}")


def create_html_map(optimized_route: list[list[str]], graph: Graph, output_filename: str = "ruta_optima.html"):
    # Obtener todas las coordenadas
    all_coords = []
    for cluster in optimized_route:
        for point_name in cluster:
            vertex = graph.vertices[point_name]
            lon, lat = vertex.coords.split(",")
            all_coords.append((float(lat), float(lon)))  # Folium usa (lat, lon)
    
    # Crear mapa centrado en el primer punto
    map_center = all_coords[0] if all_coords else (0, 0)
    m = folium.Map(location=map_center, zoom_start=14)
    
    # Añadir línea de la ruta
    folium.PolyLine(
        locations=all_coords,
        color="red",
        weight=5,
        opacity=0.8
    ).add_to(m)
    
    # Añadir marcadores numerados
    for idx, (lat, lon) in enumerate(all_coords):
        folium.Marker(
            location=(lat, lon),
            popup=f"Punto {idx+1}: {optimized_route[0][idx] if idx < len(optimized_route[0]) else ''}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
    
    m.save(output_filename)
    print(f"Mapa HTML generado: {output_filename}")
    

def create_kml_real_route(optimized_route: list[list[str]], graph: Graph, output_filename: str = "ruta_real.kml"):
    kml = Kml()
    
    # ==============================================
    # 1. Configuración de estilos
    # ==============================================
    # Estilo para las líneas intra-cúmulo
    intra_style = Style()
    intra_style.linestyle.color = Color.rgb(0, 128, 255)  # Azul
    intra_style.linestyle.width = 4
    
    # Estilo para las líneas inter-cúmulo
    inter_style = Style()
    inter_style.linestyle.color = Color.rgb(255, 0, 0)    # Rojo
    inter_style.linestyle.width = 4
    inter_style.linestyle.dash = "10 5"
    
    # Estilo para los marcadores de puntos
    poi_style = Style()
    poi_style.iconstyle.icon = Icon(
        href="http://maps.google.com/mapfiles/kml/paddle/blank.png"
    )
    poi_style.iconstyle.scale = 1.5

    # ==============================================
    # 2. Estructura de carpetas
    # ==============================================
    main_folder = kml.newfolder(name="Ruta Turística Optimizada")
    lines_folder = main_folder.newfolder(name="Tramos")
    points_folder = main_folder.newfolder(name="Puntos de Interés (POIs)")

    # ==============================================
    # 3. Procesar todos los puntos y tramos
    # ==============================================
    point_counter = 1  # Contador global para numeración
    
    for cluster_idx, cluster in enumerate(optimized_route, 1):
        # -- A. Crear carpeta para el cúmulo actual --
        cluster_folder = lines_folder.newfolder(name=f"Cúmulo {cluster_idx}")
        
        # -- B. Procesar cada punto del cúmulo --
        for point_order, point_name in enumerate(cluster, 1):
            # Obtener coordenadas del vértice
            poi = points_folder.newpoint(name=f"{point_counter}) {point_name}")
            vertex = graph.vertices[point_name]
            lon, lat = map(float, vertex.coords.split(","))
            poi.coords = [(lon, lat)]

            point_style = Style()
            point_style.iconstyle.color = Color.rgb(0, 0, 255)

            poi.style = point_style
            poi.description = (
                f"Orden en ruta: {point_counter}\n"
                f"Orden en cúmulo: {point_order}\n"
                f"Coordenadas: {lat:.6f}°N, {lon:.6f}°E"
            )
            
            point_counter += 1  # Incrementar contador global

        # -- C. Dibujar tramos intra-cúmulo --
        for i in range(len(cluster)-1):
            origin = cluster[i]
            dest = cluster[i+1]
            
            edge = graph.edges.get(frozenset({origin, dest}))
            
            # Crear línea con geometría real o fallback
            if edge and edge.geometry:
                coords = [(lon, lat) for (lat, lon) in edge.geometry]
            else:
                v1 = graph.vertices[origin]
                v2 = graph.vertices[dest]
                coords = [
                    tuple(map(float, v1.coords.split(","))),
                    tuple(map(float, v2.coords.split(",")))
                ]
            
            line = cluster_folder.newlinestring(name=f"{origin} → {dest}")
            line.coords = coords
            line.style = intra_style

        # -- D. Conexiones inter-cúmulos --
        if cluster_idx < len(optimized_route):
            next_cluster = optimized_route[cluster_idx]
            last_point = cluster[-1]
            first_next = next_cluster[0]
            
            v1 = graph.vertices[last_point]
            v2 = graph.vertices[first_next]
            
            conn_line = lines_folder.newlinestring(name=f"Enlace C{cluster_idx}-C{cluster_idx+1}")
            conn_line.coords = [
                tuple(map(float, v1.coords.split(","))),
                tuple(map(float, v2.coords.split(",")))
            ]
            conn_line.style = inter_style

    # ==============================================
    # 4. Guardar y exportar
    # ==============================================
    kml.save(output_filename)
    print(f"KML profesional generado: {output_filename}")


def create_html_real_route(optimized_route: list[list[str]], graph: Graph, output_filename: str = "ruta_real.html"):
    m = folium.Map(location=(48.1351, 11.5820), zoom_start=14)  # Ejemplo para Múnich
    
    # Dibujar todos los segmentos
    for cluster in optimized_route:
        for i in range(len(cluster)-1):
            origin = cluster[i]
            dest = cluster[i+1]
            
            edge = graph.edges.get(frozenset({origin, dest}))
            if edge and edge.geometry:
                # Convertir geometría a (lat, lon) para Folium
                folium_coords = [(lat, lon) for (lon, lat) in edge.geometry]
                folium.PolyLine(
                    locations=folium_coords,
                    color="#FF0000",
                    weight=3,
                    opacity=0.8
                ).add_to(m)
    
    m.save(output_filename)
    print(f"Mapa HTML con rutas reales generado: {output_filename}")