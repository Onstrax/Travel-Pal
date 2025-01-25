from simplekml import Kml, Style, Color
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
