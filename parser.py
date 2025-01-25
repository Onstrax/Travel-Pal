import xml.etree.ElementTree as ET
import os

import sys
sys.dont_write_bytecode = True

def parse_kml_to_dict(kml_name: str) -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kml_filename = os.path.join(base_dir, kml_name)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Cargar el archivo KML
    tree = ET.parse(kml_filename)
    root = tree.getroot()
    
    places = {}
    
    # Buscar todos los elementos <Placemark>
    for placemark in root.findall('.//kml:Placemark', ns):
        # Extraer el nombre (si existe)
        name_element = placemark.find('kml:name', ns)
        if name_element is None or name_element.text.strip() == '':
            continue  # Saltar placemarks sin nombre
        
        name = name_element.text.strip()
        
        # Extraer coordenadas (pueden estar en Point, LineString, Polygon, etc.)
        coordinates_element = placemark.find('.//kml:coordinates', ns)
        if coordinates_element is None or coordinates_element.text is None:
            continue  # Saltar si no hay coordenadas
        
        coordinates = ','.join(' '.join(coordinates_element.text.strip().splitlines()).rsplit(',', 1)[:-1])
        
        # Guardar en el diccionario
        places[name] = coordinates
    
    return places

# Ejemplo de uso
if __name__ == "__main__":
     lugares = parse_kml_to_dict("POIs.kml")
     print(f"Lugares encontrados: {len(lugares)}")
     for nombre, coords in lugares.items():
          print(f"{nombre}: {coords}")