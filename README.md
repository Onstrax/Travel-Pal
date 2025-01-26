Datos geograficos distancias
https://download.geofabrik.de

Habilitar WSL y Virtual Machine Platform
Instalar el kernel de WSL2
Configurar WSL2 como versión predeterminada
Instalar una distribución de Linux
Descargar Docker Desktop para Windows
Configurar docker con WSL2

docker pull osrm/osrm-backend

docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-extract -p /opt/foot.lua /data/austria-latest.osm.pbf

docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-partition /data/austria-latest.osrm

docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-customize /data/austria-latest.osrm

docker run -t -i -p 5000:5000 -v ${PWD}:/data osrm/osrm-backend osrm-routed --algorithm mld /data/austria-latest.osrm

docker ps
docker stop <ID_o_NOMBRE_CONTENEDOR>