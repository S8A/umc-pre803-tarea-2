"""Módulo principal del programa."""


import sys
from typing import Optional, Tuple

import argparse
import http.server
import socketserver
import os


DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_WEB = os.path.join(DIRECTORIO_ACTUAL, "web")
DIRECTORIO_TF = os.path.join(DIRECTORIO_WEB, "tf")
ARCHIVO_MODEL_JSON = os.path.join(DIRECTORIO_TF, "model.json")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Despachador de peticiones HTTP."""

    def __init__(self, request, client_address, server, directory = None) -> None:
        super().__init__(request, client_address, server, directory=DIRECTORIO_WEB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "rn_vocales",
        description=(
            "Corre el servidor web para probar la red neuronal de reconocimiento de "
            "vocales manuscritas."
        )
    )
    parser.add_argument(
        "--puerto",
        type=int,
        help="puerto para el servidor web",
        default=8000,
    )
    parser.add_argument(
        "--exportar-modelo",
        action="store_true",
        help="exporta el modelo de la red neuronal al directorio apropiado"
    )
    args = parser.parse_args()

    # Si se especifica --exportar-modelo, exportar modelo antes de iniciar servidor web
    if args.exportar_modelo:
        # NO RECOMENDADO: Requiere tener GPU y librerías NVIDIA
        import rn_vocales.red_neuronal as rn
        r = rn.preparar_modelo()
        rn.exportar_modelo_tfjs(r["modelo"], str(DIRECTORIO_TF))

    # Muestra error y sale si no existe el archivo exportado del modelo
    if not os.path.isfile(ARCHIVO_MODEL_JSON):
        print(
            "ERROR: Debe exportar el modelo en formato de TensorFlow.js al "
            "directorio `rn_vocales/web/tf` antes de iniciar el servidor."
        )
        # sys.exit(1)

    with socketserver.TCPServer(("", args.puerto), Handler) as httpd:
        print("Servidor corriendo en el puerto", args.puerto)
        httpd.serve_forever()
