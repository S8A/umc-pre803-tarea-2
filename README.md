# umc-pre803-tarea-2

Red neuronal para reconocimiento de vocales escritas a mano. Hecho como segunda tarea de Programación Emergente (PRE803).


## Equipo

- Jesús Pacheco (V-23.597.642)
- Miguelángel Chávez (V-25.770.050)
- Samuel Ochoa (V-27.225.685)
- Wuilker Álvarez (V-26.440.949)


## Instalación y uso

Clone el repositorio y ejecute los siguientes comandos dentro del directorio clonado para instalar las librerías requeridas y correr el servidor:

```bash
pip install -r requirements.txt
python -m rn_vocales
```

Se recomienda crear un [entorno virtual](https://docs.python.org/3/library/venv.html) y activarlo antes de ejecutar los comandos anteriores, de manera que se eviten posibles problemas de conflictos con las librerías:
```bash
python -m venv venv # Crear (multiplataforma)
source venv/bin/activate # Activar (Linux, bash)
venv\Scripts\activate.bat # Activar (Windows, cmd.exe)
```

Una vez que el servidor esté corriendo podrá acceder a la interfaz web en su navegador a través de `http://localhost`.

Opcionalmente puede especificar el puerto en el que quiera correr el servidor. Por ejemplo:
```
python -m rn_vocales --puerto 8001
```

En este caso accedería a la interfaz web mediante `http://localhost:8001/`.

Una vez que esté en la interfaz web, simplemente dibuje cualquier vocal (mayúscula o minúscula) en el lienzo, y presione el botón *Predecir* para que la red neuronal trate de identificar la vocal dibujada y muestre su predicción.


## Cómo funciona

La red neuronal artificial que realiza la predicción se trata de una red neuronal convolucional compuesta de tres capas de convolución y sus respectivas tres capas de agrupación, seguidas de una capa de abandono (*dropout*) de 50%, dos capas ocultas densas, y una capa de salida de cinco neuronas (correspondientes a las cinco vocales).

Para entrenar la red se utilizó el conjunto de datos [EMNIST Balanced](https://www.tensorflow.org/datasets/catalog/emnist), modificado para eliminar los números y las consonantes, y unir las vocales minúsculas a las mayúsculas.

El código de la red neuronal, hecha en Python 3 con la librería TensorFlow, se encuentra en el archivo `rn_vocales/red_neuronal.py`. Como correr este código requiere mucho hardware, además de algunas librerías de NVIDIA algo engorrosas de instalar, se puso el código en una libreta interactiva de Google Colab, disponible [aquí](https://colab.research.google.com/drive/1Q-SOFa3TAJ5ibwl3jAani7K0gMlZ_j-g). A partir de ese Colab se exportó el modelo de la red neuronal al formato utilizado por TensorFlow JS, y se copiaron los archivos resultantes a `rn_vocales/web/tf/` para que sean accesibles por el código Javascript de la interfaz web.

Al entrar a la interfaz web, el código carga los archivos del modelo TensorFlow JS de forma asíncrona antes de habilitar el botón de *Predecir*. El lienzo utiliza la librería Fabric.js para habilitar el dibujo a mano alzada. Al presionar el botón *Predecir*, los píxeles del lienzo son redimensionados y copiados a un lienzo oculto más pequeño (28x28 píxeles, las dimensiones de las imágenes de EMNIST), se normalizan los valores y se pasan en el formato apropiado al modelo TensorFlow JS para que realice la predicción, y la vocal identificada y la confianza de predicción se muestran en las casillas de resultados.
