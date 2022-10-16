"""Red neuronal de reconocimiento de vocales manuscritas.

Aquí se encuentran las funciones para construir y entrenar la red neuronal. 
Para correrlo se requiere un buen hardware (especialmente GPU) además de 
instalar unas librerías de NVIDIA algo engorrosas de instalar. Por lo tanto,
es mejor correr este código en Google Colab y descargar los archivos exportados 
de ahí.

Libreta en Google Colab:
https://colab.research.google.com/drive/1Q-SOFa3TAJ5ibwl3jAani7K0gMlZ_j-g?usp=sharing
"""


from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflowjs as tfjs


# Por alguna razón el conjunto de datos emnist/letters de TensorFlow NO ES
# el mismo EMNIST Letters que se describe en el paper (26 clases balanceadas 
# correspondientes a las letras sin importar mayúsculas o minúsculas).
# En cambio, se trata del subconjunto de las 37 clases de letras del conjunto 
# EMNIST Balanced (26 letras mayúsculas + 11 minúsculas que difieren).
# Por lo tanto, tendremos que unir manualmente las etiquetas de las A y las E 
# que están cada una separadas en mayúscula y minúscula.
# https://arxiv.org/pdf/1702.05373v1.pdf
ETIQUETAS_MNIST = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

VOCALES = list("AEIOU")
ETIQUETAS_VOCALES = [
    i for i, letra in enumerate(ETIQUETAS_MNIST) if letra.upper() in VOCALES
]
TAMANO_LOTE = 128


def transponer_imagen(
    imagen: tf.Tensor, etiqueta: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Transpone los píxeles de las imágenes. Es necesario para que las imágenes 
    de EMNIST estén en la orientación correcta.
    """
    return tf.image.transpose(imagen), etiqueta


def normalizar_pixeles(
    imagen: tf.Tensor, etiqueta: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Normaliza los píxeles convirtiendo enteros (0-255) a decimales (0-1)"""
  return tf.cast(imagen, tf.float32) / 255., etiqueta


def es_vocal(imagen: tf.Tensor, etiqueta: tf.Tensor) -> bool:
    """Indica si la etiqueta corresponde a una vocal."""
    # Retorna verdadero si el número de etiqueta es igual a alguno de los 
    # números de etiqueta correspondientes a vocales mayúsculas o minúsculas
    return tf.math.reduce_any(
        tf.math.equal(etiqueta, tf.constant(ETIQUETAS_VOCALES, dtype=np.int64))
    )


def reenumerar_etiqueta_vocal(
    imagen: tf.Tensor, etiqueta: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reenumera la etiqueta a un valor de 0 a 4."""
    # Índice del número de etiqueta de vocal que corresponde a la etiqueta dada
    etiqueta_vocal = tf.math.argmax(
        tf.math.equal(etiqueta, tf.constant(ETIQUETAS_VOCALES, dtype=np.int64)),
        output_type=np.int32
    )
    # Asociar a cada índice 0-6 (A, E, I, O, U, a, e) una función que retorne 
    # su etiqueta deseada 0-4 (A, E, I, O, U)
    branch_fns = (
        [lambda: tf.constant(i, dtype=np.int64) for i in range(0, 5)]
        + [lambda: tf.constant(i, dtype=np.int64) for i in range(0, 2)]
    )
    # Retorna el número de etiqueta deseado según el índice de la etiqueta vocal
    return imagen, tf.switch_case(etiqueta_vocal, branch_fns=branch_fns)


def preparar_datos() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Prepara los datos de entrenamiento y de prueba.
    
    Retorna:
        Los datos de entrenamiento, los datos de prueba, y la lista de nombres
        de etiquetas (vocales).
    """
    # Cargar conjunto de datos EMNIST Letters usando la librería TensorFlow 
    # Datasets y separarlos en datos de entrenamiento y datos de prueba.
    dataset, metadata = tfds.load(
        'emnist/letters',
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    datos_entrenamiento: tf.data.Dataset = dataset["train"]
    datos_prueba: tf.data.Dataset = dataset["test"]

    # Excluir las letras consonantes de los datos de entrenamiento
    datos_entrenamiento = datos_entrenamiento.filter(es_vocal)
    # Convertir números de etiquetas de vocales al rango [0, 5)
    datos_entrenamiento = datos_entrenamiento.map(reenumerar_etiqueta_vocal)
    # Transponer imágenes para que tengan la orientación correcta
    datos_entrenamiento = datos_entrenamiento.map(
        transponer_imagen, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Normalizar los datos de entrenamiento (convertir valores 0-255 a 0-1)
    datos_entrenamiento = datos_entrenamiento.map(
        normalizar_pixeles, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Guardar datos de entrenamiento en caché para mejorar el rendimiento
    datos_entrenamiento = datos_entrenamiento.cache()
    # Aleatorizar datos de entrenamiento
    datos_entrenamiento = datos_entrenamiento.shuffle(
        metadata.splits['train'].num_examples
    )
    # Agrupar los datos de entrenamiento en lotes
    datos_entrenamiento = datos_entrenamiento.batch(TAMANO_LOTE)
    # Hacer que el conjunto de datos de entrenamiento "pre-cargue" cuantos 
    # elementos pueda mientras se procesa el elemento actual
    datos_entrenamiento = datos_entrenamiento.prefetch(tf.data.AUTOTUNE)

    # Excluir las letras consonantes de los datos de prueba
    datos_prueba = datos_prueba.filter(es_vocal)
    # Convertir números de etiquetas de vocales al rango [0, 5)
    datos_prueba = datos_prueba.map(reenumerar_etiqueta_vocal)
    # Transponer imágenes para que tengan la orientación correcta
    datos_prueba = datos_prueba.map(
        transponer_imagen, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Normalizar los datos de prueba (convertir valores 0-255 a 0-1)
    datos_prueba = datos_prueba.map(
        normalizar_pixeles, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Guardar datos de prueba en caché para mejorar el rendimiento
    datos_prueba = datos_prueba.cache()
    # Agrupar los datos de prueba en lotes
    datos_prueba = datos_prueba.batch(TAMANO_LOTE)
    # Guardar los datos de prueba en caché para mejorar el rendimiento
    # Nótese que para los datos de prueba se guarda en caché después de agrupar 
    # en lotes porque en las pruebas los lotes pueden ser iguales en cada época
    datos_prueba = datos_prueba.cache()
    # Hacer que el conjunto de datos de prueba "pre-cargue" cuantos elementos 
    # pueda mientras se procesa el elemento actual
    datos_prueba = datos_prueba.prefetch(tf.data.AUTOTUNE)

    return datos_entrenamiento, datos_prueba


def construir_modelo() -> tf.keras.Sequential:
    """Crea y compila un modelo secuencial de red neuronal convolucional.
    
    Retorna:
        El modelo de red neuronal convolucional que se utilizará para 
        identificar vocales manuscritas.
    """
    modelo = tf.keras.Sequential([
        # Capa convolucional 2D de entrada con los siguientes parámetros:
        # - 32 filtros de salida de la convolución
        # - Núcleo de dimensiones 3x3
        # - Función de activación ReLU (x si x >= 0, 0 si x < 0)
        # - Forma de los datos de entrada: Arreglo tridimensional 28x28x1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Capa de agrupación cuyas dimensiones de salida son 2x2
        tf.keras.layers.MaxPooling2D(2, 2),

        # Capa convolucional 2D con los siguientes parámetros:
        # - 64 filtros de salida de la convolución
        # - Núcleo de dimensiones 3x3
        # - Función de activación ReLU
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Capa de agrupación cuyas dimensiones de salida son 2x2
        tf.keras.layers.MaxPooling2D(2, 2),

        # Capa convolucional 2D con los siguientes parámetros:
        # - 128 filtros de salida de la convolución
        # - Núcleo de dimensiones 3x3
        # - Función de activación ReLU
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # Capa de agrupación cuyas dimensiones de salida son 2x2
        tf.keras.layers.MaxPooling2D(2, 2),

        # Capa de abandono aleatorio con tasa de 50%:
        # Hace que un 50% de las neuronas, seleccionadas aleatoriamente en cada 
        # época, se desactiven
        tf.keras.layers.Dropout(0.5),
        # Capa de aplanamiento:
        # Toma las salidas de las capas anteriores y las aplana en una sola 
        # capa normal
        tf.keras.layers.Flatten(),
        # Capa densa de 512 neuronas con función de activación ReLU
        tf.keras.layers.Dense(512, activation='relu'),
        # Capa densa de salida con los siguientes parámetros:
        # - 5 neuronas de salida, correspondientes a las 5 vocales
        # - Función de activación softmax, estándar para redes de clasificación
        tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    # Compilar el modelo
    modelo.compile(
        # Optimizador: Algoritmo Adam con tasa de aprendizaje de 0.001
        # El algoritmo de optimización Adam es un tipo particular de método 
        # estocástico de descenso de gradiente. Ajusta los parámetros de las 
        # neuronas conforme aprende la red.
        optimizer=tf.keras.optimizers.Adam(0.001),
        # Función de pérdida: Entropía cruzada categórica dispersa
        # Es una función particular de pérdida, es decir que su propósito es 
        # calcular el error de categorización durante cada iteración del 
        # entrenamiento. La red tratará de minimizar la pérdida.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # Métrica: Precisión categórica dispersa
        # Calcula la frecuencia con la que las predicciones de la red son 
        # correctas, es decir, qué tan frecuentemente identifica la categoría 
        # correcta para cada imagen.
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return modelo


def entrenar_modelo(
    modelo: tf.keras.Model,
    datos_entrenamiento: tf.data.Dataset,
    datos_prueba: tf.data.Dataset,
) -> tf.keras.callbacks.History:
    """Entrena el modelo con los datos dados y evalúa su rendimiento.
    
    Args:
        modelo: El modelo de red neuronal a utilizar.
        datos_entrenamiento: Los datos de entrenamiento agrupados en lotes.
        datos_prueba: Los datos de prueba agrupados en lotes.

    Retorna:
        Historial de entrenamiento y evaluación de desempeño del modelo.
    """
    return modelo.fit(
        datos_entrenamiento,
        epochs=6,
        validation_data=datos_prueba,
    )


def preparar_modelo() -> Dict[str, Any]:
    """Construye, entra y evalúa el modelo de red neuronal."""
    datos_entrenamiento, datos_prueba = preparar_datos()
    modelo = construir_modelo()
    historial = entrenar_modelo(modelo, datos_entrenamiento, datos_prueba)
    return {
        "datos_entrenamiento": datos_entrenamiento,
        "datos_prueba": datos_prueba,
        "modelo": modelo,
        "historial": historial,
    }


def exportar_modelo_tfjs(modelo: tf.keras.Model, directorio: str):
    """Exporta el modelo dado como archivo para ser usado por TensorFlow.js."""
    tfjs.converters.save_keras_model(modelo, directorio)
