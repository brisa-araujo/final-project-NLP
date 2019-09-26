# Ironhack Data Analytics Bootcamp Mexico Final Project

## Metodología para el Análisis de Evaluación Sensorial
___

### Antecedentes y justificación:

- La industria alimenticia es una de las que más consume investigación de mercados actualmente
- Siguen trabajando con metodologías tradicionales y con un abismo grande entre la investigación con consumidores y sus áreas de R&D
- Las recomendaciones dadas por los consultores de MRX están en su mayoría desconectadas de las bdds y corpus especializados, lo que muchas veces genera un trabajo doble a los food scientists de reinterpretar los datos obtenidos por la investigación de mercados, con el fin de traducirlos a datos técnicos necesarios para su trabajo.
- Este problema se da específicamente en la expresión linguística de variables sensoriales por consumidores no entrenados, que suelen ser muy vagas y repetitivas - por ejemplo, un sabor "dulce empalagoso" o una textura "un poco chiclosa".
- Aunque haya intentos de mejorar el input de los consumidores con el uso de imágenes y escalas representativas de intensidad de sabor, dulzor, etc durante la investigación, la industria como un todo se beneficiaría de métodos de análisis que logren encontrar conexiones ocultas en la data, de forma a reducir el abismo entre el lenguaje "común" y el input necesario para mejorar procesos científicos.
- La relación entre lenguaje y métricas idealmente se daría a través de inputs tanto cualitativos como cuantitativos, una vez que los cuantitativos pueden ser posteriormente modelados en conjunto con métricas existentes en la indústria, mientras los cualitativos generarían la conexión semántica entre datos especializados y no especializados.

### Propuesta:
Desarrollar un modelo de análisis de la libre expresión linguística de los consumidores con relación a los atributos sensoriales de los productos.

#### Objetivo general:
Entender a profundidad el significado de las descripciones de sabores de productos alimentícios por consumidores no entrenados.

#### Objetivos Específicos:
- Analizar el discurso de consumidores con relación a atributos sensoriales de productos de cerveza evaluados en la plataforma [www.beeradvocate.com]:
    - Sabores
    - Texturas
    - Atributos emocionales / subjetivos
- Clusterizar productos de acuerdo con sus descripciones, generando un mapeo de productos similares y distintos

#### Recolección de datos:
- Dataset original obtenido en la plataforma Kaggle[https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews]

#### Etapas de trabajo:
- Limpieza de Datos:
    - Preparación de los datasets de reviews y cerveza para análisis por producto
    - Selección de variables a ser utilizadas
    - Procesamiento de texto
    - Vectorización de texto (Bag of Words)
- Modelos - Aprendizaje no supervisado:
    - LDA: separación de los reviews en tópicos
    - T-SNE: clusterización de tópicos
- Análisis descriptivo:
    - Visualización de clusters
