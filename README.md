## Proyecto de visión computacional: Detección de cubrebocas y de sana distancia

Segundo Proyecto de la materia de 'Proyecto integrador de tecnologías emergentes' impartida por el profesor Daniel Cabrera

Equipo #3

María Fernanda Mendoza A01745728

Carolina Obregon A01251983

Isabel Navarro A00823132

Jaime Montemayor A01176573

Gustavo De Los Ríos Alatorre A01410922

## Introducción

La enfermedad por coronavirus ha causado millones de muertes alrededor del mundo. Ya que los coronavirus humanos se transmiten de una persona infectada a otra a través del aire al toser y estornudar y la gente no practica distancia social, es importante detectar si un grupo de personas cumplen con los protocolos de sana distancia a traves de la inteligencia artificial. 

Con la ayuda de YOLOv3 y OpenCV se puede construir un modelo de deteccion de distancia social y de idenficiacion de cubrebocas para poder alrmar cuando alguien esta en alto riesgo de contagio. YOLO es un sistema de detección de objetos en tiempo real que utiliza solo una red convolucional, la cual  puede dividir una imagen de entrada en regiones y  por cada región dibujar  en forma de rectángulos los objetos encontrados y obtener la probabilidad de las clases por cada rectángulo. Las capas iniciales extraen las características de la imagen mientras que las últimas predicen las probabilidades y coordenadas. OpenCV es una biblioteca de software de código abierto de visión computacional y aprendizaje automático. En conjunto creamos un modelo que debe identificar  a  todas  las  personas  en  un  video  e  identificar  la  distancia entre  ellos  y  si  están  usando  cubrebocas. Las  siguientes  alertas son desplegadas por el 
sistema:  

  - “Alto riesgo” si hay  personas  que  no  respetan  una  sana  distancia  y  no  están  usando cubrebocas.
  - “Riesgo” si hay  personas que están respetando la sana distancia, pero no estás usando cubrebocas. 
  
## Requerimientos
**Se debe de ejectuar el código en Google Colab para asegurar su funcionamiento adecuado.**
  




## Uso del código
El archivo final del modelo: DeteccionDeMascaras.ipynb

Al realizar el clone al repositorio se obtienen todos los archivos, en caso de no hacerlo, verificar que se tengan todos los archivos de los datos para su correcto funcionamiento.

## Clonado del repositorio
```
git clone https://github.com/carooobregon/proyecto-investigacion2.git
```
## Base de datos
El conjunto de datos utilizado es de ‘Face Mask Dataset (YOLO Format)’ de [Kaggle](https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format). Contiene imágenes de las cuales están clasificadas en ‘no_mask’ y ‘mask’ y está dividido en 3 conjuntos para mayor comodidad. Las imágenes ya están anotadas en formato yolo lo cual implica que ya no se tiene que hacer una modificación al etiquetado. Este conjunto es una recopilación de imágenes de Google, Bing y otros conjuntos de datos de Kaggle. Tiene un total de 1400 imágenes para train, 240  para test y 200 para validation.

Se utiliza un conjunto de datos llamado COCO que consta de 80 etiquetas, que incluyen, entre otras:
- Personas
- Bicicletas
- Autos y camionetas
- Aviones
- Señales de stop y bocas de incendio
- Animales como perros, gatos pájaros, etc.
- Objetos de cocina y comedor, como copas de vino, tazas tenedores, cuchillos, etc.

## Descripción y entrenamiento de los modelos
### Detección de distancia
### Detección de cubrebocas
1. Construir Darknet: clonando darknet del repositorio de AlexeyAB’s, permitir OPENCV y GPU para el makefile, y proceder a construir el Darknet.
2. Descargar los pesos pre entrenados de YOLOv4: YOLOv4 ha sido entrenado con el dataset de coco el cual tiene 80 clases.
3. Etiquetado de dataset: Se utilizó el dataset de Face Mask Dataset (YOLO Format) de Kaggle, del cual solo se usaron las imágenes de train y de test con sus anotaciones en yolo, estas carpetas comprimidas se subieron a una carpeta en drive.
4. Descomprimir carpetas de train y test en el folder de darknet: Se copiaron las carpetas de train y test en la raíz del directorio de Colab VM para después poder descomprimirlas en el folder de darknet.
5. Configurar los archivos para el entrenamiento: copiar yolov4.cfg al google drive para poder editarlo conforme al objeto que queremos detectar:
    - max_batches = (número de clases) * 2000( mínimo 6000, maximo 10,000)
    - steps = (80% de max_batches), (90% de max_batches)
    - filters = (número de  classes + 5) * 3
    - channels=3 for RGB images, 1 for grayscale images
    - classes= número de clases
    - width
    - height
    - volver a subirlo al cloud VM de google drive
6. Se crea un nuevo archivo llamado obj.names: Este archivo contiene el nombre de cada clase línea por línea en el mismo orden que el archivo de classes.txt
7. Se crea un nuevo archivo llamado obj.data: Este archivo contiene el número de clases, las ubicaciones de las carpetas de train, test, obj.names y la ubicación donde se almacenará el entrenamiento.
8. Crear train.txt y test.txt: Estos contienen las ubicaciones de las imágenes de entrenamiento y de testing.
9. Descargar los pesos pre entrenados para capas convolucionales.
10. Entrenar el código: Cada 100 iteraciones darknet guarda el entrenamiento en un archivo en la carpeta que se destino como backup y cada 1000 iteraciones crea un nuevo archivo donde se encontrarán las siguientes 1000 iteraciones.
11. Revisar el Mean Average Precision de cada archivo de pesos, y encontrar el que tenga un valor más alto.
12. Correr el detector de objetos en una imagen o video.



## Resultados

## Páginas consultadas
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788)

https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

