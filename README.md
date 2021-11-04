## Clasificador para datos tabulares de Alzheimer

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
### Detección de cubrebocas


## Resultados

## Páginas consultadas
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788)

https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

