import cv2
import os
import numpy as np

DATASET_DIR = "./processed"
DATA = "./data"
min_w = 20
min_h = 20

# Cargar archivo clasificador
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# obtener una lista de las subcarpetas en data [A, B, C....]
subfolders = os.listdir(DATA)

# creamos las subcarpetas halladas, en la carpeta processed
for sf in subfolders:
    if not os.path.exists(f"{DATASET_DIR}/{sf}"):
        os.mkdir(f"{DATASET_DIR}/{sf}")

for filename_sf in subfolders:
    # ruta de cada subcarpeta de data
    sf_path = os.path.join(DATA,filename_sf) 
    #nombre de cada imagen de cada subcarpeta
    filename_image = os.listdir(sf_path)

    for file_name in filename_image:
        #ruta de cada imagen de cada subcarpeta de data
        image_path_src = sf_path + "/" + file_name
        image_path_src = os.path.abspath(image_path_src)
        image_path_processed = image_path_src.replace("data", "processed" )
        #leer cada imagen
        img = cv2.imread(image_path_src)
        # cambio de color
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # deteccion de la cara
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0 :
            continue
        faces = sorted(faces, key = lambda f:f[2]*f[3])
        #print (face)
        # marcado
        for face in faces[-1:] :
            x, y, w, h = face 
            if w >= min_w or h >= min_h:
                offset = 10
                face_section = img[y - offset : y + h + offset, x - offset : x + w + offset]
                face_section = np.array(face_section)
                if 0 in face_section.shape:
                    continue
      
                #print(face_section)
            
                face_section = cv2.resize(face_section, (224, 224))
                print (image_path_processed)
                cv2.imwrite(
                        image_path_processed, face_section
                )
