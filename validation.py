import os
import random
import shutil

processed_dir = "./processed"
validation_dir = "./validation"
validation_ratio = 0.2

# Obtener una lista de las subcarpetas en "processed"
subfolders = os.listdir(processed_dir)

# Crear la carpeta "validation" si no existe
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

# Iterar sobre las subcarpetas
for sf in subfolders:
    # Ruta de cada subcarpeta en "processed"
    sf_path = os.path.join(processed_dir, sf)
    # Lista de archivos en la subcarpeta
    files = os.listdir(sf_path)
    # Calcular la cantidad de archivos a mover a "validation"
    num_files = len(files)
    num_validation = int(num_files * validation_ratio)
    
    # Crear la subcarpeta correspondiente en "validation"
    validation_sf_path = os.path.join(validation_dir, sf)
    if not os.path.exists(validation_sf_path):
        os.mkdir(validation_sf_path)
    
    # Mover los archivos a "validation"
    files_to_move = random.sample(files, num_validation)
    for file_name in files_to_move:
        src = os.path.join(sf_path, file_name)
        dst = os.path.join(validation_sf_path, file_name)
        shutil.move(src, dst)

print("Mover datos a la carpeta de validación completado.")
