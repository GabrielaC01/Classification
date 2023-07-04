from torchvision import transforms
import torch
import cv2

from index_to_letter import index_to_letter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Dispositivo actual:", DEVICE)
PATH_WEIGHTS = "./weights/best_weights_50.pt"
PATH_TEST = "./test"

# Cargar el modelo entrenado
asl_model = torch.load(PATH_WEIGHTS)
asl_model.to(DEVICE)
asl_model.eval()

min_w = 20
min_h = 20

# Cargar el clasificador de caras
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# Capturar video de la cámara
captura = cv2.VideoCapture(0)

while 1:
    # Leer el video de la cámara
    ret, img = captura.read()
    # Cambiar el color de la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detección de caras
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for face in faces[-1:]:
        x, y, w, h = face
        if w >= min_w or h >= min_h:
            offset = 10
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_section = img[y - offset: y + h + offset, x - offset: x + w + offset]
            
            # Aplicar transformaciones a la sección de la cara
            transformations = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            face_section_tensor = transformations(face_section)
            face_section_tensor = torch.unsqueeze(face_section_tensor, 0)
            face_section_tensor = face_section_tensor.to(DEVICE)
            #print(face_section_tensor)

            # Realizar la predicción utilizando el modelo
            prediction = asl_model(face_section_tensor)
            index = torch.argmax(prediction).item()
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 80)
            fontScale = 2
            color = (0, 84, 211)
            thickness = 3
            nombre = ''

            if index == 0:
                nombre = "HOMBRE"
            elif index == 1:
                nombre = "MUJER"

            # Mostrar el resultado de la predicción en la imagen
            cv2.putText(
                img,
                nombre,
                org,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    # Mostrar la imagen
    cv2.imshow("img", img)

    # Salir del bucle al presionar la tecla 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Liberar la captura de video
captura.release()
# Cerrar la ventana
cv2.destroyAllWindows()
