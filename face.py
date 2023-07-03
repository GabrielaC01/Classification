import cv2

min_w = 20
min_h = 20

# Cargar archivo clasificador
face_casacade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# para leer la webCam
captura = cv2.VideoCapture(0)

while 1:
    # leer la webCam
    ret, img = captura.read()
    # cambio de color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # deteccion de la face
    face = face_casacade.detectMultiScale(gray, 1.1, 5)

    # marcado
    for (x, y, w, h) in face:
        if w >= min_w or h >= min_h:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # cv2.putText(img,"face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    # mostrar la imagen
    cv2.imshow("img", img)

    # mostrar la imagen
    if cv2.waitKey(1) == ord("q"):
        break

# liberar la captura
captura.release()
# cerrar la ventana
cv2.destroyAllWindows()
