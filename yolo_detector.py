import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# --------------- LEER MODELO DE RED NEURONAL ---------------
# Configuración del modelo
config = "C:/Users/Usuario/Desktop/yolo_object_detection/model/yolov3.cfg"
# Pesos
weights = "C:/Users/Usuario/Desktop/yolo_object_detection/model/yolov3.weights"
# Etiquetas
LABELS = open("C:/Users/Usuario/Desktop/yolo_object_detection/model/coco.names").read().split("\n")
#print(LABELS, len(LABELS))
# Colores aleatorios para etiquetas
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
print("colors.shape:", colors.shape)

# Cargar el modelo
net = cv2.dnn.readNetFromDarknet(config, weights)

# Función para seleccionar una imagen
def select_image():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir="C:/Users/Usuario/Desktop/yolo_object_detection/Images/", title="Seleccionar Imagen",
                                           filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp"), ("Todos los archivos", "*.*")))
    root.destroy()
    
    if file_path:
        return file_path
    else:
        return None

# --------------- LEER LA IMAGEN Y PROCESAMIENTO ---------------
image_path = select_image()
if image_path:
    image = cv2.imread(image_path)
#image = cv2.imread("C:/Users/Usuario/Desktop/yolo_object_detection/Images/imagen_0002.jpg")
height, width, _ = image.shape
#
# Crear un blob
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# --------------- DETECCIONES Y PREDICCIONES ---------------
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

net.setInput(blob)
outputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > 0.5:
            box = detection[:4] * np.array([width, height, width, height])
            (x_center, y_center, w, h) = box.astype("int")
            x = int(x_center - (w / 2))
            y = int(y_center - (h / 2))

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIDs.append(classID)

idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
print("idx:", idx)

if len(idx) > 0:
    for i in idx:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = colors[classIDs[i]].tolist()
        text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])

        # Nuevo color de texto: negro
        text_color = (0, 0, 0)

        # Dibujar el rectángulo
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Poner el texto encima del cuadro de detección
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
