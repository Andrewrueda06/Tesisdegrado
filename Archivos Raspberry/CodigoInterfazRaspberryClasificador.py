import cv2
import tensorflow_hub as hub
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import time
from tkinter import ttk
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from tkinter import messagebox
import pandas as pd

modelo = load_model('/home/pi1/Modelofinal7.h5')

clases = {0: 'Basal Cell Carcinoma', 1: 'Pigmented Benign Keratosis', 2: 'Melanoma', 3: 'Nevus'}

camara = cv2.VideoCapture(0)
w1=camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
h1=camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ruta_guardado = '/home/pi1/Downloads/DatabaseCapture4'

labels_true = []
labels_pred = []

modelo.summary()

def clasificar_imagen(im0):
    im0 = cv2.resize(im0,(180,180))
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    im0 = im0.astype("float32") /255.0
    im0 = np.expand_dims(im0, axis=0)
    
    results = modelo.predict(im0)
    class_ind = np.argmax(results, axis = -1)
    
    return clases[class_ind[0]]

def actualizar_imagen():
    ret, frame = camara.read()
    if ret:
        etiqueta_clasificacion.config(text=f'Classification: {clasificar_imagen(frame)}')
       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)
        panel_imagen.config(image=img)
        panel_imagen.image = img

        ventana.after(500, actualizar_imagen)

def capturar_imagen():
    ret, frame = camara.read()
    if ret:
        label_number = entry_label.get()
        clase_actual = clasificar_imagen(frame)
        carpeta_clase = os.path.join(ruta_guardado, clase_actual)

        if not os.path.exists(carpeta_clase):
            os.makedirs(carpeta_clase)

        timestamp = int(time.time())
        nombre_archivo = f'imagen_{timestamp}.jpg'
        ruta_guardado_imagen = os.path.join(carpeta_clase, nombre_archivo)

        cv2.imwrite(ruta_guardado_imagen, frame)
        status_var.set(f'Image saved in: {ruta_guardado_imagen}')
        
        labels_true.append(int(label_number))
        labels_pred.append(list(clases.keys())[list(clases.values()).index(clase_actual)])
def actualizar_matriz_confusion():
    cm = confusion_matrix(labels_true, labels_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette("rocket_r", as_cmap=True), xticklabels=clases.values(), yticklabels=clases.values())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def reiniciar_matriz_confusion():
    global labels_true, labels_pred
    labels_true = []
    labels_pred = []
    status_var.set('Confusion matrix reset')

def mostrar_matriz_confusion():
    actualizar_matriz_confusion()
    
def mostrar_metricas():
    accuracy = accuracy_score(labels_true, labels_pred)
    precision = precision_score(labels_true, labels_pred, average='weighted', zero_division=1)
    recall = recall_score(labels_true, labels_pred, average='weighted', zero_division=1)
    f1 = f1_score(labels_true, labels_pred, average='weighted', zero_division=1)
    
    report = classification_report(labels_true, labels_pred, target_names=clases.values(), output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.sort_index(inplace=True)
    print("\n")
    print(df)
    print("\n")
    

ventana = tk.Tk()
ventana.title('SKIN CANCER IMAGE DETECTOR')
ventana.configure(bg='#FAB279')

btn_bg_color = '#DC88FA'
btn_relief = "raised"

panel_imagen = tk.Label(ventana)
panel_imagen.grid(row=0, column=0, columnspan=2)

etiqueta_clasificacion = tk.Label(ventana, text='Classification: ', bg='#DC88FA')
etiqueta_clasificacion.grid(row=2, column=0, pady=10)

label_frame = tk.Frame(ventana, bg='#DC88FA')
label_frame.grid(row=2, column=1, pady=10)
label_entry = tk.Label(label_frame, text='Label', bg='#DC88FA')
label_entry.grid(row=0, column=1)
entry_label = tk.Entry(label_frame)
entry_label.grid(row=0, column=2)

btn_capturar = tk.Button(ventana, text='Save Image', command=capturar_imagen, bg=btn_bg_color, relief=btn_relief)
btn_capturar.grid(row=3, column=0, pady=10)

btn_cerrar = tk.Button(ventana, text='Close Window', command=ventana.destroy, bg=btn_bg_color, relief=btn_relief)
btn_cerrar.grid(row=4, column=0, pady=10)

btn_reiniciar_matriz = tk.Button(ventana, text='Reset Matrix', command=reiniciar_matriz_confusion, bg='#DC88FA')
btn_reiniciar_matriz.grid(row=3, column=1, pady=10)

btn_actualizar_matriz = tk.Button(ventana, text='Update Matrix', command=mostrar_matriz_confusion, bg=btn_bg_color, relief=btn_relief)
btn_actualizar_matriz.grid(row=4, column=1, pady=10)

btn_metricas = tk.Button(ventana, text='Show Metrics', command=mostrar_metricas, bg=btn_bg_color, relief=btn_relief)
btn_metricas.grid(row=5, column=0, pady=10)

status_var = tk.StringVar()
status_bar = tk.Label(ventana, textvariable=status_var, anchor="w", font=("Helvetica", 8), bg='#DC88FA')
status_bar.grid(row=6, column=0, columnspan=2, sticky="ew")

matriz_confusion_window = tk.Toplevel(ventana)
matriz_confusion_window.title('Confusion Matrix')

actualizar_imagen()

ventana.mainloop()

camara.release()

