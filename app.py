import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yolov5

# Configuración inicial
st.set_page_config(page_title="Detección de Objetos con YOLOv5", page_icon="🤖")
st.title("🤖 Detección de Objetos en Imágenes")
st.write("Usa esta aplicación para detectar objetos en imágenes capturadas o subidas.")

# Cargar modelo YOLOv5
try:
    model = yolov5.load('yolov5s.pt')
except Exception as e:
    st.error("⚠️ Error al cargar el modelo. Asegúrate de que el archivo 'yolov5s.pt' esté en el directorio adecuado.")
    st.stop()

# Parámetros configurables del modelo
st.sidebar.header("⚙️ Configuración del Modelo")
st.sidebar.write("Ajusta los umbrales de confianza e IoU para optimizar la detección.")
model.iou = st.sidebar.slider("📏 IoU (Intersección sobre Unión)", 0.0, 1.0, 0.45, 0.01)
model.conf = st.sidebar.slider("🔍 Confianza mínima", 0.0, 1.0, 0.25, 0.01)

# Entrada de la imagen
st.header("📸 Captura o Subida de Imagen")
picture = st.camera_input("Captura una foto") or st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if picture:
    # Procesar la imagen capturada o cargada
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Redimensionar para visualización rápida
    max_width = 800
    height, width, _ = cv2_img.shape
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, int(height * scale))
        cv2_img = cv2.resize(cv2_img, new_size)

    # Realizar detección de objetos
    results = model(cv2_img)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Visualizar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2_img, channels="BGR", caption="Imagen Original")
    with col2:
        results.render()
        st.image(cv2_img, channels="BGR", caption="Imagen con Detección")

    # Conteo y visualización de categorías detectadas
    st.subheader("📊 Resumen de Detecciones")
    label_names = model.names
    category_count = {}
    for category in categories:
        category_name = label_names[int(category)]
        category_co

