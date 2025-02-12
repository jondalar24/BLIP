# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:54:29 2024

@author: jonda
"""
# Primero, necesitaremos un modelo de clasificación de imágenes. 
# Para este tutorial, utilizaremos un modelo Resnet-18 
import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# 2. Definición de una función de prediccion
import requests
from PIL import Image
from torchvision import transforms

# Se descargan las etiquetas legibles por humanos para ImageNet
# Las etiquetas se usan para interpretar las salidas del modelo
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Función de prediccion
def predict(inp):
    #Convierte imagen de entrada en imagen PIL y en un tensor pytorch
    inp = transforms.ToTensor()(inp).unsqueeze(0)# indicamos que solo se carga una imagen
    # no calcula gradientes para acelerar el cálculo
    with torch.no_grad():
        # los logits se pasan por softmax para crear probabilidades
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        # Dicc que mapea la probabilidad a la etiqueta
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
        
    return confidences

# 3. Crear la interfaz de Gradio
import gradio as gr

gr.Interface(fn=predict,
        # usa la función prefabricada de arrastrar y soltar
       inputs=gr.Image(type="pil"),
       #muestra las 3 etiquetas más probables
       outputs=gr.Label(num_top_classes=3),
       examples=["leon.jpg", "chimpance.jpg"]).launch()