Un entorno virtual en Python es como una caja aislada donde puedes instalar y usar paquetes 
sin interferir con otros proyectos o con tu instalación global de Python.

pip3 install virtualenv: Estás instalando virtualenv

virtualenv my_env: Estás creando un nuevo entorno virtual llamado my_env
			cualquier paquete que instales se instalará solo en este entorno

source my_env/bin/activate: Estás activando el entorno virtual my_env. 
			Esto cambia el contexto de Python a este entorno, de modo que cualquier paquete que 
			instales se instalará solo en este entorno 

#Estás instalando varias bibliotecas en el entorno virtual my_env
pip install langchain==0.1.11 gradio==4.21.0 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1: 
#langchain==0.1.11: Una biblioteca para construir aplicaciones de procesamiento de lenguaje natural.
#gradio==4.21.0: Una biblioteca para crear interfaces de usuario interactivas para modelos de aprendizaje automático.
#transformers==4.38.2: Una biblioteca de modelos de lenguaje preentrenados desarrollada por Hugging Face.
#bs4==0.0.2: Beautiful Soup, una biblioteca para analizar documentos HTML y XML.
#requests==2.31.0: Una biblioteca para hacer solicitudes HTTP de manera sencilla.
#torch==2.2.1: PyTorch, una biblioteca de aprendizaje profundo