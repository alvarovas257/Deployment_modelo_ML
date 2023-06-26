# Usa un tiempo de ejecución oficial de python como imagen base
FROM python:3.9

# Setea el directorio de trabajo en el container
WORKDIR /app

# Copia el archivo requirements.txt dentro del container
COPY requirements.txt .

# Instala las dependencias de python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el modelo y el normalizador dentro del container
COPY modelo_xgboost.pkl .
COPY scaler.pkl .

# Copia el script de python dentro del container
COPY main.py .

# Corre el script de python con lanzamiento de contenedores
ENTRYPOINT ["python","main.py"]
