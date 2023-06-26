
import pickle
import sys
import pandas as pd


PATH_FILE_MODEL = 'modelo_xgboost.pkl'
PATH_FILE_NORM = 'scaler.pkl'

# Cargar el modelo
model_xgb = pickle.load(open(PATH_FILE_MODEL, 'rb'))
print("El modelo se ha cargado con exito.")

# Cargar el normalizador
scaler = pickle.load(open(PATH_FILE_NORM, 'rb'))
print("El normalizador se ha cargado con exito.")

if len(sys.argv)>0:

  # Captura la ruta del archivo ingresado por CMD
  table_path = sys.argv[1]
  df = pd.read_csv(table_path)

  # Normaliza las variables input
  X_normalized = scaler.transform(df)

  # Prediccion
  predictions = model_xgb.predict_proba(X_normalized)

  # Añadir la prediccion al input y mostrarlo en pantalla
  df['Predicciones'] = predictions
  print(df)

else:
  print('Ingrese la ruta del archivo para poder realizar las predicciones')

