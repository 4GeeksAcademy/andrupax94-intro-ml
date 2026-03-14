from dotenv import load_dotenv
from sqlalchemy import create_engine

import pandas as pd
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

def create_factor_transf_and_json(column, df,folder_name, transformation = False,transformation_func = lambda x: x):
    file_name = ""
        
    if transformation:
         # ahora se le pasa a la funcion una funcion de trnasformacion
        df_aux = df[[column]].copy()
        df[column+"_transf"] = df_aux[column].apply(transformation_func)
        folder_path = '../data/processed/tranformations/' + folder_name
        df_aux = df[[column,column+"_transf"]].copy()
    else:
        #Creo la columna Factor
        df[column+"_factor"] = pd.factorize(df[column])[0]
        folder_path = '../data/processed/factories/' + folder_name
        df_aux = df[[column, column+"_factor"]].copy()
        
    #crea la carpeta si no existe
    os.makedirs(folder_path, exist_ok=True)

    for col in df_aux.columns:
        if pd.api.types.is_datetime64_any_dtype(df_aux[col]):
            df_aux[col] = df_aux[col].dt.strftime('%Y-%m-%d')
    
    # si tranformatios es true creamos renombramos parea que guarde en la carpeta tranformations de otra manera en factories
    if transformation:
        file_name = f'{column}_transformation_rules.json'
        df_to_save = df_aux[[column, column+"_transf"]]
        data_to_export = df_to_save.to_dict(orient = 'records')
    else:
      
        file_name = f'{column}_factory_rules.json'
        df_unique = df_aux[[column, column+"_factor"]].drop_duplicates()
        data_to_export = {row[column]: row[column+"_factor"] for _, row in df_unique.iterrows()}

    full_path = os.path.join(folder_path, file_name)
    with open(full_path, 'w') as f:
        json.dump(data_to_export, f, indent=4)
    #elimino la columna antigua antigua
    df.drop([column],  axis = 1,  inplace = True)
    print(f"Json guardado en: {full_path}")
def train_print_model(x_train, x_train_out, y_train, x_test, y_test, x_test_out,type_model="lg", class_weight=None, max_iter=10000,max_depth=7, umbral=0.5):
    # Preparo El Modelo dependiendo del valor de class_weight y el tipo de modelo seleccionado
    if type_model == "lg":
        model = LogisticRegression(class_weight=class_weight, max_iter=max_iter)
        model_out = LogisticRegression(class_weight=class_weight, max_iter=max_iter)
    elif type_model == "rf":
        # Para Random Forest usamos n_estimators. El class_weight funciona igual.
        model = RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=42,max_depth=max_depth)
        model_out = RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=42,max_depth=max_depth)

    # entrenamos (usando las variables globales x_train y x_train_out)
    model.fit(x_train, y_train)
    model_out.fit(x_train_out, y_train)
    
    # Obtener Probabilidades (en lugar de clases directas)
    # antes usaba predict pero para modificar el umbral necesito predict_proba
    probs = model.predict_proba(x_test)[:, 1]
    probs_out = model_out.predict_proba(x_test_out)[:, 1]

    # Aplico el umbral
    predictions = (probs >= umbral).astype(int)
    predictions_out = (probs_out >= umbral).astype(int)
    
    # Revisamos Las Métricas De Clasificacion como esta en el collab
    report = classification_report(y_test, predictions)
    report_out = classification_report(y_test, predictions_out)
    
    print(f"--- Reporte de Modelo: {type_model.upper()} (Umbral: {umbral}) ---")
    print("Reporte Sin Outliers:\n")
    print(report)
    print("Reporte Con Outliers:\n")
    print(report_out)
# No se si me servira en un futuro en los siguientes ejercicios pero cree una funcion que haga el split,los scalers, separe los outliers y guarde los limites en un json de una vez.
def train_prepare_test_data(df, target_col,folder_name, test_size=0.2, random_state=42, scaler_type=1, stratify = False):
 
    # Separamos X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # hacemos el split dependiendo del valor de stractify, stractify le dice al split que intente dividirlos distribuyendo bien los valores de una columna
    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Copiamos para crear los outliers y escalar sin destruir los originales
    x_train_out = x_train.copy()
    x_test_out = x_test.copy()
    
    # creamos un diccionario limpio para almacenar los limites mas adelante
    outlier_limits = {}
    # buscamos las variables numericas
    numeric_cols = x_train_out.select_dtypes(include=["number"]).columns
    
    #luego recorremos las columnas
    for col in numeric_cols:
        # calculamos el Rango Intercuartil como vimos en clases (IQR), que requiere del primer quartil y el tercero
        Q1 = x_train_out[col].quantile(0.25)
        Q3 = x_train_out[col].quantile(0.75)
        IQR = Q3 - Q1
        #nuestro limite bajo
        low = Q1 - 1.5 * IQR
        #nuestro limite alto
        high = Q3 + 1.5 * IQR
        
        # Guardamos los limites
        outlier_limits[col] = {"lower_bound": float(low), "upper_bound": float(high)}
        
        # suavizamos los outliers, baiscamente si es mayor high lo convierte en high y si es menor a low lo convierte a low
        x_train_out[col] = np.clip(x_train_out[col], low, high)
        x_test_out[col] = np.clip(x_test_out[col], low, high)
        
    #Creamos la carpeta si no existe
    path = f"../data/processed/outliers/{folder_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    # Guardardamos los limites en un JSON
    with open(f"{path}outlier_limits.json", "w") as f:
        json.dump(outlier_limits, f, indent=4)
        
    # Escalamos dependiendo de la variable scaler_type puede se StandardScaler, MinMaxScaler

    if scaler_type == 1:
        sc = StandardScaler()
         # Aprende Y Transforma El Train Normal
        x_train[numeric_cols] = sc.fit_transform(x_train[numeric_cols])
        
        # Solo Transforma El Resto
        x_test[numeric_cols] = sc.transform(x_test[numeric_cols])
        x_train_out[numeric_cols] = sc.transform(x_train_out[numeric_cols])
        x_test_out[numeric_cols] = sc.transform(x_test_out[numeric_cols])
        
    elif scaler_type == 2:
        mx = MinMaxScaler()
        # Aprende Y Transforma El Train Normal
        x_train[numeric_cols] = mx.fit_transform(x_train[numeric_cols])
        
        # Solo Transforma El Resto
        x_test[numeric_cols] = mx.transform(x_test[numeric_cols])
        x_train_out[numeric_cols] = mx.transform(x_train_out[numeric_cols])
        x_test_out[numeric_cols] = mx.transform(x_test_out[numeric_cols])
   
    return x_train, x_test, y_train, y_test, x_train_out, x_test_out
     
# meti la logica de entrenar al modelo porque creo que lo voy a necesitar mas de una ves, se le pasa los x_test el class_weight que es un diccionario con la proporcion de clases, 
# el max_iter son maximas iteraciones y el umbral para ajustar la sensibilidad del modelo
