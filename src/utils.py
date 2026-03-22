from dotenv import load_dotenv
from sqlalchemy import create_engine

import pandas as pd
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report,r2_score,mean_squared_error,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder 
from collections import namedtuple
# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


""" ----CREATE_FACTOR_TRANSF_AND_JSON----
    Utilidad: Sirve para factorizar o transformar columnas y guardar las reglas en un json.
    Devuelve: Nada, modifica el dataframe directamente (inplace).
    Parametros: 
        * column: La columna del dataframe a procesar.
        * df: El dataframe a modificar.
        * folder_name: Nombre de la carpeta para el json.
        * target_column: Nombre de la columna objetivo para mantenerla al final.
        * transformation: Si es True aplica funcion, si es False factoriza.
        * transformation_func: Funcion lambda o def para transformar los datos.
        * label_encoder: Si es "le" usa LabelEncoder, si es "pdf" usa factorize y si es "pdd" usa pandas dummies.
"""
def create_factor_transf_and_json(column, df,folder_name,target_column=None, transformation = False,transformation_func = lambda x: x, label_encoder="le"):
    file_name = "" 
    # Depende del valor de "transformation" toma un camino u otro
    if transformation:
        # Creamos una copia temporal
        df_aux = df[[column]].copy()
        # Ahora se le pasa a la funcion "transformation_func" con "apply" y creamos las columna "..._transf"
        df[column+"_transf"] = df_aux[column].apply(transformation_func)
        folder_path = '../data/processed/tranformations/' + folder_name
        df_aux = df[[column,column+"_transf"]].copy()
    else:
        #Creo la Columna "..._Factor"
        folder_path = '../data/processed/factories/' + folder_name
        if label_encoder == "le":
            le = LabelEncoder()
            df[column + "_factor"] = le.fit_transform(df[column])
            df_aux = df[[column, column+"_factor"]].copy()
        elif label_encoder == "pdf":
            df[column+"_factor"] = pd.factorize(df[column])[0]
            df_aux = df[[column, column+"_factor"]].copy()
        elif label_encoder == "pdd":
            # Guardamos los dummies en un DF temporal
            dummies = pd.get_dummies(df[column], prefix=column)
            
    #crea la carpeta si no existe
    os.makedirs(folder_path, exist_ok=True)
    
    # Esto prepara algunas columnas de tipo date para el json
    if 'df_aux' in locals():
        for col in df_aux.columns:
            if pd.api.types.is_datetime64_any_dtype(df_aux[col]):
                df_aux[col] = df_aux[col].dt.strftime('%Y-%m-%d')
    
    # si "transformation" es true creamos el archivo y que guarde en la carpeta tranformations de otra manera en factories
    if transformation:
        file_name = f'{column}_transformation_rules.json'
        df_to_save = df_aux[[column, column+"_transf"]]
        data_to_export = df_to_save.to_dict(orient = 'records')
    elif label_encoder == "pdd":
        for col in dummies.columns:
            df[f"{col}_dummy"] = dummies[col]
        file_name = f'{column}_dummies_rules.json'
        data_to_export = {"original_column": column, "dummy_columns": list(dummies.columns)}
    else:
        file_name = f'{column}_factory_rules.json'
        df_unique = df_aux[[column, column+"_factor"]].drop_duplicates()
        data_to_export = {row[column]: row[column+"_factor"] for _, row in df_unique.iterrows()}

    #Contruimos La ruta final de guardado
    full_path = os.path.join(folder_path, file_name)
    with open(full_path, 'w') as f:
        json.dump(data_to_export, f, indent=4)
        
    #Elimino la columna antigua
    df.drop([column],  axis = 1,  inplace = True)
    
    # Reordenar columnas para asegurar que el target esté al final
    if target_column is not None and target_column in df.columns:
        cols = [c for c in df.columns if c != target_column] + [target_column]
        df.__init__(df.reindex(columns=cols))
        
    print(f"Json guardado en: {full_path}")




""" ----TRAIN_PREPARE_TEST_DATA----
    Utilidad: Sirve para CREAR datos de entrenamiento y test para luego pasarselos al modelo.
    Devuelve: Objeto ModelPrepareResults con datos divididos, escalados y filtrados.
    Parametros: 
        * df: el dataframe original.
        * target_col: Columna objetivo (y).
        * folder_name: Nombre de la carpeta para guardar los limites de outliers.
        * test_size: Proporcion de la division de datos (defecto 0.2).
        * random_state: Semilla de aleatoriedad.
        * scaler_type: 0: sin escalado, 1: StandardScaler, 2: MinMaxScaler.
        * stratify: Si es True, mantiene la proporcion de clases en el split.
"""
""" ModelPrepareResults es una tupla especial para evitar recibir muchos parametro cuando se llame la funcion, por lo general seria asi :
         x_train_out,x_test_out,y_train_out...= prepare_test_data(...)
    
    Ahora:
         ptd = prepare_test_data(...)
        
    Y Se accede con ptd.x_train_out al valor

"""
ModelPrepareResults = namedtuple('ModelPrepareResults', ["x_train_out","x_test_out","y_train_out","y_test_out","x_train_no_out","x_test_no_out","y_train_no_out","y_test_no_out"])
def prepare_test_data(df, target_col,folder_name, test_size=0.2, random_state=42, scaler_type=1, stratify = False):
    #Convertimos Trues a 1 y falses a 0
    df = df.replace({True: 1, False: 0})
    
    # Separamos X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if stratify:
        x_train_out, x_test_out, y_train_out, y_test_out = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    else:
        x_train_out, x_test_out, y_train_out, y_test_out = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Copiamos para crear los outliers y escalar sin destruir los originales
    x_train_no_out = x_train_out.copy()
    x_test_no_out = x_test_out.copy()
    y_train_no_out = y_train_out.copy()
    y_test_no_out = y_test_out.copy()
    # creamos un diccionario limpio para almacenar los limites mas adelante
    outlier_limits = {}

    # Filtramos para que NO incluya columnas que contengan etiquetas de dummies,factor o transf
    numeric_cols = [
        col for col in x_train_no_out.select_dtypes(include=["number"]).columns 
        if "_factor" not in col and "_dummy" not in col and "_transf" not in col
    ]
    
    #luego recorremos las columnas
    for col in numeric_cols:
        # calculamos el Rango Intercuartil como vimos en clases (IQR), que requiere del primer quartil y el tercero
        Q1 = x_train_no_out[col].quantile(0.25)
        Q3 = x_train_no_out[col].quantile(0.75)
        IQR = Q3 - Q1
        #nuestro limite bajo
        low = Q1 - 1.5 * IQR
        #nuestro limite alto
        high = Q3 + 1.5 * IQR
        # Guardamos los limites
        outlier_limits[col] = {"lower_bound": float(low), "upper_bound": float(high)}
        # Suavizamos los outliers, basicamente si es mayor a high lo convierte en high y si es menor a low lo convierte en low
        x_train_no_out[col] = np.clip(x_train_no_out[col], low, high)
        x_test_no_out[col] = np.clip(x_test_no_out[col], low, high)
        
    #Creamos la carpeta si no existe
    path = f"../data/processed/outliers/{folder_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    # Guardardamos los limites en un JSON
    with open(f"{path}outlier_limits.json", "w") as f:
        json.dump(outlier_limits, f, indent=4)
        
    # Escalamos dependiendo de la variable scaler_type.
    if scaler_type == 1:
        sc = StandardScaler()
        x_train_out[numeric_cols] = sc.fit_transform(x_train_out[numeric_cols])
        x_test_out[numeric_cols] = sc.transform(x_test_out[numeric_cols])
        x_train_no_out[numeric_cols] = sc.transform(x_train_no_out[numeric_cols])
        x_test_no_out[numeric_cols] = sc.transform(x_test_no_out[numeric_cols])
        
    elif scaler_type == 2:
        mx = MinMaxScaler()
        x_train_out[numeric_cols] = mx.fit_transform(x_train_out[numeric_cols])
        x_test_out[numeric_cols] = mx.transform(x_test_out[numeric_cols])
        x_train_no_out[numeric_cols] = mx.transform(x_train_no_out[numeric_cols])
        x_test_no_out[numeric_cols] = mx.transform(x_test_no_out[numeric_cols])
   
    # Devolvemos los datos de entrenamiento con y sin outliers
    return ModelPrepareResults(x_train_out, x_test_out, y_train_out, y_test_out, x_train_no_out, x_test_no_out,y_train_no_out, y_test_no_out)
     
     
     
     
     
""" ----TRAIN_PRINT_MODEL----
    Utilidad: Inicializa y entrena modelos de prediccion (clasificacion o regresion).
    Devuelve: Objeto ModelResults con metricas, reportes y probabilidades segun el modelo.
    Parametros: 
        * ptd: Objeto ModelPrepareResults con datos de entrenamiento y test.
        * type_model: tipo del modelo a usar (lr, lg, dt, rf).
        * class_weight: Distribucion del peso de las clases.
        * umbral: Umbral de decision para clasificacion (por defecto 0.5).
        * max_iter: Maximo de iteraciones para LogisticRegression.
        * max_depth: Profundidad maxima para arboles (DT y RF).
        * calibrate_cv: Numero de cortes para CalibratedClassifierCV.
"""
""" ModelResults es una tupla especial para evitar recibir muchos parametro cuando se llame la funcion, por lo general seria asi :
         ptd, type_model="lg", class_weight=None...= train_print_model(...)
    
    Ahora:
         tpm = train_print_model(...)
        
    Y Se accede con tpm.report_out al valor

"""
fields=["report_out","report_no_out","accuracy_out","accuracy_no_out","confusion_matrix_out","confusion_matrix_no_out","probs_out","probs_no_out","preds_out","preds_no_out","r2_out","r2_no_out","mse_out","mse_no_out"]
ModelResults = namedtuple('ModelResults', fields, defaults = (None,) * len(fields))

def train_print_model(ptd, type_model="lg", class_weight=None, umbral=0.5, max_iter=10000,max_depth=7,random_state=42, calibrate_cv=None):
    
    if ptd is not None and isinstance(ptd, ModelPrepareResults):
        x_train_out = ptd.x_train_out
        x_test_out = ptd.x_test_out
        y_train_out = ptd.y_train_out
        y_test_out = ptd.y_test_out
        x_train_no_out = ptd.x_train_no_out
        x_test_no_out = ptd.x_test_no_out
        y_train_no_out = ptd.y_train_no_out
        y_test_no_out = ptd.y_test_no_out
    else:
        return "Sin Datos de entrenamiento"
    # Preparo El Modelo dependiendo del valor de class_weight y el tipo de modelo seleccionado
    if type_model == "lg":
        title_model="LogisticRegression"
        model_out = LogisticRegression(class_weight=class_weight, max_iter=max_iter)
        model_no_out = LogisticRegression(class_weight=class_weight, max_iter=max_iter)
    elif type_model == "dt":
        title_model="DecisionTreeClassifier"
        model_out = DecisionTreeClassifier(class_weight= class_weight,random_state=random_state, max_depth=max_depth)
        model_no_out = DecisionTreeClassifier(class_weight= class_weight,random_state=random_state, max_depth=max_depth)
        if calibrate_cv is not None:
            model_out = CalibratedClassifierCV(model_out, method='sigmoid', cv=calibrate_cv)
            model_no_out = CalibratedClassifierCV(model_no_out, method='sigmoid', cv=calibrate_cv)
    elif type_model == "rf":
        title_model="RandomForestClassifier"
        model_out = RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=random_state,max_depth=max_depth)
        model_no_out = RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=random_state,max_depth=max_depth)
    elif type_model == "lr":
        title_model="LinearRegression"  # linear regression
        model_out = LinearRegression()
        model_no_out = LinearRegression()
    # entrenamos usando x_train_out y x_train_no_out
    model_out.fit(x_train_out, y_train_out)
    model_no_out.fit(x_train_no_out, y_train_no_out)
    
    if type_model != "lr":
        # Obtener Probabilidades (en lugar de clases directas)
        # antes usaba predict pero para modificar el umbral necesito predict_proba
        probs_out = model_out.predict_proba(x_test_out)[:, 1]
        probs_no_out = model_no_out.predict_proba(x_test_no_out)[:, 1]

        # Aplico el umbral
        predictions_out = (probs_out >= umbral).astype(int)
        predictions_no_out = (probs_no_out >= umbral).astype(int)

        # Revisamos Las Metricas De Clasificacion como esta en el collab
        raw_report_out = classification_report(y_test_out, predictions_out)
        raw_report_no_out = classification_report(y_test_no_out, predictions_no_out)
        
        # Agregamos el titulo con f-strings
        report_out = f"--- REPORTE CON OUTLIERS ({title_model}) ---\n{raw_report_out}"
        report_no_out = f"--- REPORTE SIN OUTLIERS ({title_model}) ---\n{raw_report_no_out}"
        if type_model == "dt":
             accuracy_out = accuracy_score(y_test_out,predictions_out)
             accuracy_no_out = accuracy_score(y_test_no_out,predictions_no_out)
             confusion_matrix_out = confusion_matrix(y_test_out,predictions_out)
             confusion_matrix_no_out = confusion_matrix(y_test_no_out,predictions_no_out)
             return ModelResults(report_out, report_no_out,accuracy_out, accuracy_no_out, confusion_matrix_out, confusion_matrix_no_out, probs_out, probs_no_out)
            
        return ModelResults(report_out, report_no_out, probs_out, probs_no_out)
    else:
        #predicciones
        preds_out = model_out.predict(x_test_out)
        preds_no_out = model_no_out.predict(x_test_no_out)
        # error cuadratico medio 
        mse_out = mean_squared_error(y_test_out, preds_out)
        mse_no_out = mean_squared_error(y_test_no_out, preds_no_out)
        # y r2
        r2_out = r2_score(y_test_out, preds_out)
        r2_no_out = r2_score(y_test_no_out, preds_no_out)
        return ModelResults(preds_out = preds_out, preds_no_out = preds_no_out, r2_out = r2_out, r2_no_out = r2_no_out, mse_out = mse_out, mse_no_out = mse_no_out)





