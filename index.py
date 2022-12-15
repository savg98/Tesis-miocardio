#Importar librerias a utilizar
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

#Definición de variable "COLUMNA"
COLUMNA=['age','sex','chest pain type','cholesterol','max heart rate','oldpeak','ST slope']
app=Flask(__name__)

#Ruta y definición de función "data_analist" para retornar al programa "data_analist" en html
@app.route('/', methods=["GET", "POST"])
def data_analist():
    #Método POST a utilizar para recolectar los datos del programa en HTML
    if request.method == "POST": 
        #Definición de la lista "data" y recolección de datos mediante el loop for y envío de la lista "data" a diferentes funciones
        data = []
        for val in COLUMNA:
            (data.append( float(request.form.get(val))))
        b, c = analisis(data)
        return render_template ('index.html', a = b, b=c) 
    return render_template ('index.html')

#-------------------------------------------------------------------------------------
#Definición de función analisis la cual 
def analisis(data):
    df= pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
    data_df= df[['age','sex','chest pain type','cholesterol','max heart rate','oldpeak','ST slope', 'target']]
    age = data_df["age"].values
    sex = data_df["sex"].values
    chest_pain_type = data_df["chest pain type"].values
    cholesterol = data_df["cholesterol"].values
    max_heart_rate = data_df["max heart rate"].values
    oldpeak = data_df["oldpeak"].values
    ST_slope = data_df["ST slope"].values
    target = data_df["target"].values

    X = np.array([age, sex, chest_pain_type, cholesterol, max_heart_rate, oldpeak, ST_slope]).T
    y = np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    from sklearn.ensemble import RandomForestRegressor

    bar = RandomForestRegressor (n_estimators = 800, max_depth = 10)
    bar.fit(X_train, y_train)
    y_pred = bar.predict([data])
    y_pred_porc = round(y_pred[0], 2) * 100
    message = ""
    message_2 = ""
    if y_pred < 0.6: 
        message = f"Tienes una probabilidad de padecer una afeccion cardiaca por complicaciones de miocardio que oscila el valor de {y_pred_porc}%."
    else:
        message = f"Tienes una probabilidad de padecer una afeccion cardiaca por complicaciones de miocardio que oscila el valor de {y_pred_porc}%."
        message_2 = f"Ya que posee una alta probabilidad de sufrir una insuficiencia cardiaca le recomendamos buscar asistencia medica."    
    return message, message_2
    
    

if __name__=='__main__':
    app.run(debug=True)


