# -*- coding: utf-8 -*-
"""
@author: Diego Bernales
"""

# Código de Evaluación - Modelo de Clasificacion Marketing para Deposito a plazo fijo
############################################################################

import pandas as pd
import pickle
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename):
    df_test = pd.read_csv(os.path.join(".\\data\\processed\\", filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '.\\models\\best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df_test.drop(['y'],axis=1)
    y_test = df_test[['y']]
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test,y_pred_test))


# Validación desde el inicio
def main():
    eval_model('df_bank_valid_proc.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()
