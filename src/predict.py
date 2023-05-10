
# -*- coding: utf-8 -*-
"""
@author: Diego Bernales
"""

# Código de Scoring - Modelo de Clasificacion Marketing para Deposito a plazo fijo
############################################################################

import pandas as pd
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    
    # Cargar la tabla transformada
    df_scores = pd.read_csv(os.path.join(".\\data\\processed\\", filename))
    print(filename, ' cargado correctamente')
    
    # Leemos el modelo entrenado!
    package = '.\\models\\best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
       
    # Predecimos sobre el set de datos de implementacion con el modelo entrenado
    scores=model.predict(df_scores).reshape(-1,1)
    
    # Exportamos el resultado del modelo para cargarlo en el Feature Store o Data Mart de Modelos
    # Le asignamos nombres a las columnas
    df_scores = pd.DataFrame(scores, columns=['PREDICT'])
    # Exportamos la solucion
    df_scores.to_csv('.\\data\\scores\\final_score.csv')
    
    print(scores, 'exportado correctamente en la carpeta scores')
    

# Scoring desde el inicio
def main():
    score_model('df_bank_score_proc.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()
