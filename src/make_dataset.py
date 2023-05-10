# -*- coding: utf-8 -*-
"""
@author: Diego Bernales
"""

# Script de Preparación de Datos
###################################

import pandas as pd
from imblearn.combine import SMOTEENN
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join(".\\data\\raw\\", filename))
    print(filename, ' cargado correctamente')
    return df

#df_train=pd.read_csv(".\\data\\raw\\df_bank_train.csv")

# Realizamos la transformación de datos
def data_preparation(df):
    to_drop = ['default','loan','marital','education','campaign','day','previous']
    df.drop(columns=to_drop, axis=1, inplace = True)
    
    numeric_columns = set(df.select_dtypes(include=['number']).columns)
    non_numeric_columns = set(df.columns) - numeric_columns
    numeric_columns = list(numeric_columns)
    non_numeric_columns= list(non_numeric_columns)
    
    for col in df.select_dtypes(['object']):
        df[col],unique = df[col].factorize(sort= True)
        
    def limites(columna):
        c = [.25,.75]
        q1,q3=columna.quantile(c)
        ## calculamos
        lim_inf = q3 - 1.5* (q3-q1)
        lim_sup = q3 + 1.5* (q3-q1)
        return(lim_inf ,lim_sup)
    
    for col in df[numeric_columns]:
        if col != 'pdays':
            a,b = limites(df[col])
            #recorte
            df.loc[df[col]<a,col] = a 
            df.loc[df[col]>b,col] = b 
        else:
            df[col]= df[col]
            
    df["pdays"]= df["pdays"].replace([-1],0)
    
    ## Estandarizacion
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join(".\\data\\processed\\", filename), index = False)
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df_train = read_file_csv('df_bank_train.csv')
    df_train = data_preparation(df_train)
    ## Balanceo de clases
    X = df_train.drop('y',axis=1)
    y = df_train['y']

    SMT=SMOTEENN() 
    x_bal, y_bal= SMT.fit_resample(X,y) 

    df_train_proc=pd.concat([x_bal, y_bal], axis=1) 
    data_exporting(df_train_proc,'df_bank_train_proc.csv')
    
    # Matriz de Validación
    df_valid = read_file_csv('df_bank_valid.csv')
    df_valid = data_preparation(df_valid)
    data_exporting(df_valid,'df_bank_valid_proc.csv')
    
    # Matriz de Scoring
    df_score = read_file_csv('df_bank_score.csv')
    df_score = data_preparation(df_score)
    data_exporting(df_score,'df_bank_score_proc.csv')

    
if __name__ == "__main__":
    main()
