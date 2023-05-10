# -*- coding: utf-8 -*-
"""
@author: Diego Bernales
"""

# Código de Entrenamiento - Modelo de Clasificacion Marketing para Deposito a plazo fijo
############################################################################


import pandas as pd
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    # Cargar la tabla transformada
    df_train_proc = pd.read_csv(os.path.join(".\\data\\processed\\", filename))
    print(filename, ' cargado correctamente')
    
    X_train = df_train_proc.drop(['y'],axis=1)
    y_train = df_train_proc[['y']]
    
    # Entrenamos el modelo con toda la muestra
    # Entrenamos el modelo con toda la muestra
    xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                  gamma=0.05, gpu_id=-1, importance_type=None,
                  interaction_constraints='', learning_rate=0.1, max_delta_step=0,
                  max_depth=8, min_child_weight=1,
                  monotone_constraints='()', n_estimators=200, n_jobs=-1,
                  num_parallel_tree=1, predictor='auto', random_state=27,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27,
                  subsample=0.8, tree_method='exact', validate_parameters=1,
                  verbosity=None)
    
    xgb_model.fit(X_train, y_train)
    
    print('Modelo entrenado')
    
    # Guardamos el modelo entrenado para usarlo en produccion
    filename = '.\\models\\best_model.pkl'
    pickle.dump(xgb_model, open(filename, 'wb'))
    
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('df_bank_train_proc.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
