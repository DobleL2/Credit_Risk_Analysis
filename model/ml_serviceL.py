import json
import os
import time
import boto3
import numpy as np
import redis
import settings
from PIL import Image
import lightgbm as lgb
import pandas as pd
import zipfile
from pathlib import Path
import cloudpickle
from utils import *
import logging
import pickle
import joblib

# Preprocess
#from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
#from sklearn.impute import SimpleImputer
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.model_selection import train_test_split

db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

model_p = lgb.LGBMClassifier(
    boosting_type='gbdt',
    class_weight=None,
    colsample_bytree=1.0,
    importance_type='split',
    learning_rate=0.1,
    max_depth=-1,
    min_child_samples=20,
    min_child_weight=0.001,
    min_split_gain=0.0,
    n_estimators=100,
    n_jobs=-1,
    num_leaves=31,
    objective=None,
    random_state=42,
    reg_alpha=0.0,
    reg_lambda=0.0,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=0
)

def predict(random_clients_df):
    class_name=0
    pred_probability =0.1
    
    test = {'SEX': {0: 1.0},
 'PAYMENT_DAY': {0: 0.1678336140240951},
 'MARITAL_STATUS': {0: -1.3836195852815556},
 'QUANT_DEPENDANTS': {0: 1.0699798986808822},
 'NACIONALITY': {0: -3.6819012378973213},
 'FLAG_RESIDENCIAL_PHONE': {0: 0.4424553317066655},
 'RESIDENCE_TYPE': {0: 0.5789848184067318},
 'MONTHS_IN_RESIDENCE': {0: 3.8652296397487595},
 'PERSONAL_MONTHLY_INCOME': {0: 123.375260485318},
 'OTHER_INCOMES': {0: 0.28408418132750435},
 'QUANT_BANKING_ACCOUNTS': {0: -0.27650491818687867},
 'PERSONAL_ASSETS_VALUE': {0: 0.13422886682439855},
 'QUANT_CARS': {0: 2.1476343323409104},
 'FLAG_PROFESSIONAL_PHONE': {0: 1.641694685238739},
 'PROFESSION_CODE': {0: -1.8845897028550707},
 'OCCUPATION_TYPE': {0: 0.8344895932385018},
 'PRODUCT': {0: 0.1914533582934268},
 'AGE': {0: -0.4829082452779595},
 'RESIDENCIAL_ZIP_3': {0: -1.4689276009237735},
 'HAS_ANY_CARD': {0: -0.442212577389577}}
    
    test = pd.DataFrame(test)
    # Load the pickled model
    return os.getcwd(), os.getcwd()
    with open('/src/trained_model.pkl', 'rb') as model_file:
        loaded_model = joblib.load(model_file)
    
    # Example input data for prediction (replace this with your actual data)

    # Make predictions using the loaded model
    predictions = os.getcwd()#loaded_model.predict(test)#type(loaded_model)#
    return os.getcwd(), os.getcwd()
    proba = loaded_model.predict_proba(test)[0]
    return proba, proba
    predictions = loaded_model.predict(test)

    # Print the predictions
    print("Predictions:", predictions)
    return proba, predictions
    
    
    names = pd.read_csv('/src/dataset/PAKDD2010_VariablesList.csv')

    file_path_3 = '/src/dataset/PAKDD2010_Modeling_Data.txt'
    mod = pd.read_csv(file_path_3, delimiter='\t', header=None, encoding='latin1')

    
    # Clean Data
    df_clean = clean_data_predict(mod,names)
    X_train, X_test, y_train, y_test = train_test_split(df_clean.drop(['TARGET'],axis = 1), df_clean["TARGET"], test_size=0.2, random_state=42)
    
    # Preprocess data
    train_data = pd.read_csv('/src/dataset/train_data.csv')
    test_data = pd.read_csv('/src/dataset/test_data.csv')
    #train_data, test_data = preprocess_data(X_train, X_test)
    
    model_p.fit(train_data, y_train)
    return 12,98
    
    train, value_test = preprocess_data(X_train,random_clients_df)
    
      #---    
    try:            
        # Make predictions
        prediction = model_p.predict(value_test)
        print("Predicted value for new data:", prediction)
        prediction_proba = model_p.predict_proba(value_test)
        positive_class_proba = prediction_proba[:, 1]
        print("Probability of positive class for new data:", positive_class_proba)

        class_name = prediction
        pred_probability = float(np.max(prediction_proba))
        
    except Exception as e:
        print(f"Error processing image: {e}")

    return class_name, pred_probability

def classify_process():

    while True:
        try:
            qname, mensaje = db.brpop(str(settings.REDIS_QUEUE))
            mensaje_json = mensaje.decode("utf8")
            data = json.loads(mensaje_json)
            
            class_name, pred_probability = predict(data["user_data"])

            prediction_result = {
                "prediction": class_name,
                "score": float(pred_probability),
            }

            result_key = data["id"]
            db.set(result_key, json.dumps(prediction_result))

        except Exception as e:
            print(f"Error in classify_process: {e}")

        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    print("Launching ML service...")
    classify_process()
    