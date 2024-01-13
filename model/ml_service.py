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

# Preprocess
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

#DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
#os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

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
    class_name = 1
    pred_probability = 0.5
    
    return class_name, pred_probability

# In model_predict_from_form
def model_predict_from_form(form):
    # Extract values from form fields
    features = {
        "id_client": form.id_client,
        "payment_day": form.payment_day,
        "sex": form.sex,
        "marital_status": form.marital_status,
        "quant_dependants": form.quant_dependants,
        "nacionality": form.nacionality,
        "flag_residencial_phone": form.flag_residencial_phone,
        "residence_type": form.residence_type,
        "months_in_residence": form.months_in_residence,
        "personal_monthly_income": form.personal_monthly_income,
        "other_incomes": form.other_incomes,
        "has_any_card": form.has_any_card,
        "quant_banking_accounts": form.quant_banking_accounts,
        "personal_assets_value": form.personal_assets_value,
        "quant_cars": form.quant_cars,
        "flag_professional_phone": form.flag_professional_phone,
        "profession_code": form.profession_code,
        "occupation_type": form.occupation_type,
        "product": form.product,
        "age": form.age,
        "residencial_zip_3": form.residencial_zip_3,
    }

    # Call the function that uses these features for prediction
    prediction, score = 0,0.5#make_prediction(features)

    return prediction, score

def classify_process():
    while True:
        try:
            qname, mensaje = db.brpop(str(settings.REDIS_QUEUE))
            mensaje_json = mensaje.decode("utf8")
            data = json.loads(mensaje_json)
            
            class_name, pred_probability = predict(data["image_name"])

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
    