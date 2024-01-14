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
    try:            
        # Load the saved pipeline and model
        with open('/src/pipeline_train.pkl', 'rb') as pipeline_file:
            loaded_pipeline = cloudpickle.load(pipeline_file)

        with open('/src/model_p.pkl', 'rb') as model_file:
            loaded_model = cloudpickle.load(model_file)

        # Transform the new data using the loaded pipeline
        value_test = loaded_pipeline.transform(random_clients_df)

        # Make predictions
        prediction = loaded_model.predict(value_test)
        print("Predicted value for new data:", prediction)

        prediction_proba = loaded_model.predict_proba(value_test)
        positive_class_proba = prediction_proba[:, 1]
        print("Probability of the positive class for new data:", positive_class_proba)

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
    