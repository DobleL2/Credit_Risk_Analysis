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

from joblib import load

#DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
#os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)




def predict(random_clients_df):
    test = {'SEX': {0: 0.0},
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
    # TODO (Marco: Add Ml_Service)
    loaded_model = load('/src/joblib_model.joblib')
    score = loaded_model.predict_proba(test)[0]
    prediction = loaded_model.predict(test)
    
    return prediction,score
    dict_features = {'PAYMENT_DAY': [14], 'SEX': ['M'], 'MARITAL_STATUS': [0.32316679991959296], 'QUANT_DEPENDANTS': [1.9077678811444132], 'NACIONALITY': [0.22034757035662245], 'FLAG_RESIDENCIAL_PHONE': [1], 'RESIDENCE_TYPE': [1.7495547046722986], 'MONTHS_IN_RESIDENCE': [48.20525143724116], 'PERSONAL_MONTHLY_INCOME': [896524.4425945266], 'OTHER_INCOMES': [317.9287421872096], 'QUANT_BANKING_ACCOUNTS': [0.22551415109553608], 'PERSONAL_ASSETS_VALUE': [8712.210598266734], 'QUANT_CARS': [1.3497474993388459], 'FLAG_PROFESSIONAL_PHONE': [1], 'PROFESSION_CODE': [5.172599117336276], 'OCCUPATION_TYPE': [3.7989584602272704], 'PRODUCT': [1.4676470986579082], 'AGE': [36.037719317278004], 'RESIDENCIAL_ZIP_3': [250.58099041358662], 'HAS_ANY_CARD': [0]}
    df = pd.DataFrame(dict_features)

    model = load_model('dt_pipeline')
    pred = predict_model(model, raw_score = True, data = df)

    class_name = pred['prediction_label']
    pred_probability = pred['prediction_score_0']
    return class_name,pred_probability
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
    