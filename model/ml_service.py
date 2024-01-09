import json
import os
import time

import numpy as np
import redis
import settings

db = redis.Redis(
    host = settings.REDIS_IP,
    port = settings.REDIS_PORT,
    db = settings.REDIS_DB_ID
)

from ..api.forms import MyForm

model = ""

def predict_from_form(form):
    """
    Run our ML model to get predictions based on the form data.

    Parameters
    ----------
    form : MyForm
        Form containing user input.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # Example: You might need to adjust how you extract data from the form fields
    # For demonstration, let's assume you are processing 'age' and 'sex' fields
    user_age = form.age.data
    user_sex = form.sex.data

    # Process the form data as needed for your model
    # Example: Create dummy features based on the form data
    features = [user_age, user_sex]  # Adjust this as per your model's requirements

    # Pass features to your ML model for prediction
    prediction = model.predict(features)  # Replace 'features' with your processed data

    # Process the prediction result (replace this with your actual prediction logic)
    # Example: Assume the prediction output is a class name and a probability
    class_name = "Predicted Class"
    pred_probability = 0.85  # Adjust this with your model's output

    return class_name, pred_probability

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load form data from the corresponding queue based on the job ID
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data (form data in this case)
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO
        
        queue_name, msg = db.brpop(settings.REDIS_QUEUE)
        
        msg = json.loads(msg)
        
        # Extract form data from the job
        form_data = msg['user_data']
        
        # Predict using the ML model with the form data
        class_name, pred_probability = predict_from_form(MyForm(data=form_data))
        
        result_dictionary = {
            "prediction": class_name,
            "score": float(pred_probability)
        }
        
        job_id = msg["id"]
        
        db.set(job_id, json.dumps(result_dictionary))
        
        time.sleep(settings.SERVER_SLEEP)