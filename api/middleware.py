import json
import time
from uuid import uuid4

import redis
import settings
from forms import MyForm

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host = settings.REDIS_IP,
    port = settings.REDIS_PORT,
    db = settings.REDIS_DB_ID
)

def model_predict_from_form(user_data):
    """
    Receives form data and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    form : YourForm
        Form containing user input.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    prediction = None
    score = None


    # Assign an unique ID for this job and add it to the queue.
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "user_data": user_data,
    }

    # Send the job to the model service using Redis
    msg_str = json.dumps(job_data)
    db.lpush(settings.REDIS_QUEUE, msg_str)
    
    # Loop until we received the response from our ML model
    while True:
        output = db.get(job_id)

        if output is not None:
            output = json.loads(output.decode("utf-8"))
            prediction = output["prediction"]
            score = output["score"]

            db.delete(job_id)
            break

        time.sleep(settings.API_SLEEP)

    return prediction, score

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
        queue_name, msg = db.brpop(settings.REDIS_QUEUE)
        
        msg = json.loads(msg)
        
        # Extract form data
        form_data = msg['user_data']
        
        # Pass form data to your model prediction function
        class_name, pred_probability = model_predict_from_form(form_data)
        
        result_dictionary = {
            "prediction": class_name,
            "score": float(pred_probability)
        }
        
        job_id = msg["id"]
        
        db.set(job_id, json.dumps(result_dictionary))
        
        time.sleep(settings.SERVER_SLEEP)