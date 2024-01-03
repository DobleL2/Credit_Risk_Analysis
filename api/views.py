import os
import settings
import utils
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from middleware import model_predict

from model import ml_service as ml


router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/")
def index():
    return render_template('index.html')

# Route to process form data
@router.route('/process_form',methods=['POST'])
def process_form():
    user_input = request.form.get('user_input')
    
    # Pass the user input to your machine learning service function
    processed_data = ml.predict(user_input)
    
    return f"Processed data: {processed_data}"

# TODO Make data validation about the input of the user
@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : str
        Input image we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "prediction": str,
                "score": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "prediction" model predicted class as string.
        - "score" model confidence score for the predicted class as float.
    """
    # To correctly implement this endpoint you should:
    #   1. Check a file was sent and that file is an image
    #   2. Store the image to disk
    #   3. Send the file to be processed by the `model` service
    #   4. Update and return `rpse` dict with the corresponding values
    # If user sends an invalid request (e.g. no file provided) this endpoint
    # should return `rpse` dict with default values HTTP 400 Bad Request code
    rpse = {"success": False, "prediction": None, "score": None}

    if "file" in request.files and utils.allowed_file(request.files["file"].filename):
        file = request.files["file"]
        file_hash = utils.get_file_hash(file)
        dst_filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], file_hash)
        if not os.path.exists(dst_filepath):
            file.save(dst_filepath)
        prediction, score = model_predict(file_hash)
        rpse["success"] = True
        rpse["prediction"] = prediction
        rpse["score"] = score
        return jsonify(rpse)

    return jsonify(rpse), 400

