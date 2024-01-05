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

from forms import MyForm
from uuid import uuid4


router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/Form",methods=['GET'])
def index():
    return render_template('indexC.html')

@router.route("/Home",methods=['GET'])
def index():
    return render_template('index.html')

# Route to process form data
@router.route('/process_form', methods=['POST','GET'])
def process_form():
    if request.method == 'POST':
        # Retrieve data from the form
        form = MyForm()

        form.id_client = request.form['id_client',str(uuid4())]
        form.payment_day =  request.form['PAYMENT_DAY']
        form.sex = request.form['SEX']
        form.marital_status = request.form['MARITAL_STATUS']
        form.quant_dependants = request.form['QUANT_DEPENDANTS']
        form.nacionality = request.form['NACIONALITY']
        form.flag_residencial_phone = request.form['FLAG_RESIDENCIAL_PHONE']
        form.residence_type = request.form['RESIDENCE_TYPE']
        form.months_in_residence = request.form['MONTHS_IN_RESIDENCE']
        form.personal_monthly_income = request.form['PERSONAL_MONTHLY_INCOME']
        form.other_incomes = request.form['OTHER_INCOMES']
        form.has_any_card = request.form['HAS_ANY_CARD']
        form.quant_banking_accounts = request.form['QUANT_BANKING_ACCOUNTS']
        form.personal_assets_value = request.form['PERSONAL_ASSETS_VALUE']
        form.quant_cars = request.form['QUANT_CARS']
        form.flag_professional_phone = request.form['FLAG_PROFESSIONAL_PHONE']
        form.profession_code = request.form['PROFESSION_CODE']
        form.occupation_type = request.form['OCCUPATION_TYPE']
        form.product = request.form['PRODUCT']
        form.age = request.form['AGE']
        form.residencial_zip_3 = request.form['RESIDENCIAL_ZIP_3',000000]

        prediction, score = model_predict(form)
        
        context = {
            'prediction': prediction,
            "score": score,
        }
        
        # TODO Definir la template para la presentación de los datos 
        return render_template('index.html', context=context)  # Pass data to success template
    if request.method == 'GET':
        return render_template('index.html')

    return f"Processed data"