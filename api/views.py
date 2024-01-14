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
from middleware import model_predict_from_form

from forms import MyForm
from uuid import uuid4
import logging


router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/",methods=['GET'])
def index():
    return render_template('index.html')

# Route to process form data
@router.route('/process_form', methods=['POST','GET'])
def process_form():
    if request.method == 'POST':
        
        test = {'PAYMENT_DAY': [14], 'SEX': ['M'], 'MARITAL_STATUS': [0.32316679991959296], 'QUANT_DEPENDANTS': [1.9077678811444132], 'NACIONALITY': [0.22034757035662245], 'FLAG_RESIDENCIAL_PHONE': [1], 
                'RESIDENCE_TYPE': [1.7495547046722986], 'MONTHS_IN_RESIDENCE': [48.20525143724116], 'PERSONAL_MONTHLY_INCOME': [896524.4425945266], 'OTHER_INCOMES': [317.9287421872096], 
                'QUANT_BANKING_ACCOUNTS': [0.22551415109553608], 'PERSONAL_ASSETS_VALUE': [8712.210598266734], 'QUANT_CARS': [1.3497474993388459], 'FLAG_PROFESSIONAL_PHONE': [1],
                'PROFESSION_CODE': [5.172599117336276], 'OCCUPATION_TYPE': [3.7989584602272704], 'PRODUCT': [1.4676470986579082], 'AGE': [36.037719317278004], 'RESIDENCIAL_ZIP_3': [250.58099041358662], 'HAS_ANY_CARD': [0]}
        
        SEX = request.form.get('SEX','M')
        PAYMENT_DAY =  request.form.get('PAYMENT_DAY',[14])
        MARITAL_STATUS = request.form.get('MARITAL_STATUS',[0.32316679991959296])
        QUANT_DEPENDANTS = request.form.get('QUANT_DEPENDANTS',[1.9077678811444132])
        NACIONALITY = request.form.get('NACIONALITY',[0.22034757035662245])
        FLAG_RESIDENCIAL_PHONE = request.form.get('FLAG_RESIDENCIAL_PHONE',[1])
        RESIDENCE_TYPE = request.form.get('RESIDENCE_TYPE',[1.7495547046722986])
        MONTHS_IN_RESIDENCE = request.form.get('MONTHS_IN_RESIDENCE',[48.20525143724116])
        PERSONAL_MONTHLY_INCOME = request.form.get('PERSONAL_MONTHLY_INCOME',[896524.4425945266])
        OTHER_INCOMES = request.form.get('OTHER_INCOMES',[317.9287421872096])
        HAS_ANY_CARD = request.form.get('HAS_ANY_CARD',[0])
        QUANT_BANKING_ACCOUNTS = request.form.get('QUANT_BANKING_ACCOUNTS',[0.22551415109553608])
        PERSONAL_ASSETS_VALUE = request.form.get('PERSONAL_ASSETS_VALUE',[8712.210598266734])
        QUANT_CARS = request.form.get('QUANT_CARS',[1.3497474993388459])
        FLAG_PROFESSIONAL_PHONE = request.form.get('FLAG_PROFESSIONAL_PHONE',[1])
        PROFESSION_CODE = request.form.get('PROFESSION_CODE',[5.172599117336276])
        OCCUPATION_TYPE = request.form.get('OCCUPATION_TYPE',[3.7989584602272704])
        PRODUCT = request.form.get('PRODUCT',[1.4676470986579082])
        AGE = request.form.get('AGE',[36.037719317278004])
        RESIDENCIAL_ZIP_3 = request.form.get('RESIDENCIAL_ZIP_3',[250.58099041358662])
        
        data =  {
                'SEX': SEX,
                'PAYMENT_DAY' : PAYMENT_DAY, 
                'MARITAL_STATUS' : MARITAL_STATUS,   
                'QUANT_DEPENDANTS' : QUANT_DEPENDANTS,   
                'NACIONALITY' : NACIONALITY, 
                'FLAG_RESIDENCIAL_PHONE' : FLAG_RESIDENCIAL_PHONE,   
                'RESIDENCE_TYPE' : RESIDENCE_TYPE,   
                'MONTHS_IN_RESIDENCE' : MONTHS_IN_RESIDENCE, 
                'PERSONAL_MONTHLY_INCOME' : PERSONAL_MONTHLY_INCOME, 
                'OTHER_INCOMES' : OTHER_INCOMES, 
                'HAS_ANY_CARD' : HAS_ANY_CARD,   
                'QUANT_BANKING_ACCOUNTS' : QUANT_BANKING_ACCOUNTS,   
                'PERSONAL_ASSETS_VALUE' : PERSONAL_ASSETS_VALUE, 
                'QUANT_CARS' : QUANT_CARS,   
                'FLAG_PROFESSIONAL_PHONE' : FLAG_PROFESSIONAL_PHONE, 
                'PROFESSION_CODE' : PROFESSION_CODE, 
                'OCCUPATION_TYPE' : OCCUPATION_TYPE, 
                'PRODUCT' : PRODUCT, 
                'AGE' : AGE, 
                'RESIDENCIAL_ZIP_3' : RESIDENCIAL_ZIP_3,       
        }
        # TODO (Marco: Add Ml_Service)
        prediction, score = model_predict_from_form(data)
        
        context = {
            'prediction': prediction,
            "score": score,
            'data': {
                'SEX': SEX,
                'PAYMENT_DAY' : PAYMENT_DAY, 
                'MARITAL_STATUS' : MARITAL_STATUS,   
                'QUANT_DEPENDANTS' : QUANT_DEPENDANTS,   
                'NACIONALITY' : NACIONALITY, 
                'FLAG_RESIDENCIAL_PHONE' : FLAG_RESIDENCIAL_PHONE,   
                'RESIDENCE_TYPE' : RESIDENCE_TYPE,   
                'MONTHS_IN_RESIDENCE' : MONTHS_IN_RESIDENCE, 
                'PERSONAL_MONTHLY_INCOME' : PERSONAL_MONTHLY_INCOME, 
                'OTHER_INCOMES' : OTHER_INCOMES, 
                'HAS_ANY_CARD' : HAS_ANY_CARD,   
                'QUANT_BANKING_ACCOUNTS' : QUANT_BANKING_ACCOUNTS,   
                'PERSONAL_ASSETS_VALUE' : PERSONAL_ASSETS_VALUE, 
                'QUANT_CARS' : QUANT_CARS,   
                'FLAG_PROFESSIONAL_PHONE' : FLAG_PROFESSIONAL_PHONE, 
                'PROFESSION_CODE' : PROFESSION_CODE, 
                'OCCUPATION_TYPE' : OCCUPATION_TYPE, 
                'PRODUCT' : PRODUCT, 
                'AGE' : AGE, 
                'RESIDENCIAL_ZIP_3' : RESIDENCIAL_ZIP_3,       
            },
        }
        
        return render_template('index.html',target_tab_id=4,context=context)
    
    if request.method == 'GET':
        print('HOLA')
        logging.info("This is an info message")
        return render_template('index.html',target_tab_id=4)

    return f"Processed data"

         
         
         
         
         
         

         
         
         
         
         
         
         
         
         
         
         