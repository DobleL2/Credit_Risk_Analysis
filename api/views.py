import os
import settings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def _init_(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

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


from uuid import uuid4
import logging
import joblib
from joblib import dump,load


router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/",methods=['GET'])
def index():
    return render_template('index.html')

# Route to process form data
@router.route('/process_form', methods=['POST','GET'])
def process_form():
    if request.method == 'POST':
        
        SEX = request.form.get('SEX','M')
        PAYMENT_DAY =  request.form.get('PAYMENT_DAY',14)
        MARITAL_STATUS = request.form.get('MARITAL_STATUS',0.32316679991959296)
        QUANT_DEPENDANTS = request.form.get('QUANT_DEPENDANTS',1.9077678811444132)
        NACIONALITY = request.form.get('NACIONALITY',0.22034757035662245)
        FLAG_RESIDENCIAL_PHONE = request.form.get('FLAG_RESIDENCIAL_PHONE',1)
        RESIDENCE_TYPE = request.form.get('RESIDENCE_TYPE',1.7495547046722986)
        MONTHS_IN_RESIDENCE = request.form.get('MONTHS_IN_RESIDENCE',48.20525143724116)
        PERSONAL_MONTHLY_INCOME = request.form.get('PERSONAL_MONTHLY_INCOME',896524.4425945266)
        OTHER_INCOMES = request.form.get('OTHER_INCOMES',317.9287421872096)
        HAS_ANY_CARD = request.form.get('HAS_ANY_CARD',0)
        QUANT_BANKING_ACCOUNTS = request.form.get('QUANT_BANKING_ACCOUNTS',0.22551415109553608)
        PERSONAL_ASSETS_VALUE = request.form.get('PERSONAL_ASSETS_VALUE',8712.210598266734)
        QUANT_CARS = request.form.get('QUANT_CARS',1.3497474993388459)
        FLAG_PROFESSIONAL_PHONE = request.form.get('FLAG_PROFESSIONAL_PHONE',1)
        PROFESSION_CODE = request.form.get('PROFESSION_CODE',5.172599117336276)
        OCCUPATION_TYPE = request.form.get('OCCUPATION_TYPE',3.7989584602272704)
        PRODUCT = request.form.get('PRODUCT',1.4676470986579082)
        AGE = request.form.get('AGE',36.037719317278004)
        RESIDENCIAL_ZIP_3 = request.form.get('RESIDENCIAL_ZIP_3',250.58099041358662)
        
        def aux(n):
            try:
                return float(n)
            except:
                return 0

        
        data =  {
                'SEX': [SEX],
                'PAYMENT_DAY' : [aux(PAYMENT_DAY)], 
                'MARITAL_STATUS' : [aux(MARITAL_STATUS)],   
                'QUANT_DEPENDANTS' : [aux(QUANT_DEPENDANTS)],   
                'NACIONALITY' : [aux(NACIONALITY)], 
                'FLAG_RESIDENCIAL_PHONE' : [aux(FLAG_RESIDENCIAL_PHONE)],   
                'RESIDENCE_TYPE' : [aux(RESIDENCE_TYPE)],   
                'MONTHS_IN_RESIDENCE' : [aux(MONTHS_IN_RESIDENCE)], 
                'PERSONAL_MONTHLY_INCOME' : [aux(PERSONAL_MONTHLY_INCOME)], 
                'OTHER_INCOMES' : [aux(OTHER_INCOMES)], 
                'HAS_ANY_CARD' : [aux(HAS_ANY_CARD)],   
                'QUANT_BANKING_ACCOUNTS' : [aux(QUANT_BANKING_ACCOUNTS)],   
                'PERSONAL_ASSETS_VALUE' : [aux(PERSONAL_ASSETS_VALUE)], 
                'QUANT_CARS' : [aux(QUANT_CARS)],   
                'FLAG_PROFESSIONAL_PHONE' : [aux(FLAG_PROFESSIONAL_PHONE)], 
                'PROFESSION_CODE' : [aux(PROFESSION_CODE)], 
                'OCCUPATION_TYPE' : [aux(OCCUPATION_TYPE)], 
                'PRODUCT' : [aux(PRODUCT)], 
                'AGE' : [aux(AGE)], 
                'RESIDENCIAL_ZIP_3' : [aux(RESIDENCIAL_ZIP_3)],       
        }
        

        test_pre = pd.DataFrame(data)
        preprocess = joblib.load('preprocess_pipeline.joblib')
        test_pre_preprocess = preprocess.transform(test_pre)
        
        loaded_model = load('/src/model_fit.joblib')
        score = loaded_model.predict_proba(test_pre_preprocess)[0]
        prediction = loaded_model.predict(test_pre_preprocess)
        
        #prediction, score = model_predict_from_form(data)
        
        # Categoría
        #--------------#
        # Etiqueta de categoría

        a1, a2, a3 = 0.3, 0.5, 0.7

        # Categorize customers based on their predicted default probabilities
        def categorize_clients(probabilities):
            categories = []
            for prob in probabilities:
                if prob > a3:
                    categories.append("Bad")
                elif a2 < prob <= a3:
                    categories.append("Regular")
                else:
                    categories.append("Good")
            return categories

        cat = categorize_clients(score)       
        #--------------#
        # Perdida esperada del cliente (función)
        #Expected Loss = Exposure at Default (EAD) * Probability of Default (PD) * Loss Given Default (LGD)

        def exp_loss(positive_class_proba):

            #Datos de un prestamo
            loan_amount = 1000 #USD/Reales
            loss_given_default = 0.5 # LGD is the Loss Given Default, the proportion of the loan that won't be recovered after default. Let's assume it's 50%.
            risk_free_rate = 0.1175  # Interest rate (3%)
            discounted_rate = 1 / (1 + risk_free_rate)
            random_clients_df = pd.DataFrame()
            # Associate probabilities with client data
            random_clients_df['probability_of_default'] = positive_class_proba

            # Calculate Expected Loss for each client based on their probability of default
            random_clients_df['expected_loss'] = loan_amount * (1-random_clients_df['probability_of_default']) * loss_given_default * discounted_rate

            # Calculate Risk Premium based on the model's prediction
            random_clients_df['risk_premium'] = random_clients_df['expected_loss']* discounted_rate

            # Calculate Total Risk Exposure
            random_clients_df['total_risk_exposure'] = random_clients_df['expected_loss'] + random_clients_df['risk_premium']

            # Display or use random_clients_df for further analysis
            #print(random_clients_df[['probability_of_default', 'expected_loss', 'risk_premium', 'total_risk_exposure']].to_string(index=False))
            print()

            # Aggregate values for all clients
            total_expected_loss = round(random_clients_df['expected_loss'].sum(), 2)
            total_risk_premium = round(random_clients_df['risk_premium'].sum(), 2)
            total_risk_exposure = round(random_clients_df['total_risk_exposure'].sum(), 2)
            total_loan = round(loan_amount * len(random_clients_df), 2)
            total_loans = len(random_clients_df) * loan_amount
            loss_percentage = round((total_expected_loss / total_loans) * 100, 2)
            risk_exposure_percentage = round((total_risk_exposure / total_loans) * 100, 2)
            
            print("Total Loan: ", total_loan)
            print("Total Expected Loss:", total_expected_loss)
            print("Total Risk Premium:", total_risk_premium)
            print("Total Risk Exposure:", total_risk_exposure)
            print("Percentage of Loss of Total Loans: {:.2f}%".format(loss_percentage))
            print("Percentage of Loss of Total Loans: {:.2f}%".format(risk_exposure_percentage))
            print()

            return total_loan,total_expected_loss,total_risk_premium,total_risk_exposure,loss_percentage, risk_exposure_percentage
        
        print("Para un prestamo de 1,000 reales al cliente, suponiendo los siguientes datos: ")
        print("Monto del prestamo: 1000" )
        print("LGD, 50%") # LGD is the Loss Given Default, the proportion of the loan that won't be recovered after default. Let's assume it's 50%.
        print("risk-free rate (3%)")
        print()
        print("Information: ","\n")
        total_loan,total_expected_loss,total_risk_premium,total_risk_exposure,loss_percentage, risk_exposure_percentage = exp_loss(score)               
        
        #--------------#
                
        context = {
            'prediction': prediction,
            "score": score,
            "data": {key: value[0] for key,value in data.items()},
            "category": cat,
            "loan": total_loan,
            "exp_loss": total_expected_loss,
            "premium" : total_risk_premium,
            "exposure": total_risk_exposure,
            "per_loss": loss_percentage,
            "per_risk": risk_exposure_percentage,
            #"comparison_df": comparison_df.to_dict(),  # Convert DataFrame to dictionary
            #"ahorro_potencial": ahorro_potencial,
            #"porcentaje_ahorro": (ahorro_potencial / (1000 * len(comparison_df))) * 100,
        }
        
          
        return render_template('index.html',target_tab_id=4,context=context)
    
    if request.method == 'GET':
        print('HOLA')
        logging.info("This is an info message")
        return render_template('index.html',target_tab_id=4)

    return f"Processed data"

         
         
         
         
         
         

         
         
         
         
         
         
         
         
         
         
         