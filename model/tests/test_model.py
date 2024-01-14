import unittest
import Credit_Risk_Analysis.model.ml_serviceA as ml_serviceA
import pandas as pd

class TestMLService(unittest.TestCase):
    def test_predict(self):
        ml_serviceA.settings.UPLOAD_FOLDER = "tests/"        
        dict_features = {'PAYMENT_DAY': [14], 'SEX': ['M'], 'MARITAL_STATUS': [0.32316679991959296], 'QUANT_DEPENDANTS': [1.9077678811444132], 'NACIONALITY': [0.22034757035662245], 'FLAG_RESIDENCIAL_PHONE': [1], 'RESIDENCE_TYPE': [1.7495547046722986], 'MONTHS_IN_RESIDENCE': [48.20525143724116], 'PERSONAL_MONTHLY_INCOME': [896524.4425945266], 'OTHER_INCOMES': [317.9287421872096], 'QUANT_BANKING_ACCOUNTS': [0.22551415109553608], 'PERSONAL_ASSETS_VALUE': [8712.210598266734], 'QUANT_CARS': [1.3497474993388459], 'FLAG_PROFESSIONAL_PHONE': [1], 'PROFESSION_CODE': [5.172599117336276], 'OCCUPATION_TYPE': [3.7989584602272704], 'PRODUCT': [1.4676470986579082], 'AGE': [36.037719317278004], 'RESIDENCIAL_ZIP_3': [250.58099041358662], 'HAS_ANY_CARD': [0]}
        df = pd.DataFrame(dict_features)   
        class_name, pred_probability = ml_serviceA.predict(df)        
        self.assertAlmostEqual(class_name, 0, 5)   
        self.assertAlmostEqual(pred_probability, 0.6979019834188147, 5)       

if __name__ == "__main__":
    unittest.main()
