# Preprocess
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# Function to determine if any card is present
def has_card(row):
  return 1 if row['FLAG_VISA'] == 1 or row['FLAG_MASTERCARD'] == 1 or row['FLAG_DINERS'] == 1 or row['FLAG_AMERICAN_EXPRESS'] == 1 or row['FLAG_OTHER_CARDS'] == 1 else 0

def clean_data_predict(df,names):
  col_names = names['Var_Title'].to_list()
  col_names[43] = 'EDUCATION_LEVEL_2'
  col_names[53] = 'TARGET'
  df.columns = col_names

  # Columns to eliminate
  columns_to_exclude = ['QUANT_ADDITIONAL_CARDS', 'EDUCATION_LEVEL', 'FLAG_HOME_ADDRESS_DOCUMENT',
                          'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF','FLAG_EMAIL',
                          'CLERK_TYPE','POSTAL_ADDRESS_TYPE','FLAG_MOBILE_PHONE',
                          'COMPANY','MONTHS_IN_THE_JOB','MATE_PROFESSION_CODE','EDUCATION_LEVEL_2',
                          'FLAG_ACSP_RECORD','PROFESSIONAL_CITY','PROFESSIONAL_BOROUGH','PROFESSIONAL_STATE',
                          'PROFESSIONAL_ZIP_3','QUANT_SPECIAL_BANKING_ACCOUNTS',
                          'PROFESSIONAL_PHONE_AREA_CODE','RESIDENCIAL_PHONE_AREA_CODE']
  filtered_df = df.drop(columns=columns_to_exclude)
  # Columns to balance
  columns_to_balance = ['PROFESSION_CODE', 'OCCUPATION_TYPE','RESIDENCE_TYPE','MONTHS_IN_RESIDENCE']
  for column in columns_to_balance:
      filtered_df[column] = filtered_df[column].replace(0, np.nan)
      mean_value = filtered_df[column].mean()
      filtered_df[column] = filtered_df[column].fillna(round(mean_value,0))
  #Error in this columns
  filtered_df['RESIDENCIAL_ZIP_3'] = filtered_df['RESIDENCIAL_ZIP_3'].replace('#DIV/0!', 0)
  filtered_df['RESIDENCIAL_ZIP_3'] = filtered_df['RESIDENCIAL_ZIP_3'].astype(int)
  #Balance age
  valid_age_data = filtered_df[(filtered_df['AGE'] > 14) & (filtered_df['AGE'] < 100)]

  #Clean SEX
  valid_age_data['SEX'] = valid_age_data['SEX'].replace({' ': 'N'})
  valid_df = valid_age_data[valid_age_data['SEX'] != 'N']

  # Create a new column 'HAS_ANY_CARD' based on the conditions
  valid_df['HAS_ANY_CARD'] = valid_df.apply(has_card, axis=1)

  # Drop individual card columns if needed
  valid_df = valid_df.drop(['FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS'], axis=1)

  #Upper categories
  upper_columns = ['RESIDENCIAL_BOROUGH', 'CITY_OF_BIRTH', 'RESIDENCIAL_BOROUGH']
  for col in upper_columns:
      valid_df[col] = valid_df[col].apply(lambda x: x.upper() if isinstance(x, str) else x)

  #Replace some values
  valid_df['CITY_OF_BIRTH'] = valid_df['CITY_OF_BIRTH'].replace({' ': 'NULL'})
  valid_df['STATE_OF_BIRTH'] = valid_df['STATE_OF_BIRTH'].replace({' ': 'NULL'})
  valid_df['FLAG_RESIDENCIAL_PHONE'] = valid_df['FLAG_RESIDENCIAL_PHONE'].replace({'Y':1,'N':0})
  valid_df['FLAG_PROFESSIONAL_PHONE'] = valid_df['FLAG_PROFESSIONAL_PHONE'].replace({'Y':1,'N':0})

  valid_df = valid_df.drop(['ID_CLIENT','APPLICATION_SUBMISSION_TYPE','STATE_OF_BIRTH', 'CITY_OF_BIRTH', 'RESIDENCIAL_CITY','RESIDENCIAL_STATE','RESIDENCIAL_BOROUGH'],axis = 1)

  return valid_df

class DataFrameSelector(BaseEstimator, TransformerMixin):
  def __init__(self, attribute_names):
      self.attribute_names = attribute_names

  def fit(self, X, y=None):
      return self

  def transform(self, X):
      return X[self.attribute_names].values

def preprocess_data(train_df, test_df):
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train_df.select_dtypes(exclude=['object']).columns.tolist()

    # Initialize lists for binary and non-binary categorical columns
    binary_categorical_cols = []
    non_binary_categorical_cols = []

    # Iterate through the categorical columns
    for col in categorical_cols:
        unique_values = train_df[col].nunique()
        if unique_values == 2:
            binary_categorical_cols.append(col)
        else:
            non_binary_categorical_cols.append(col)
    
    # Define the pipeline steps
    binary_categorical_pipeline = Pipeline([
        ('selector', DataFrameSelector(attribute_names=binary_categorical_cols)),
        ('ordinal_encoder', OrdinalEncoder())
    ])

    non_binary_categorical_pipeline = Pipeline([
        ('selector', DataFrameSelector(attribute_names=non_binary_categorical_cols)),
        ('onehot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

    numerical_pipeline = Pipeline([
        ('selector', DataFrameSelector(attribute_names=numerical_cols)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine all pipelines using ColumnTransformer
    preprocess_pipeline = ColumnTransformer([
        ('binary_categorical', binary_categorical_pipeline, binary_categorical_cols),
        ('non_binary_categorical', non_binary_categorical_pipeline, non_binary_categorical_cols),
        ('numerical', numerical_pipeline, numerical_cols)
    ])

    train_processed = preprocess_pipeline.fit_transform(train_df)
    test_processed = preprocess_pipeline.transform(test_df)

    # Convert the processed arrays back to DataFrames
    train_processed_df = pd.DataFrame(train_processed, columns=binary_categorical_cols + non_binary_categorical_cols + numerical_cols)
    test_processed_df = pd.DataFrame(test_processed, columns=binary_categorical_cols + non_binary_categorical_cols + numerical_cols)

    print(train_processed_df.shape)
    print(test_processed_df.shape)

    return train_processed_df, test_processed_df
