"""
This configuration module contains constant variables,
the path to the directories, and initial settings.
"""

# Import Libraries
import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH=os.path.join(PACKAGE_ROOT, 'datasets')
SAVED_MODEL_PATH=os.path.join(PACKAGE_ROOT, 'trained_models')

TRAIN_FILE='train.csv'
TEST_FILE='test.csv'

TARGET='Loan_Status'

# Features to Keep
FEATURES=['Gender', 'Married', 'Dependents',
        'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicationIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area']

NUMERICAL_FEATURES=['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CATEGORIAL_FEATURES=['Gender', 'Married', 'Dependents', 'Education', 
                    'Self_Employed', 'Credit_History', 'Property_Area']

FEATURES_TO_ENCODE=['Gender', 'Married', 'Dependents', 'Education', 
                    'Self_Employed', 'Credit_History',
                    'Property_Area']

TEMPORAL_FEATURES=['ApplicantIncome']
TEMPORAL_ADDITION='CoapplicantIncome'
LOG_FEATURES=['ApplicantIncome', 'LoanAmount'] # Features for Log Transformation

DROP_FEATURES=['CoapplicantIncome'] # Features to Drop