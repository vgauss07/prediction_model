"""
This module fetches a 
single record from the 
validation data and verifies 
the output using assert statements

It validates the following checks:
    - the output is not null.
    - the output datatype is str
    - the output is Y for given data (fixed)
"""

# Import Libraries
import pytest


# Import files/modules
from prediction_model.config import config
from prediction_model.preocessing.data_management import load_dataset
from prediction_model.predict import make_prediction

@pytest.fixture
def single_prediction():
    '''This function will predict the result of a single record'''
    test_data = load_dataset(file_name=config.TEST_FILE)
    singe_test = test_data[0:1]
    result = make_prediction(single_test)
    return result

# Test Prediction
def test_single_prediction_not_none(single_prediction):
    ''' This function will check if the result of prediction is not None'''
    assert single_prediction is not None

def test_single_prediction_dtype(single_prediction):
    '''This function will check if the data type of the prediction 
    result is string'''
    assert isinstance(single_prediction('prediction')[0], str)

def test_single_prediction_output(single_prediction):
    '''This function will check if the result of the prediction
     is Y'''
    assert single_prediction.get('prediction')[0] == 'Y'
    