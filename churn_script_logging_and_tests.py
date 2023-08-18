'''
Contain unit tests for the churn_library.py functions.
'''
import os
import logging
import numpy as np
from math import ceil
import cv2
import churn_library as cls


logging.basicConfig(
    format='%(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='./logs/churn_library.log',
    filemode='w+',
    level=logging.INFO
)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data(import_data)
        assert df.empty == False
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(path):
    '''
    test perform eda function
    '''
#     path = 'images/eda/'
    images = [
        'Churn.png',
        'Customer_Age.png',
        'Marital_Status.png',
        'Total_Trans_Ct.png',
        'Dark2_r.png']
    

    count = 0
    imgs = []
    for path in os.scandir(path):

        s = os.path.split(path)
        if path.is_file():
            image = cv2.imread(os.path.normpath(path))
            try:
                assert np.mean(image) != 255
                logging.info(
                    "SUCCESS: Testing perform_eda - image is properly written")
            except AssertionError as err:
                logging.error(
                    "ERROR: " +
                    os.path.normpath(path) +
                    " Image is empty")
                raise err
            try:
                assert s[1] in images
                imgs.append(s[1])
            except AssertionError as err:
                logging.error(
                    "ERROR: " +
                    os.path.normpath(path) +
                    " Not found")
                raise err
            count += 1

    try:
        assert count == 5
        logging.info("SUCCESS: Testing perform_eda")
    except AssertionError as err:
        logging.info("ERROR: Testing perform_eda -There are not 5 files in eda folder, "
                     + str(list(set(images).difference(imgs))) + " is missing")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    try:
        assert encoder_helper.columns.values.tolist() == keep_cols
        logging.info("SUCCESS: Testing encoder_helper")
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing encoder_helper - Dataframe' columns differ from required columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, percent):
    '''
    test perform_feature_engineering
    '''
    try:
        assert len(perform_feature_engineering) == 4
        logging.info("SUCCESS: Testing perform_feature_engineering")
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering - There should be 2 dataframes and 2 Series.")
        raise err

    try:
        assert perform_feature_engineering[2].shape[0] and perform_feature_engineering[3].shape[0] == percent
        logging.info(
            "SUCCESS: Testing perform_feature_engineering -  Test series's size are correct")
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering - Test data has not 30% from original df")
        raise err


def test_train_models(train_models):
    # def test_train_models():
    '''
    test train_models
    '''

    path = 'models/'
    count = 0
    for file in os.scandir(path):
        if file.is_file():
            count += 1

    # test for 2 models
    try:
        assert count == 2
        logging.info(
            "SUCCESS: Testing train_models - 2 models have been saved")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing train_models - There are not 2 models in models folder")
        logging.error("There are " + str(count) + " files")
        raise err

    # test for images/results/roc_curve_result image
    pth = 'images/results/'
    try:
        assert os.path.isfile(pth + 'roc_curve_result.png')
        logging.info(
            "SUCCESS: Testing train_models - roc_curve_result image have been saved in images/results")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing train_models - lrc_roc_curve_resultplot image has not been found in images/results")
        raise err

    # test results/feature_importance image
    try:
        assert os.path.isfile(pth + 'feature_importance.png')
        logging.info(
            "SUCCESS: Testing train_models - feature_importance image have been saved in images/results")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing train_models - feature_importance image has not been saved in images/results")
        raise err

    # test for two classification_report images inside results folder
    try:
        assert os.path.isfile(
            pth +
            'lr_classification_report.png') and os.path.isfile(
            pth +
            'rf_classification_report.png')
        logging.info(
            "SUCCESS: Testing train_models - feature_importance image have been saved in images/results")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing train_models - One of the classification report images has not been found inside images/results folder")
        raise err


if __name__ == "__main__":

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    path = 'data/bank_data.csv'

    # test import_data
    test_import(path)

    # test perform_eda   
    test_eda('images/eda/')


    # test encoder_helper
    df = cls.import_data(path)
    x = cls.encoder_helper(df, cat_columns, response=None)
    test_encoder_helper(x)

    # test perform_feature_engineering
    y = df['Churn']
    percent = ceil(x.shape[0] * 0.3)
    test_perform_feature_engineering(
        cls.perform_feature_engineering(x, y),
        percent)
    
    # test train_models
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(x, y)
    test_train_models(cls.train_models(X_train, X_test, y_train, y_test))
