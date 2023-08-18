# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

In this project, we are going to identify credit card customers that are most likely to churn. It will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). 

This project also provides testing and logging to make sure functions are working as expected. 


## Files and data description

This project is using data ectracted from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).  It consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, among other fields.



The structure of this project is as follow:
```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── empty
│   └── results
│       ├── feature_importances.png
│       ├── lr_classification_report.png
│       ├── rf_classification_report.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── empty
├── __pycache__
│   ├── churn_library2.cpython-36.pyc
│   └── churn_library.cpython-36.pyc
├── README.md
└── requirements.txt
```


## Running Files
This project has 2 python files that can be run.

In order to run these files, the first we need to do is to install necessary packages by running:

python -m pip install -r requirements_py3.6.txt


1. churn_library.py - Is a library of functions to find customers who are likely to churn. 

- This file can be run typing in terminal: python churn_library.py


After running churn_library.py the structure of the project should look like this:
```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Gender.png
│   │   ├── heatmap.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Amt.png
│   └── results
│       ├── feature_importances.png
│       ├── lr_classification_report.png
│       ├── rf_classification_report.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── __pycache__
│   ├── churn_library2.cpython-36.pyc
│   └── churn_library.cpython-36.pyc
├── README.md
└── requirements.txt
```

1. churn_script_logging_and_tests.py - Contain unit tests for the churn_library.py functions and logs any errors and INFO messages. 

- This file can be run typing in terminal: python churn_script_logging_and_tests.py





