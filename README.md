# Titanic Survival Prediction 🚢

## Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning.
The model is trained on the Titanic dataset and uses passenger information such as age, gender, and ticket class to predict survival.

This project demonstrates the basic workflow of a Machine Learning project including data preprocessing, model training, and evaluation.

## Dataset

Dataset used: Titanic passenger dataset.

Features include:

* PassengerId
* Pclass (Passenger class)
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked

Target Variable:

* Survived (0 = Did not survive, 1 = Survived)

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Machine Learning Model

Logistic Regression was used to train the classification model to predict passenger survival.

## Project Workflow

1. Load the dataset
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature selection and encoding
5. Train-test split
6. Model training using Logistic Regression
7. Model evaluation using accuracy score

## Results

The model predicts whether a passenger survived or not based on the input features.

Evaluation metrics used:

* Accuracy Score
* Confusion Matrix

## Project Structure

Titanic-Survival-Prediction
│
├── data
│   └── titanic.csv
│
├── titanic_model.py
|
└── README.md

