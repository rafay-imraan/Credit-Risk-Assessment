# Credit Risk Assessment
This repository includes code for machine learning models that classify whether a person is high(1) or low(0) credit risk. Each model was trained using one of the following algorithms:

- Decision Trees
- Naïve Bayes Classifier
- Logistic Regression
- KNNs (K-Nearest Neighbors)
- SVMs (Support Vector Machines)
- Random Forest Classifier
- XGBoost

The code was written entirely in [Python](https://python.org) version 3.13.2 on Jupyter Notebook. All of these algorithms, excluding XGBoost, were imported from [scikit-learn](https://www.scikit-learn.org). Other scikit-learn functions used here include [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), and [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html). [Pandas](https://pandas.pydata.org/) and [xgboost](https://xgboost.ai) was also used in this project. All of the models perform on the same dataset downloaded from [kaggle.com](https://www.kaggle.com).

The code outputs the measure of the accuracy of the predictions made, along with a classification report of the entire training/testing session. The number of high and low credit risks predicted is displayed in the support column.

**MAKE SURE THE AFOREMENTIONED LIBRARIES ARE INSTALLED BEFORE EXECUTING THE CODE.**
