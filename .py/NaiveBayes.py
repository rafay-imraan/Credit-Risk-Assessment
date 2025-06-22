# CREDIT RISK ASSESSMENT USING NAIVE BAYES (GaussianNB)

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Map CSV to a DataFrame
df = pd.read_csv("credit_risk_dataset.csv")

# Converting Y and N to 1 and 0
df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({'Y': 1, 'N': 0})

# Declare labels
X = df.drop(["loan_status"], axis = 1)
y = df["loan_status"]

# Dropping rows with NaNs
X = X.dropna()
y = y[X.index]

# Hot-Encoding non-binary values
X = pd.get_dummies(X, columns = ["person_home_ownership", "loan_intent"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the results
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))