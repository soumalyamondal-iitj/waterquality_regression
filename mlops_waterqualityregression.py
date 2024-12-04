"""### Part 1: Data Collection and Preprocessing

#### Step 1: Load the Dataset
"""

import pandas as pd


# Load the dataset
data = pd.read_csv("water_potability.csv")

data.head()

"""#### Step 2: Clean and Preprocess the Data"""

# Handle missing values by filling with the median of each column
data.fillna(data.median(), inplace=True)

# Display basic statistics
data.describe()

"""#### Step 3: Perform Exploratory Data Analysis (EDA)

"""

import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()

# Pairplot for a subset of features
sns.pairplot(data[["ph", "Hardness", "Solids", "Chloramines", "Sulfate"]])
plt.show()

"""### Part 2: Model Implementation

#### Step 1: Split the Data into Training and Testing Sets
"""

from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop(columns=["Potability"])
y = data["Potability"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#### Step 2: Implement Ridge and Lasso Regression

"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV


# Define the parameter grid for Ridge and Lasso
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Ridge Regression (L2 Penalty)
ridge = LogisticRegression(penalty='l2', solver='liblinear')
ridge_grid = GridSearchCV(ridge, param_grid, cv=5, scoring='accuracy')
ridge_grid.fit(X_train, y_train)

# Lasso Regression (L1 Penalty)
lasso = LogisticRegression(penalty='l1', solver='liblinear')
lasso_grid = GridSearchCV(lasso, param_grid, cv=5, scoring='accuracy')
lasso_grid.fit(X_train, y_train)

# Best parameters
best_ridge_C = ridge_grid.best_params_['C']
best_lasso_C = lasso_grid.best_params_['C']

# Best models
best_ridge_model = ridge_grid.best_estimator_
best_lasso_model = lasso_grid.best_estimator_

# Make predictions
ridge_pred = best_ridge_model.predict(X_test)
lasso_pred = best_lasso_model.predict(X_test)

# Evaluate the models
ridge_accuracy = accuracy_score(y_test, ridge_pred)
lasso_accuracy = accuracy_score(y_test, lasso_pred)

print(f"Ridge Regression - Best C: {best_ridge_C}, Accuracy: {ridge_accuracy}")
print(f"Lasso Regression - Best C: {best_lasso_C}, Accuracy: {lasso_accuracy}")


### Part 3: MLflow Integration with Hyperparameter Tuning


import mlflow
import mlflow.sklearn

# Set the experiment name
mlflow.set_experiment("Water_Quality_Regression")

with mlflow.start_run(run_name="Ridge vs Lasso Regression with Hyperparameter Tuning") as run:
    # Log best parameters
    mlflow.log_param("Ridge_best_C", best_ridge_C)
    mlflow.log_param("Lasso_best_C", best_lasso_C)
    
    # Log metrics
    mlflow.log_metric("Ridge_Accuracy", ridge_accuracy)
    mlflow.log_metric("Lasso_Accuracy", lasso_accuracy)
    
    # Log models
    mlflow.sklearn.log_model(best_ridge_model, "Best_Ridge_Model")
    mlflow.sklearn.log_model(best_lasso_model, "Best_Lasso_Model")

    # Log artifacts (e.g., model files, figures)
    plt.savefig("correlation_heatmap.png")
    mlflow.log_artifact("correlation_heatmap.png")

