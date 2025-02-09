"""
Predicting golfer performance: Using Scikit-learn to build a linear regression model predicting golf performance

PGA Tour Golf Data - (2015-2022)
https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022?resource=download

https://github.com/upjohnc/Golf-Statistics/blob/master/data/PGA%20Stats.csv
"""

# 1. Data collection and prep
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('pga_tour_stats.csv')

# 2. Data exploration and preprocessing
print(data.head())
print(data.describe())

# Select features and target variable
features = ['Driving_Distance', 'Driving_Accuracy', 'Greens_in_Regulation', 'Putts_per_Round']
target = 'Scoring_Average'

X = data[features]
y = data[target]


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Build and train model: Create a linear regression model and train it on data
model = LinearRegression()
model.fit(X_train, y_train)


# 5. Make predictions and evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# 6. Visualize results: Create a scatter plot of predicted vs. actual scores
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scoring Average")
plt.ylabel("Predicted Scoring Average")
plt.title("Actual vs. Predicted Scoring Average")
plt.show()

# 7. Interpret model: Analyze coefficients to understand the features having the most significant impact on scoring average
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")


"""
1. Try different ML algorithms (e.g., Random Forest, Gradient Boosting)
2. Include more features or engineer new ones
3. Analyze data from multiple seasons to identify trends
4. Create a simple web app to allow users to input stats and get predictions
"""