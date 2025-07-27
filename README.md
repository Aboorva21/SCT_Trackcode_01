# SCT_Trackcode_01
A simple machine learning project that uses Linear Regression to predict house prices based on square footage, number of bedrooms, and number of bathrooms. Built using Python, Pandas, and Scikit-learn with visualization support from Matplotlib and Seaborn.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Sample data (You can replace this with your CSV dataset)
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'bathrooms': [2, 2, 3, 3, 4],
    'price': [300000, 360000, 400000, 500000, 550000]
}

df = pd.DataFrame(data)
print(df.head())
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()
X = df[['square_feet', 'bedrooms', 'bathrooms']]  # features
y = df['price']  # label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
new_house = np.array([[2500, 4, 3]])  # [sqft, bedrooms, bathrooms]
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

