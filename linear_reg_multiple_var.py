# Linear Regression Variable
from webbrowser import open_new

import pandas as pd
from sklearn import linear_model

df = pd.read_csv("houseprices.csv")

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
print(df)

lr = linear_model.LinearRegression()
lr.fit(df[['area', 'bedrooms', 'age']], df.price)

area, bedrooms, age = input("Enter area, bedrooms, age values: ").split()
area, bedrooms, age = float(area), int(bedrooms), int(age)

input_data = pd.DataFrame([[area, bedrooms, age]], columns=['area', 'bedrooms', 'age'])
prediction = lr.predict(input_data)[0]
print(f"Predicted price for area {area} bedrooms {bedrooms}, age {age}: {prediction}")

area_coef = lr.coef_[0]
bedroom_coef = lr.coef_[1]
age_coef = lr.coef_[2]
reg_intercept = lr.intercept_

print("Calculated house predicted price using formula:",(area_coef*area) + (bedroom_coef*bedrooms) + (age_coef*age) + reg_intercept)

# Save Model Using Pickle
import pickle
with open("house_model_pickle", "wb") as f:
    pickle.dump(lr, f)

# Load Saved Model (U can use this in any file)
with open("house_model_pickle", "rb") as file:
    model = pickle.load(file)

input_data = pd.DataFrame([[2000, 3, 30]], columns=['area', 'bedrooms', 'age'])
prediction = model.predict(input_data)[0]
print("Predicted House Price:", prediction)
