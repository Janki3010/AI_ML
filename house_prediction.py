import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("house_pr.csv")
print(df)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show()

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

print("Predicted Price:", reg.predict([[3300]])[0])
reg_coef = reg.coef_[0]
reg_intercept = reg.intercept_

""" Formula to find predicted price
Y = m * X + b (X is area , m is coefficient and b is intercept) """
print(reg_coef*3300+reg_intercept)

# Generate CSV file with list of home price predication
area_df = pd.read_csv("area.csv")
area_df.head(3)

p = reg.predict(area_df)
area_df['price'] = p

area_df.to_csv("prediction.csv")
