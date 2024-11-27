import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Size': [1000, 1500, 1800, 2000, 2500, 3000, 1200, 1400, 1600, 2200, 2400, 2800, 3500, 4000, 4500],
    'Rooms': [3, 3, 4, 4, 5, 5, 3, 3, 4, 5, 5, 6, 6, 6, 7],
    'Location': [1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2],
    'Price': [250000, 300000, 350000, 400000, 500000, 600000, 270000, 320000, 360000, 440000, 480000, 550000, 700000,
              750000, 800000]
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

print(df)

# Features (X) and target (y)
X = df[['Size', 'Rooms', 'Location']]  # Independent Variables
y = df['Price']  # Dependent Variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""test_size=0.2: 20% of the data is used for testing, and 80% for training.  
random_state=42 ensuring consistent results each time you run the code."""

# Create a Linear Regression model
model = LinearRegression()
# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

Size = int(input("Enter size of the house:"))
Rooms = int(input("Enter the total rooms in the house:"))
Location = int(input("Enter the Location:"))

data = {
    'Size': [Size],
    'Rooms': [Rooms],
    'Location': [Location]
}
new_data = pd.DataFrame(data)
predicted_price = model.predict(new_data)
print(f'Predicted Price for {Size}, {Rooms}, {Location}: {predicted_price[0]}')

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
