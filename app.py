import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def home():
    data = {
        'Years_of_Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Age': [23, 25, 28, 30, 32, 35, 38, 40, 45, 50],
        'Education_Level': [1, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        'Salary': [35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
    }

    df = pd.DataFrame(data)
    print(df)

    X = df[['Years_of_Experience', 'Age', 'Education_Level']]
    y = df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predicted_salary = 0
    if request.method == 'POST':
        Years_of_Experience = request.form.get('Years_of_Experience')
        Years_of_Experience = float(Years_of_Experience)
        Age = request.form.get('age')
        Education_Level = request.form.get('Education_Level')

        data = {
            'Years_of_Experience': [Years_of_Experience],
            'Age': [Age],
            'Education_Level': [Education_Level]
        }
        new_data = pd.DataFrame(data)
        predicted_salary = model.predict(new_data)
        print(f'Predicted salary for new data:{predicted_salary[0]}', )

    return render_template('index.html', salary=predicted_salary[0])


if __name__ == '__main__':
    app.run()

#
# data = {
#     'Years_of_Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'Age': [23, 25, 28, 30, 32, 35, 38, 40, 45, 50],
#     'Education_Level': [1, 2, 2, 3, 3, 3, 3, 3, 3, 3],
#     'Salary': [35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
# }
#
# df = pd.DataFrame(data)
#
# X = df[['Years_of_Experience', 'Age', 'Education_Level']]
# y = df['Salary']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)

# Years_of_Experience = int(input('Enter Years_of_Experience:'))
# Age = int(input("Enter Age:"))
# Education_Level = int(input("Enter Education_Level:"))
# data = {
#     'Years_of_Experience': [Years_of_Experience],
#     'Age': [Age],
#     'Education_Level': [Education_Level]
# }
# new_data = pd.DataFrame(data)
# predicted_salary = model.predict(new_data)
# print(f'Predicted salary for new data:{predicted_salary[0]}', )

