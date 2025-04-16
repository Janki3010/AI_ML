import pandas as pa
from word2number import w2n
from sklearn import linear_model

df = pa.read_csv("salary_prediction.csv")
df.experience = df.experience.fillna("zero")

df.experience = df.experience.apply(w2n.word_to_num)
df.test_score = df.test_score.fillna(df.test_score.median())
print(df)

lr = linear_model.LinearRegression()
lr.fit(df[['experience', 'test_score', 'interview_score']], df.salary)

experience, test_score, in_score = input("Enter experience, test_score, interview_score values: ").split()
experience, test_score, in_score = int(experience), int(test_score), int(in_score)
print(f"Predicted Salary for experience {experience} test_score {test_score}, interview_score {in_score}: {lr.predict([[experience, test_score, in_score]])[0]}")

ex_coef = lr.coef_[0]
ts_coef = lr.coef_[1]
is_coef = lr.coef_[2]
reg_intercept = lr.intercept_

print("Calculation of predicted salary using formula", (ex_coef*experience) + (ts_coef*test_score) + (is_coef*in_score) + reg_intercept)

# Save Model To a File Using Python Pickle
import pickle

with open('salary_model_pickle', 'wb') as file:
    pickle.dump(lr,file)

# Load Saved Model
with open('salary_model_pickle', 'rb') as file:
    mp = pickle.load(file)

print("Predicated Salary:" ,mp.predict([[5, 8, 9]])[0])