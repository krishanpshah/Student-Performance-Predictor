import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Student_Performance.csv')
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes':1,'No':0})

X = data.drop('Performance Index', axis=1)
y = data['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

regressor = LinearRegression()

regressor.fit(X_train,y_train)



pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')